"""R-KV integration layer for SGLang v0.5.14.

Bridges the pure, device-agnostic R-KV algorithm (:mod:`.algo`) to SGLang's
paged KV cache and the FlashInfer decode path. Read ``R-KV/doc/DESIGN.md``
(especially
sections 5-9) for the architecture rationale behind the decisions encoded here.

Scope of this module (phase 1, correctness first):

* FlashInfer backend, ``batch >= 1`` with **per-request triggering**: each
  request independently decides when to compress based on its own KV length
  (see :meth:`RKVCompressor.observe_decode_layer`).
* ``page_size == 1`` token pool, so evicted slots free cleanly one-by-one
  (the paged allocator frees at page granularity; see R-KV/doc/DESIGN.md
  section 8).
* Reduction of the algorithm's per-head / per-layer scores into a single global
  per-token eviction decision:

  - cross-head:  **mean** of the per-head joint scores.
  - cross-layer: **sum** over every layer's scores (aggregate all layers).

* Trigger cadence: compress once every ``buffer_size`` generated tokens, and
  only once the request's KV length has reached ``budget``.
* Physical compaction runs **after the full forward pass** (all layers have
  written their KV), not inside a single layer's ``forward_decode``.

The module is deliberately standalone: it imports only :mod:`.algo` and
``torch`` at module load time, so it can be unit-tested against mock pools
without pulling in the heavy serving stack.

Wiring into the runtime (``forward_decode`` hook, ``model_runner`` after-forward
call, ``scheduler`` request begin/end) is intentionally **not** done in this
file yet -- those touch core files and need on-GPU validation. See the TODOs at
the bottom and the roadmap in ``R-KV/doc/DESIGN.md`` section 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.mem_cache.rkv.algo import R1KV

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy imports
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class RKVConfig:
    """Hyper-parameters for R-KV compression.

    ``budget=1024`` is a serving-oriented default (the R-KV reference class and
    its HF eval scripts use ``budget=128``); ``buffer_size`` is the staging
    buffer ``B_buffer`` that controls how often compression fires. Note the
    reference HF eval scripts run ``mix_lambda=0.1, retain_ratio=0.2`` while the
    reference class defaults are ``mix_lambda=0.07, retain_ratio=0.1``; override
    via the ``--rkv-*`` flags / ``--rkv-config`` to match a specific
    configuration exactly.
    """

    budget: int = 1024
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.1
    retain_ratio: float = 0.1
    retain_direction: str = "last"
    # Compress every ``buffer_size`` newly generated tokens per request.
    buffer_size: int = 128
    # Minimum KV length before compression is ever considered. Defaults to
    # ``budget`` (there is nothing to drop below budget).
    min_seq_len: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_seq_len is None:
            self.min_seq_len = self.budget
        assert self.budget > self.window_size, "budget must exceed window_size"
        assert self.buffer_size >= self.window_size, (
            "buffer_size must be >= window_size, otherwise the first compaction "
            "scores against zero-initialized queries in the observation window"
        )
        assert (
            self.min_seq_len >= self.budget
        ), "min_seq_len must be >= budget (select_indices keeps budget tokens)"


class RKVRequestState:
    """Per-request bookkeeping the integration layer maintains.

    One instance per active request, keyed by ``req_pool_idx`` in
    :class:`RKVCompressor`. Holds the trigger counter, the per-layer observation
    query window (R-KV needs the last ``window_size`` queries, and queries are
    per-layer), and the running cross-layer score accumulator used during a
    compaction step.
    """

    def __init__(
        self,
        req_pool_idx: int,
        num_layers: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.req_pool_idx = req_pool_idx
        self.num_layers = num_layers
        self.window_size = window_size
        self.device = device
        self.dtype = dtype

        # Generated steps since the last compaction (trigger cadence counter),
        # advanced once per decode step by ``RKVCompressor.begin_decode_step``
        # (which runs every step, unlike ``observe_decode_layer`` which is
        # skipped on CUDA-graph-replayed steps).
        self.steps_since_compact = 0

        # Logical absolute position of the request's next token (vestigial:
        # decode positions are derived from the request in
        # ``override_decode_positions``, not from this field).
        self.next_position = 0

        # Per-layer observation-window queries, shape
        # (num_layers, window_size, q_head_num, head_dim). Filled over the
        # ``window_size`` decode steps ending at a compaction; those steps are
        # forced to run eager so this hook actually executes. Allocated lazily.
        self.window_q: Optional[torch.Tensor] = None
        # Slot in ``window_q`` to write this step's query into (set by
        # ``begin_decode_step``); -1 means this step is not a window step.
        self.this_step_window_slot: int = -1

        # Transient cross-layer per-token score accumulator, allocated lazily at
        # the start of a compaction step and cleared afterwards.
        self.score_accum: Optional[torch.Tensor] = None

        # Back-reference to the owning request, so compaction can update its
        # physical-length bookkeeping (kv_committed_len / kv_allocated_len).
        # Duck-typed: only needs kv_committed_len / kv_allocated_len /
        # origin_input_ids / output_ids. Set in ``on_request_begin``.
        self.req: Optional[Req] = None

    def write_window(self, layer_idx: int, q: torch.Tensor) -> None:
        """Write this step's query into the observation window.

        ``q`` is ``(q_head_num, head_dim)``. No-op unless this step is a window
        step (``this_step_window_slot >= 0``, set by ``begin_decode_step``). The
        per-layer buffer is allocated lazily from ``q``'s shape. Temporal order
        is guaranteed by the slot index: slot 0 is the oldest window token,
        ``window_size - 1`` the newest (the compaction step itself).
        """
        if self.this_step_window_slot < 0:
            return
        if self.window_q is None:
            q_head_num, head_dim = q.shape[-2], q.shape[-1]
            self.window_q = torch.zeros(
                (self.num_layers, self.window_size, q_head_num, head_dim),
                device=self.device,
                dtype=q.dtype,
            )
        self.window_q[layer_idx, self.this_step_window_slot].copy_(q)


class RKVCompressor:
    """Coordinates R-KV compression across a request's lifetime.

    Borrows the lifecycle-hook naming from ``mem_cache/sparsity`` but shares no
    code with it (that framework is non-destructive; R-KV truly evicts and frees
    slots -- see R-KV/doc/DESIGN.md section 8).
    """

    def __init__(
        self,
        config: RKVConfig,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        kv_allocator: BaseTokenToKVPoolAllocator,
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.kv_allocator = kv_allocator
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers = end_layer - start_layer
        self.device = device

        self.algo = R1KV(
            budget=config.budget,
            window_size=config.window_size,
            kernel_size=config.kernel_size,
            mix_lambda=config.mix_lambda,
            retain_ratio=config.retain_ratio,
            retain_direction=config.retain_direction,
        )

        # Active per-request state, keyed by req_pool_idx.
        self.states: Dict[int, RKVRequestState] = {}
        # req_pool_idx values that armed a compaction this forward pass.
        self._armed: set[int] = set()
        # req_pool_idx -> new physical KV length after the latest compaction.
        # The scheduler drains this (``take_pending_length_updates``) to update
        # the batch-level seq_lens tensors, which it owns.
        self.pending_length_updates: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    def on_request_begin(self, req: Req) -> None:
        """Register a request and initialise its R-KV state."""
        if req.req_pool_idx is None:
            return
        dtype = self.token_to_kv_pool.get_key_buffer(self.start_layer).dtype
        state = RKVRequestState(
            req_pool_idx=req.req_pool_idx,
            num_layers=self.num_layers,
            window_size=self.config.window_size,
            device=self.device,
            dtype=dtype,
        )
        state.next_position = len(req.origin_input_ids)
        state.req = req
        self.states[req.req_pool_idx] = state

    def on_request_end(self, req: Req) -> None:
        """Drop a request's R-KV state when it finishes or aborts."""
        if req.req_pool_idx is not None and self.states.pop(req.req_pool_idx, None):
            logger.debug(
                "R-KV on_request_end req_pool_idx=%d states_left=%d",
                req.req_pool_idx,
                len(self.states),
            )

    def on_request_retract(self, req: Req) -> None:
        """Discard a retracted request's R-KV state.

        Retraction frees the request's physical KV and sends it back to the
        waiting queue to be re-prefilled from scratch. Its old state (compaction
        counter, observation-window queries, physical-length bookkeeping) no
        longer matches the freed KV, so drop it here; a clean state is rebuilt by
        ``on_request_begin`` when the request re-enters decode. Must be called
        while ``req.req_pool_idx`` is still valid, i.e. BEFORE the pool frees it.
        """
        if req.req_pool_idx is not None:
            self.states.pop(req.req_pool_idx, None)

    # ------------------------------------------------------------------
    # Scheduler admission support
    # ------------------------------------------------------------------
    def admission_reserve(self, prompt_len: int, occupied: int) -> int:
        """Upper bound on the *future* physical KV a request can still consume.

        R-KV holds a request's physical KV cache at a constant ceiling no matter
        how many tokens it will still generate: it lets the length grow to at
        most ``max(prompt_len, min_seq_len) + buffer_size`` (the peak reached the
        step just before a compaction) and then frees back down to ``budget``.
        So the scheduler only needs to reserve ``ceiling - occupied`` for a
        request, NOT its full remaining ``max_new_tokens``. Reserving this (much
        smaller) bound is what lets many more R-KV requests run concurrently.

        The bound is deliberately the pre-compaction PEAK, so it is never an
        underestimate: a request's physical KV can never exceed it, hence
        admission can never over-commit the pool (memory-safe). ``occupied`` is
        the request's current physical KV length (its committed length, or its
        prompt length before prefill).
        """
        ceiling = max(prompt_len, self.config.min_seq_len) + self.config.buffer_size
        return max(0, ceiling - occupied)

    # ------------------------------------------------------------------
    # Decode-time observation (called per layer from forward_decode)
    # ------------------------------------------------------------------
    def begin_decode_step(self, forward_batch: ForwardBatch) -> bool:
        """Advance per-request decode counters and decide graph-vs-eager.

        Called once per decode step in ``ModelRunner.forward`` BEFORE the
        graph/eager decision, so it runs on EVERY step — including steps that
        will replay a captured CUDA graph, where ``observe_decode_layer`` (a
        Python hook inside ``forward_decode``) is skipped.

        For each managed request it advances ``steps_since_compact`` (only once
        the request is long enough to compress, ``seq_len >= min_seq_len``),
        marks whether this step is one of the ``window_size`` steps ending at the
        next compaction (setting ``this_step_window_slot`` so the eager
        ``observe`` writes its query), and arms the compaction on the final
        window step.

        Returns True if ANY request needs this step to run eager (a window step
        or the compaction step). The caller disables the CUDA graph for this
        step so ``observe_decode_layer`` / ``maybe_compact`` execute; all other
        steps replay the graph.
        """
        if not self.states:
            return False
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens_src = forward_batch.seq_lens_cpu
        if seq_lens_src is None:
            seq_lens_src = forward_batch.seq_lens
        seq_lens = seq_lens_src.tolist()

        window = self.config.window_size
        buffer = self.config.buffer_size
        first_window_step = buffer - window + 1  # steps_since_compact of slot 0
        need_eager = False
        for i, req_pool_idx in enumerate(req_indices):
            state = self.states.get(int(req_pool_idx))
            if state is None:
                continue
            state.next_position += 1
            state.this_step_window_slot = -1
            # Only start the compaction clock once the request can actually be
            # compressed; below budget there is nothing to evict.
            if int(seq_lens[i]) < self.config.min_seq_len:
                continue
            state.steps_since_compact += 1
            steps = state.steps_since_compact
            if steps >= first_window_step:
                slot = steps - first_window_step
                if 0 <= slot < window:
                    state.this_step_window_slot = slot
                    need_eager = True
                if steps >= buffer:
                    # Compaction step: arm scoring (accumulated in observe over
                    # this same eager forward, then consumed by maybe_compact).
                    self._armed.add(int(req_pool_idx))
                    state.score_accum = None
        return need_eager

    def observe_decode_layer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> None:
        """Capture window queries and (on the compaction step) accumulate scores.

        Called per layer from ``FlashInferAttnBackend.forward_decode`` *after*
        ``set_kv_buffer`` — but only on EAGER decode steps (graph-replayed steps
        skip this Python hook; ``begin_decode_step`` forces window/compaction
        steps to run eager so this executes for them). Counters and arming are
        owned by ``begin_decode_step``; here we only:
          * write this layer's query into the observation window (window steps);
          * on the compaction step (armed), compute + accumulate the layer score.
        """
        # No managed requests => nothing to do. This early return is also what
        # keeps CUDA-graph CAPTURE clean: capture runs dummy decode forwards with
        # an empty ``states``, and returning before the ``.tolist()`` host syncs
        # avoids a device<->host copy inside the graph capture region.
        if not self.states:
            return
        layer_idx = layer.layer_id - self.start_layer
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens_src = forward_batch.seq_lens_cpu
        if seq_lens_src is None:
            seq_lens_src = forward_batch.seq_lens
        seq_lens = seq_lens_src.tolist()

        for i, req_pool_idx in enumerate(req_indices):
            req_pool_idx = int(req_pool_idx)
            state = self.states.get(req_pool_idx)
            if state is None:
                continue

            # Write this layer's query into the window (no-op off window steps).
            q_i = q[i]
            state.write_window(layer_idx, q_i.reshape(q_i.shape[-2], q_i.shape[-1]))

            if req_pool_idx in self._armed:
                layer_score = self._layer_score(state, layer_idx, int(seq_lens[i]))
                if state.score_accum is None:
                    state.score_accum = layer_score
                else:
                    state.score_accum += layer_score

    def _layer_score(
        self, state: RKVRequestState, layer_idx: int, seq_len: int
    ) -> torch.Tensor:
        """Per-token joint R-KV score for one layer, averaged across KV heads.

        Returns shape ``(seq_len - window_size,)``.
        """
        layer_id = self.start_layer + layer_idx
        r2t = self.req_to_token_pool.req_to_token
        slots = r2t[state.req_pool_idx, :seq_len].long()

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
        # keys: (1, kv_heads, seq_len, head_dim) -- NHD buffer is
        # (num_slots, kv_heads, head_dim); gather this request's slots.
        keys = k_buffer[slots].unsqueeze(0).transpose(1, 2).contiguous()

        # queries: (1, q_heads, window_size, head_dim). window_q is filled in
        # temporal order (slot 0 oldest .. window_size-1 newest) over the eager
        # window steps that begin_decode_step forced before this compaction.
        window_q = state.window_q[layer_idx]  # (window, q_heads, head_dim)
        queries = window_q.unsqueeze(0).transpose(1, 2).contiguous()

        # (1, kv_heads, seq_len - window) -> mean over heads -> (seq_len - window,)
        final_score = self.algo._scores(keys, queries)
        return final_score.mean(dim=1).squeeze(0)

    # ------------------------------------------------------------------
    # Compaction (called once after the full forward pass)
    # ------------------------------------------------------------------
    def maybe_compact(self, forward_batch: ForwardBatch) -> None:
        """Run physical compaction for any request armed this forward pass."""
        if not self._armed:
            return

        seq_len_by_req = self._seq_len_by_req(forward_batch)
        for req_pool_idx in list(self._armed):
            state = self.states.get(req_pool_idx)
            if state is None or state.score_accum is None:
                continue
            seq_len = seq_len_by_req.get(req_pool_idx)
            if seq_len is None:
                continue
            kept = self._assemble_kept(state.score_accum, seq_len)
            self._compact_request(state, seq_len, kept)

        self._armed.clear()

    def _assemble_kept(self, score_accum: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Global kept-token indices: top past tokens + trailing window.

        ``score_accum`` covers the past tokens ``[0, seq_len - window)``.
        Returns ascending indices of length ``budget``.
        """
        num_past = self.config.budget - self.config.window_size
        past_idx = score_accum.topk(num_past).indices
        window_idx = torch.arange(
            seq_len - self.config.window_size,
            seq_len,
            device=score_accum.device,
        )
        kept = torch.cat([past_idx, window_idx])
        return torch.sort(kept).values

    def _compact_request(
        self, state: RKVRequestState, seq_len: int, kept_local: torch.Tensor
    ) -> None:
        """Physically compact one request's KV cache (page_size == 1).

        Relocates surviving slots' K/V to the front ``budget`` slots for every
        layer, frees the freed tail slots, rewrites ``req_to_token``, and clears
        the tail. Kept indices must be ascending.
        """
        idx = state.req_pool_idx
        budget = self.config.budget
        r2t = self.req_to_token_pool.req_to_token

        slots = r2t[idx, :seq_len].long().clone()  # physical slots, temporal order
        src = slots[kept_local]  # surviving physical slots (budget,)
        dst = slots[:budget]  # target front slots (budget,)

        # Relocate K/V for every layer. Clone before write so overlapping
        # src/dst ranges don't corrupt each other.
        for layer_id in range(self.start_layer, self.end_layer):
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buffer = self.token_to_kv_pool.get_value_buffer(layer_id)
            k_keep = k_buffer[src].clone()
            v_keep = v_buffer[src].clone()
            k_buffer[dst] = k_keep
            v_buffer[dst] = v_keep

        # Free the tail slots (page_size == 1 => per-slot free).
        freed = slots[budget:seq_len]
        if freed.numel() > 0:
            self.kv_allocator.free(freed.to(r2t.dtype))

        # req_to_token[:budget] already equals ``dst`` (same physical slots),
        # now holding the relocated kept KV in temporal order. Clear the tail.
        r2t[idx, budget:seq_len] = 0

        state.steps_since_compact = 0

        # Physical-length bookkeeping. The scheduler normally treats seq_lens as
        # BOTH the physical KV length AND the rotary position source; R-KV breaks
        # that identity. We shrink the *physical* length to ``budget`` on the
        # owning request here (same process, shared pools), and expose the new
        # length via ``pending_length_updates`` so the scheduler can update its
        # batch-level seq_lens / seq_lens_cpu tensors. Rotary positions stay
        # *logical* and are supplied separately via ``logical_position`` (see the
        # wiring notes at the bottom of the file). ``next_position`` is
        # intentionally NOT rewound: future tokens keep their absolute positions
        # so their rotary stays consistent with the retained keys.
        if state.req is not None:
            state.req.kv_committed_len = budget
            state.req.kv_allocated_len = budget
        self.pending_length_updates[idx] = budget

        logger.info(
            "R-KV compacted req_pool_idx=%d: phys %d -> %d slots (freed %d)",
            idx,
            seq_len,
            budget,
            freed.numel(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seq_len_by_req(forward_batch: ForwardBatch) -> Dict[int, int]:
        """Map ``req_pool_idx -> physical seq_len`` for the current batch.

        Uses ``seq_lens_cpu`` when available to avoid a device sync, falling
        back to ``seq_lens``. Supports ``batch >= 1``.
        """
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens_src = forward_batch.seq_lens_cpu
        if seq_lens_src is None:
            seq_lens_src = forward_batch.seq_lens
        seq_lens = seq_lens_src.tolist()
        return {int(r): int(s) for r, s in zip(req_indices, seq_lens)}

    @staticmethod
    def logical_position(req: Req) -> int:
        """Rotary position of a request's NEXT token.

        Equals the total number of tokens seen so far
        (``len(origin_input_ids) + len(output_ids)``), which is unaffected by
        KV eviction. Use this to override ``ForwardBatch.positions`` for
        R-KV-managed requests so rotary stays continuous after compaction, even
        though the physical seq_lens has shrunk.
        """
        return len(req.origin_input_ids) + len(req.output_ids)

    def take_pending_length_updates(self) -> Dict[int, int]:
        """Return and clear the pending {req_pool_idx: new_physical_len} map.

        The scheduler calls this right after the forward pass to apply the new
        physical lengths to its batch-level seq_lens / seq_lens_cpu tensors
        (the request-level kv_committed_len / kv_allocated_len are already
        updated in-place during compaction).
        """
        updates = self.pending_length_updates
        self.pending_length_updates = {}
        return updates

    def override_decode_positions(self, forward_batch: ForwardBatch) -> None:
        """Replace decode positions with *logical* positions for R-KV requests.

        ``seq_lens`` now tracks the (possibly shorter) physical KV length, so
        ``clamp_position(seq_lens)`` would rewind rotary after compaction. For
        every R-KV-managed request in the batch we overwrite its position with
        ``logical_position(req)`` so future tokens keep absolute positions
        consistent with the retained keys. No-op for requests we don't manage.
        """
        if forward_batch.positions is None:
            return
        req_indices = forward_batch.req_pool_indices
        for i in range(req_indices.shape[0]):
            st = self.states.get(int(req_indices[i].item()))
            if st is not None and st.req is not None:
                # logical_position() counts all tokens seen so far INCLUDING the
                # token being decoded this step (it was appended to output_ids
                # when it was sampled), so the current token's 0-based rotary
                # position is that count minus one — for an un-compacted request
                # this equals the baseline clamp_position(seq_lens) = seq_lens-1.
                forward_batch.positions[i] = self.logical_position(st.req) - 1


# ---------------------------------------------------------------------------
# Remaining wiring (NOT done here -- needs on-GPU / running-server validation;
# see R-KV/doc/DESIGN.md section 9 roadmap). Design = "scheme A": seq_lens tracks the
# PHYSICAL KV length, rotary positions stay LOGICAL.
#
#   1. FlashInferAttnBackend.forward_decode: after set_kv_buffer, call
#      compressor.observe_decode_layer(q, k, v, layer, forward_batch).
#   2. model_runner (or the backend's end-of-forward hook): after the full
#      decode forward pass, call compressor.maybe_compact(forward_batch), then
#      have the scheduler apply take_pending_length_updates() to batch.seq_lens
#      / seq_lens_cpu (kv_committed_len / kv_allocated_len are already updated).
#   3. scheduler / schedule_batch: call on_request_begin / on_request_end around
#      a request's life; disable overlap scheduling for phase 1 (simpler timing).
#   4. ForwardBatch construction (forward_batch_info): for R-KV-managed requests
#      override positions with RKVCompressor.logical_position(req) - 1 instead of
#      clamp_position(seq_lens), so rotary stays continuous after eviction. There
#      is already an override slot on the decode path (~L802).
# ---------------------------------------------------------------------------
