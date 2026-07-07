"""SnapKV integration layer for SGLang v0.5.14.

Bridges the pure, device-agnostic SnapKV algorithm (:mod:`.algo`) to SGLang's
paged KV cache and the FlashInfer **prefill** path. This mirrors the R-KV
integration (:mod:`sglang.srt.mem_cache.rkv.integration`) but compresses at a
different point in a request's life:

* **R-KV** is a *decoding-time* compressor: it fires repeatedly, every
  ``buffer_size`` generated tokens, evicting redundant / unimportant tokens as
  the chain-of-thought grows.
* **SnapKV** is a *prompt-phase* compressor: it fires **once**, right after the
  prompt has been prefilled, keeping only the ``max_capacity_prompt`` prompt
  tokens the observation window attends to most. Decode then runs against the
  shrunken prompt.

Scope of this module (phase 1, correctness first):

* FlashInfer backend, ``batch >= 1`` with **per-request** prefill compression.
* ``page_size == 1`` token pool, so evicted slots free cleanly one-by-one.
* Chunked prefill **disabled** and radix/prefix cache **disabled** (enforced in
  ``server_args._handle_snapkv_validation``): SnapKV needs every prompt query in
  a single forward pass so the observation window is complete, and it frees KV
  slots the radix tree would otherwise reference.
* Reduction of the algorithm's per-head / per-layer scores into a single global
  per-token eviction decision (``req_to_token`` maps one slot per token, shared
  across all layers and heads):

  - cross-head:  **mean** of the per-head observation ``attn_cache``.
  - cross-layer: **sum** over every layer's ``attn_cache`` (aggregate all layers).

* Physical compaction runs **after the full prefill forward pass** (all layers
  have written their prompt KV), not inside a single layer's ``forward_extend``.

The module is deliberately standalone: it imports only :mod:`.algo` and
``torch`` at module load time, so it can be unit-tested against mock pools
without pulling in the heavy serving stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.mem_cache.snapkv.algo import observation_attn_cache

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy imports
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class SnapKVConfig:
    """Hyper-parameters for SnapKV compression.

    Defaults mirror the reference ``SnapKVCluster`` (``window_size=32``,
    ``kernel_size=5``, ``pooling='avgpool'``) with a serving-oriented
    ``max_capacity_prompt=1024`` budget. A prompt shorter than or equal to
    ``max_capacity_prompt`` is left untouched.
    """

    # KV entries kept per request after prompt compression (the SnapKV budget).
    max_capacity_prompt: int = 1024
    # Trailing observation window, always retained, whose queries score the past.
    window_size: int = 32
    # 1-D pooling kernel used to cluster attention mass over neighbouring tokens.
    kernel_size: int = 5
    # 'avgpool' or 'maxpool'.
    pooling: str = "avgpool"

    def __post_init__(self) -> None:
        assert (
            self.max_capacity_prompt > self.window_size
        ), "max_capacity_prompt must exceed window_size"
        assert self.pooling in (
            "avgpool",
            "maxpool",
        ), "pooling must be 'avgpool' or 'maxpool'"


class SnapKVRequestState:
    """Per-request bookkeeping the integration layer maintains.

    One instance per active request, keyed by ``req_pool_idx`` in
    :class:`SnapKVCompressor`. SnapKV compresses at most once per request, so the
    state is lightweight: a running cross-layer score accumulator (filled during
    the prefill forward) and a flag recording whether compaction already fired.
    """

    def __init__(self, req_pool_idx: int) -> None:
        self.req_pool_idx = req_pool_idx

        # Set True once this request's prompt has been physically compacted, so
        # later prefill chunks / decode steps never re-trigger.
        self.compressed = False

        # Transient cross-layer per-past-token score accumulator, allocated
        # lazily by the first observed layer and consumed by compaction.
        self.score_accum: Optional[torch.Tensor] = None
        # Physical KV length observed when scores were accumulated.
        self.observed_seq_len: int = 0

        # Back-reference to the owning request, so compaction can update its
        # physical-length bookkeeping (kv_committed_len / kv_allocated_len) and
        # so decode positions can be kept logical. Duck-typed: only needs
        # kv_committed_len / kv_allocated_len / origin_input_ids / output_ids.
        self.req: Optional[Req] = None


class SnapKVCompressor:
    """Coordinates SnapKV prompt compression across a request's lifetime.

    Borrows the lifecycle-hook naming and the paged-pool physical-eviction
    machinery from the R-KV integration, but triggers once at the end of prefill
    rather than periodically during decode.
    """

    def __init__(
        self,
        config: SnapKVConfig,
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

        # Active per-request state, keyed by req_pool_idx.
        self.states: Dict[int, SnapKVRequestState] = {}
        # req_pool_idx values that accumulated scores this prefill forward pass.
        self._armed: set[int] = set()
        # req_pool_idx -> new physical KV length after compaction. The scheduler
        # drains this (``take_pending_length_updates``) to update its batch-level
        # seq_lens tensors, which it owns.
        self.pending_length_updates: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def request_wants_compression(req: Req) -> bool:
        """Whether a request opts into SnapKV prompt compression.

        Gated on the request's ``task_type`` (populated from the ``task_type``
        HTTP header): only ``"summarization"`` requests are compressed; any
        other value, or a missing / empty hint, leaves the request on full KV.
        """
        task_type = getattr(req, "task_type", None)
        return (task_type or "").strip().lower() == "summarization"

    def on_request_begin(self, req: Req) -> None:
        """Register a request and initialise its SnapKV state."""
        if req.req_pool_idx is None:
            return
        state = SnapKVRequestState(req_pool_idx=req.req_pool_idx)
        state.req = req
        self.states[req.req_pool_idx] = state

    def on_request_end(self, req: Req) -> None:
        """Drop a request's SnapKV state when it finishes or aborts."""
        if req.req_pool_idx is not None and self.states.pop(req.req_pool_idx, None):
            logger.debug(
                "SnapKV on_request_end req_pool_idx=%d states_left=%d",
                req.req_pool_idx,
                len(self.states),
            )

    # ------------------------------------------------------------------
    # Prefill-time observation (called per layer from forward_extend)
    # ------------------------------------------------------------------
    def observe_prefill_layer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> None:
        """Observe one layer's prefill for every request in the extend batch.

        Called from ``FlashInferAttnBackend.forward_extend`` *after*
        ``set_kv_buffer`` (the prompt K/V is in the pool) and *before* the
        attention wrapper runs. For each request whose prompt exceeds the budget
        and has not yet been compacted, computes this layer's per-past-token
        SnapKV importance and accumulates it across layers.

        ``q`` is the extend query block ``(num_extend_tokens, q_heads, head_dim)``
        with requests concatenated along dim 0; per-request slices are recovered
        from ``extend_seq_lens`` / ``extend_start_loc``.
        """
        layer_idx = layer.layer_id - self.start_layer

        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = self._to_list(forward_batch.seq_lens_cpu, forward_batch.seq_lens)
        extend_lens = self._to_list(
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens
        )
        if extend_lens is None:
            return

        # Per-request start offset into the flattened extend token dimension.
        offset = 0
        for i, req_pool_idx in enumerate(req_indices):
            req_pool_idx = int(req_pool_idx)
            extend_len = int(extend_lens[i])
            start = offset
            offset += extend_len

            state = self.states.get(req_pool_idx)
            if state is None or state.compressed:
                continue

            seq_len = int(seq_lens[i])
            # Only compress once the (full) prompt exceeds the budget. Requires
            # the whole prompt in this single forward (chunked prefill disabled),
            # i.e. extend_len == seq_len.
            if seq_len <= self.config.max_capacity_prompt:
                continue
            if extend_len < seq_len or extend_len <= self.config.window_size:
                continue

            if layer_idx == 0:
                self._armed.add(req_pool_idx)
                state.score_accum = None
                state.observed_seq_len = seq_len

            if req_pool_idx in self._armed:
                # Window queries for this request: the last window_size rows of
                # its extend block, shape (window, q_heads, head_dim).
                window_q = q[
                    start + extend_len - self.config.window_size : start + extend_len
                ]
                layer_score = self._layer_score(
                    req_pool_idx, seq_len, window_q, layer_idx
                )
                if state.score_accum is None:
                    state.score_accum = layer_score
                else:
                    state.score_accum += layer_score

    def _layer_score(
        self,
        req_pool_idx: int,
        seq_len: int,
        window_q: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Per-past-token SnapKV importance for one layer, averaged over KV heads.

        Returns shape ``(seq_len - window_size,)``.
        """
        layer_id = self.start_layer + layer_idx
        r2t = self.req_to_token_pool.req_to_token
        slots = r2t[req_pool_idx, :seq_len].long()

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
        # keys: (1, kv_heads, seq_len, head_dim) -- NHD buffer is
        # (num_slots, kv_heads, head_dim); gather this request's slots.
        keys = k_buffer[slots].unsqueeze(0).transpose(1, 2).contiguous()

        # queries: (1, q_heads, window_size, head_dim)
        queries = window_q.unsqueeze(0).transpose(1, 2).contiguous()

        # (1, kv_heads, seq_len - window) -> mean over heads -> (seq_len - window,)
        attn_cache = observation_attn_cache(
            queries,
            keys,
            self.config.window_size,
            self.config.kernel_size,
            self.config.pooling,
        )
        return attn_cache.mean(dim=1).squeeze(0)

    # ------------------------------------------------------------------
    # Compaction (called once after the full prefill forward pass)
    # ------------------------------------------------------------------
    def maybe_compact(self, forward_batch: ForwardBatch) -> None:
        """Run physical prompt compaction for any request armed this forward."""
        if not self._armed:
            return

        for req_pool_idx in list(self._armed):
            state = self.states.get(req_pool_idx)
            if state is None or state.compressed or state.score_accum is None:
                continue
            seq_len = state.observed_seq_len
            kept = self._assemble_kept(state.score_accum, seq_len)
            self._compact_request(state, seq_len, kept)

        self._armed.clear()

    def _assemble_kept(self, score_accum: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Global kept-token indices: top past tokens + trailing window.

        ``score_accum`` covers the past tokens ``[0, seq_len - window)``.
        Returns ascending indices of length ``max_capacity_prompt``.
        """
        num_past = self.config.max_capacity_prompt - self.config.window_size
        past_idx = score_accum.topk(num_past).indices
        window_idx = torch.arange(
            seq_len - self.config.window_size,
            seq_len,
            device=score_accum.device,
        )
        kept = torch.cat([past_idx, window_idx])
        return torch.sort(kept).values

    def _compact_request(
        self, state: SnapKVRequestState, seq_len: int, kept_local: torch.Tensor
    ) -> None:
        """Physically compact one request's prompt KV cache (page_size == 1).

        Relocates surviving slots' K/V to the front ``max_capacity_prompt`` slots
        for every layer, frees the freed tail slots, rewrites ``req_to_token``,
        and clears the tail. Kept indices must be ascending.
        """
        idx = state.req_pool_idx
        budget = self.config.max_capacity_prompt
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

        state.compressed = True

        # Physical-length bookkeeping. The scheduler normally treats seq_lens as
        # BOTH the physical KV length AND the rotary position source; SnapKV
        # breaks that identity. We shrink the *physical* length to ``budget`` on
        # the owning request here (same process, shared pools), and expose the
        # new length via ``pending_length_updates`` so the scheduler can update
        # its batch-level seq_lens / seq_lens_cpu tensors. Rotary positions stay
        # *logical* and are supplied separately via ``override_decode_positions``,
        # so decode tokens keep absolute positions consistent with the retained
        # prompt keys (whose rotary was baked in at their original positions).
        if state.req is not None:
            state.req.kv_committed_len = budget
            state.req.kv_allocated_len = budget
        self.pending_length_updates[idx] = budget

        logger.info(
            "SnapKV compacted req_pool_idx=%d: prompt %d -> %d slots (freed %d)",
            idx,
            seq_len,
            budget,
            freed.numel(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_list(cpu_src, dev_src) -> Optional[List[int]]:
        """Return a python list from the CPU mirror when available, else device."""
        src = cpu_src if cpu_src is not None else dev_src
        if src is None:
            return None
        if isinstance(src, list):
            return src
        return src.tolist()

    @staticmethod
    def logical_position(req: Req) -> int:
        """Rotary position count of a request's tokens seen so far.

        Equals ``len(origin_input_ids) + len(output_ids)``, unaffected by prompt
        eviction. Used to override ``ForwardBatch.positions`` for SnapKV-managed
        requests so rotary stays continuous after the prompt physically shrank.
        """
        return len(req.origin_input_ids) + len(req.output_ids)

    def take_pending_length_updates(self) -> Dict[int, int]:
        """Return and clear the pending {req_pool_idx: new_physical_len} map.

        The scheduler calls this before advancing decode to apply the new
        physical lengths to its batch-level seq_lens / seq_lens_cpu tensors (the
        request-level kv_committed_len / kv_allocated_len are already updated
        in-place during compaction).
        """
        updates = self.pending_length_updates
        self.pending_length_updates = {}
        return updates

    def override_decode_positions(self, forward_batch: ForwardBatch) -> None:
        """Replace decode positions with *logical* positions for SnapKV requests.

        ``seq_lens`` now tracks the (shorter) physical KV length after prompt
        compaction, so ``clamp_position(seq_lens)`` would rewind rotary. For
        every SnapKV-managed request we overwrite its position with
        ``logical_position(req) - 1`` (the 0-based position of the token being
        decoded this step). For an un-compacted request this equals the baseline
        ``clamp_position(seq_lens) = seq_lens - 1``, so it is a safe no-op there.
        """
        if forward_batch.positions is None:
            return
        req_indices = forward_batch.req_pool_indices
        for i in range(req_indices.shape[0]):
            st = self.states.get(int(req_indices[i].item()))
            if st is not None and st.req is not None:
                forward_batch.positions[i] = self.logical_position(st.req) - 1
