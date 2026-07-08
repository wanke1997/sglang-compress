"""R-KV as a prefill-phase compressor — SGLang serving integration.

Bridges the pure prefill algorithm (:mod:`sglang.srt.mem_cache.rkv.prefill`,
:class:`RKVPrefill`) to SGLang's paged KV cache, mirroring
:class:`sglang.srt.mem_cache.snapkv.integration.SnapKVCompressor` method-for-method
so it drops into the exact same wiring points (``observe_prefill_layer`` from
``forward_extend``, ``maybe_compact`` after an extend forward,
``override_decode_positions`` at decode, ``on_request_begin/end`` from the
scheduler). The only difference from SnapKV is the *score*: R-KV combines
attention importance with a key-redundancy term (``O(n^2)`` similarity computed
in row blocks so peak memory stays ``O(n)``).

Two modes, selected by :attr:`RKVPrefillConfig.mode`:

* ``"oneshot"`` (route A) — score the whole prompt once, at the end of prefill,
  against the true final observation window. Structurally identical to SnapKV
  (compresses once, before decode); this is the accuracy oracle.
* ``"buffered"`` (route B) — compress mid-prefill whenever the physical KV length
  exceeds ``budget + buffer``, bounding the similarity matrix. Trades a little
  fidelity (premature eviction) for prompt-length-independent memory. The
  scheduler-side mid-prefill length propagation is wired in phase 2.

See ``R-KV/doc/DESIGN.md`` and the A/B diff-test in
``R-KV/benchmark/rkv_prefill_ab.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import msgspec
import torch

from sglang.srt.mem_cache.rkv.prefill import RKVPrefill

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy imports
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RKVPrefillConfig(msgspec.Struct):
    """Hyper-parameters for R-KV prefill compression.

    Defaults mirror the R-KV reference (``window_size`` bumped to a SnapKV-like
    32 for prompt-phase scoring). ``mode`` picks the one-shot oracle or the
    buffered strategy; ``buffer`` only matters for ``"buffered"``.
    """

    mode: str = "oneshot"  # "oneshot" (route A) or "buffered" (route B)
    budget: int = 1024
    window_size: int = 32
    kernel_size: int = 7
    mix_lambda: float = 0.1
    retain_ratio: float = 0.1
    sim_threshold: float = 0.5
    buffer: int = 512
    row_block: int = 2048

    def __post_init__(self) -> None:
        assert self.mode in (
            "oneshot",
            "buffered",
        ), "mode must be 'oneshot' or 'buffered'"
        assert self.budget > self.window_size, "budget must exceed window_size"
        if self.mode == "buffered":
            assert self.buffer >= 0, "buffer must be non-negative"


class RKVPrefillRequestState:
    """Per-request bookkeeping, keyed by ``req_pool_idx``.

    For ``oneshot`` this is SnapKV-shaped: a cross-layer score accumulator plus a
    per-layer buffer of the true final observation-window queries (filled across
    chunks). For ``buffered`` the per-layer buffer instead holds the *sliding*
    last-``window_size`` queries and ``compressed`` is never latched (a request
    may be compacted several times as prefill progresses).
    """

    def __init__(self, req_pool_idx: int) -> None:
        self.req_pool_idx = req_pool_idx
        # oneshot: latched True after the single prompt compaction.
        self.compressed = False
        # Cross-layer per-past-token score accumulator (oneshot).
        self.score_accum: Optional[torch.Tensor] = None
        self.observed_seq_len: int = 0
        # Full prompt length in KV tokens (set in on_request_begin).
        self.prompt_len: int = 0
        # Per-layer observation-window queries:
        #   oneshot  -> the true final window (positions prompt_len-w .. prompt_len)
        #   buffered -> the sliding last-w queries currently resident
        # Shape (num_layers, window_size, q_heads, head_dim); lazy-allocated.
        self.window_q: Optional[torch.Tensor] = None
        # buffered only: number of KV tokens dropped so far by earlier
        # compactions of THIS request (so logical positions stay correct).
        self.dropped: int = 0
        # buffered only: ascending ORIGINAL token indices currently kept by the
        # logical (index-only) segmented compaction. Physical compaction at the
        # end of prefill relocates exactly these slots. None until first chunk.
        self.kept_orig: Optional[torch.Tensor] = None
        self.req: Optional[Req] = None


class RKVPrefillCompressor:
    """R-KV prefill compressor with SnapKV-compatible wiring hooks."""

    def __init__(
        self,
        config: RKVPrefillConfig,
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

        self.algo = RKVPrefill(
            budget=config.budget,
            window_size=config.window_size,
            kernel_size=config.kernel_size,
            mix_lambda=config.mix_lambda,
            retain_ratio=config.retain_ratio,
            sim_threshold=config.sim_threshold,
            row_block=config.row_block,
        )

        self.states: Dict[int, RKVPrefillRequestState] = {}
        self._armed: set[int] = set()
        self.pending_length_updates: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def request_wants_compression(req: Req) -> bool:
        """Gate on the ``task_type`` header, same policy as SnapKV."""
        task_type = getattr(req, "task_type", None)
        return (task_type or "").strip().lower() == "summarization"

    def on_request_begin(self, req: Req) -> None:
        if req.req_pool_idx is None:
            return
        state = RKVPrefillRequestState(req_pool_idx=req.req_pool_idx)
        state.req = req
        state.prompt_len = len(req.origin_input_ids)
        self.states[req.req_pool_idx] = state

    def on_request_end(self, req: Req) -> None:
        if req.req_pool_idx is not None and self.states.pop(req.req_pool_idx, None):
            logger.debug(
                "R-KV-prefill on_request_end req_pool_idx=%d states_left=%d",
                req.req_pool_idx,
                len(self.states),
            )

    # ------------------------------------------------------------------
    # Prefill-time observation (per layer, from forward_extend)
    # ------------------------------------------------------------------
    def observe_prefill_layer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> None:
        """Buffer this layer's observation-window queries (and, for oneshot, its
        per-past-token score) for every request in the extend batch.

        ``q`` is ``(num_extend_tokens, q_heads, head_dim)`` with requests
        concatenated along dim 0; per-request slices come from
        ``extend_seq_lens`` / ``seq_lens``.
        """
        layer_idx = layer.layer_id - self.start_layer
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = self._to_list(forward_batch.seq_lens_cpu, forward_batch.seq_lens)
        extend_lens = self._to_list(
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens
        )
        if extend_lens is None:
            return

        if self.config.mode == "buffered":
            self._observe_buffered(q, req_indices, seq_lens, extend_lens, layer_idx)
        else:
            self._observe_oneshot(q, req_indices, seq_lens, extend_lens, layer_idx)

    def _observe_oneshot(self, q, req_indices, seq_lens, extend_lens, layer_idx):
        window = self.config.window_size
        offset = 0
        for i, req_pool_idx in enumerate(req_indices):
            req_pool_idx = int(req_pool_idx)
            extend_len = int(extend_lens[i])
            start = offset
            offset += extend_len

            state = self.states.get(req_pool_idx)
            if state is None or state.compressed:
                continue
            prompt_len = state.prompt_len
            if prompt_len <= self.config.budget:
                continue

            seq_len = int(seq_lens[i])
            prefix = seq_len - extend_len
            w0 = prompt_len - window  # observation-window start (global index)

            # Capture this chunk's slice of the true final window.
            ov_start = max(prefix, w0)
            ov_end = min(prefix + extend_len, prompt_len)
            if ov_end > ov_start:
                if state.window_q is None:
                    q_heads, head_dim = q.shape[-2], q.shape[-1]
                    state.window_q = torch.zeros(
                        (self.num_layers, window, q_heads, head_dim),
                        device=q.device,
                        dtype=q.dtype,
                    )
                state.window_q[layer_idx, ov_start - w0 : ov_end - w0] = q[
                    start + (ov_start - prefix) : start + (ov_end - prefix)
                ]

            # Score + arm only on the FINAL prefill chunk.
            if prefix + extend_len < prompt_len:
                continue
            if layer_idx == 0:
                self._armed.add(req_pool_idx)
                state.score_accum = None
                state.observed_seq_len = prompt_len
            if req_pool_idx in self._armed and state.window_q is not None:
                layer_score = self._layer_score(
                    req_pool_idx, prompt_len, state.window_q[layer_idx], layer_idx
                )
                state.score_accum = (
                    layer_score
                    if state.score_accum is None
                    else state.score_accum + layer_score
                )

    def _observe_buffered(self, q, req_indices, seq_lens, extend_lens, layer_idx):
        """Route B: maintain the sliding window and the ``kept_orig`` set, and
        run a *logical* (index-only) compaction at each chunk boundary once the
        kept set exceeds ``budget + buffer``.

        No physical KV is freed here — the pool keeps every prompt token until
        the single end-of-prefill physical compaction (``_compact_buffered``).
        This keeps the extend path's logical==physical invariant intact (so
        chunked prefill is untouched) while still reproducing Route B's
        premature-eviction token choice and bounding the similarity matrix to
        ``(budget + buffer)`` tokens.
        """
        window = self.config.window_size
        last_layer = layer_idx == self.num_layers - 1
        offset = 0
        for i, req_pool_idx in enumerate(req_indices):
            req_pool_idx = int(req_pool_idx)
            extend_len = int(extend_lens[i])
            start = offset
            offset += extend_len

            state = self.states.get(req_pool_idx)
            if state is None or state.compressed:
                continue
            if state.prompt_len <= self.config.budget:
                continue
            seq_len = int(seq_lens[i])
            prefix = seq_len - extend_len

            if state.window_q is None:
                q_heads, head_dim = q.shape[-2], q.shape[-1]
                state.window_q = torch.zeros(
                    (self.num_layers, window, q_heads, head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
            # Update this layer's sliding window: the last ``window`` queries
            # seen so far. Shift older entries left when the chunk is shorter
            # than the window (rare: chunk_size >> window in practice).
            take = min(window, extend_len)
            if take > 0:
                if take < window:
                    state.window_q[layer_idx, : window - take] = state.window_q[
                        layer_idx, take:
                    ].clone()
                state.window_q[layer_idx, window - take :] = q[
                    start + extend_len - take : start + extend_len
                ]

            # Extend the kept set with this chunk's original indices (once, at
            # layer 0), then logically compact at the last layer (all layers'
            # K for this chunk are now pooled and every window is refreshed).
            if layer_idx == 0:
                new_idx = torch.arange(prefix, prefix + extend_len, device=q.device)
                state.kept_orig = (
                    new_idx
                    if state.kept_orig is None
                    else torch.cat([state.kept_orig, new_idx])
                )
            if (
                last_layer
                and state.kept_orig is not None
                and state.kept_orig.numel() > self.config.budget + self.config.buffer
            ):
                self._logical_compress(req_pool_idx, state)

    def _logical_compress(self, req_pool_idx: int, state) -> None:
        """Score the current ``kept_orig`` set (all layers summed) against the
        sliding window and shrink it to ``budget`` — index-only, no KV moves.
        """
        kept = state.kept_orig
        r2t = self.req_to_token_pool.req_to_token
        phys_slots = r2t[req_pool_idx, kept].long()
        score = None
        for layer_idx in range(self.num_layers):
            layer_id = self.start_layer + layer_idx
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
            keys = k_buffer[phys_slots].unsqueeze(0).transpose(1, 2).contiguous()
            queries = (
                state.window_q[layer_idx].unsqueeze(0).transpose(1, 2).contiguous()
            )
            s = self.algo.layer_past_score(keys, queries)
            score = s if score is None else score + s
        local = self.algo._select_from_score(score, kept.numel())
        state.kept_orig = kept.index_select(0, local)

    def _layer_score(
        self,
        req_pool_idx: int,
        seq_len: int,
        window_q: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Per-past-token joint R-KV score for one layer, averaged over kv heads.

        Reads this request's ``seq_len`` prompt keys from the pool and scores
        them against ``window_q`` (the observation window). Returns
        ``(seq_len - window_size,)``.
        """
        layer_id = self.start_layer + layer_idx
        r2t = self.req_to_token_pool.req_to_token
        slots = r2t[req_pool_idx, :seq_len].long()
        k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
        keys = k_buffer[slots].unsqueeze(0).transpose(1, 2).contiguous()
        queries = window_q.unsqueeze(0).transpose(1, 2).contiguous()
        return self.algo.layer_past_score(keys, queries)

    # ------------------------------------------------------------------
    # Compaction (after an extend forward)
    # ------------------------------------------------------------------
    def maybe_compact(self, forward_batch: ForwardBatch) -> None:
        if self.config.mode == "buffered":
            self._compact_buffered(forward_batch)
        else:
            self._compact_oneshot()

    def _compact_oneshot(self) -> None:
        if not self._armed:
            return
        for req_pool_idx in list(self._armed):
            state = self.states.get(req_pool_idx)
            if state is None or state.compressed or state.score_accum is None:
                continue
            seq_len = state.observed_seq_len
            kept = self._assemble_kept(state.score_accum, seq_len)
            self._compact_request(state, seq_len, kept, latch=True)
        self._armed.clear()

    def _compact_buffered(self, forward_batch: ForwardBatch) -> None:
        """End-of-prefill physical compaction to ``kept_orig`` (route B).

        Fires only on a request's FINAL prefill chunk (``seq_len ==
        prompt_len``). By then ``kept_orig`` holds the buffered logical
        selection; a final forced shrink to ``budget`` (scored against the true
        final window) matches the pure algorithm, then the surviving slots are
        physically relocated to the front and the tail is freed — exactly like
        one-shot, so the decode path is identical.
        """
        seq_len_by_req = self._seq_len_by_req(forward_batch)
        for req_pool_idx, state in list(self.states.items()):
            if state.compressed or state.kept_orig is None:
                continue
            if state.prompt_len <= self.config.budget:
                continue
            seq_len = seq_len_by_req.get(req_pool_idx)
            if seq_len is None or seq_len < state.prompt_len:
                continue  # not the final prefill chunk yet
            # Final forced compaction to budget, against the true final window.
            if state.kept_orig.numel() > self.config.budget:
                self._logical_compress(req_pool_idx, state)
            self._compact_request(state, seq_len, state.kept_orig, latch=True)

    def _assemble_kept(self, score_accum: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Top past tokens + trailing window, ascending, length ``budget``."""
        num_past = self.config.budget - self.config.window_size
        past_idx = score_accum.topk(num_past).indices
        window_idx = torch.arange(
            seq_len - self.config.window_size, seq_len, device=score_accum.device
        )
        return torch.sort(torch.cat([past_idx, window_idx])).values

    def _compact_request(
        self,
        state: RKVPrefillRequestState,
        seq_len: int,
        kept_local: torch.Tensor,
        latch: bool,
    ) -> None:
        """Physically compact one request's KV to ``budget`` slots (page_size 1)."""
        idx = state.req_pool_idx
        budget = self.config.budget
        r2t = self.req_to_token_pool.req_to_token

        slots = r2t[idx, :seq_len].long().clone()
        src = slots[kept_local]
        dst = slots[:budget]

        for layer_id in range(self.start_layer, self.end_layer):
            k_buffer = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buffer = self.token_to_kv_pool.get_value_buffer(layer_id)
            k_keep = k_buffer[src].clone()
            v_keep = v_buffer[src].clone()
            k_buffer[dst] = k_keep
            v_buffer[dst] = v_keep

        freed = slots[budget:seq_len]
        if freed.numel() > 0:
            self.kv_allocator.free(freed.to(r2t.dtype))
        r2t[idx, budget:seq_len] = 0

        if latch:
            state.compressed = True
            state.window_q = None
        else:
            # buffered: track how many tokens were dropped so logical positions
            # of subsequent chunks stay correct.
            state.dropped += seq_len - budget

        if state.req is not None:
            state.req.kv_committed_len = budget
            state.req.kv_allocated_len = budget
        self.pending_length_updates[idx] = budget

        logger.info(
            "R-KV-prefill(%s) compacted req_pool_idx=%d: %d -> %d (freed %d)",
            self.config.mode,
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
        src = cpu_src if cpu_src is not None else dev_src
        if src is None:
            return None
        if isinstance(src, list):
            return src
        return src.tolist()

    @staticmethod
    def _seq_len_by_req(forward_batch: ForwardBatch) -> Dict[int, int]:
        req_indices = forward_batch.req_pool_indices.tolist()
        src = forward_batch.seq_lens_cpu
        if src is None:
            src = forward_batch.seq_lens
        seq_lens = src.tolist()
        return {int(r): int(s) for r, s in zip(req_indices, seq_lens)}

    @staticmethod
    def logical_position(req: Req) -> int:
        return len(req.origin_input_ids) + len(req.output_ids)

    def take_pending_length_updates(self) -> Dict[int, int]:
        updates = self.pending_length_updates
        self.pending_length_updates = {}
        return updates

    def override_decode_positions(self, forward_batch: ForwardBatch) -> None:
        if forward_batch.positions is None:
            return
        req_indices = forward_batch.req_pool_indices
        for i in range(req_indices.shape[0]):
            st = self.states.get(int(req_indices[i].item()))
            if st is not None and st.req is not None:
                forward_batch.positions[i] = self.logical_position(st.req) - 1
