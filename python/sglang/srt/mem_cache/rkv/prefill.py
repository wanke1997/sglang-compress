"""R-KV as a **prefill-phase** KV-cache compressor (pure, device-agnostic).

This module uses R-KV (attention *importance* combined with key *redundancy*) to
compress a long prompt right after prefill: keep the ``budget`` most
important / least-redundant tokens, which adds an ``O(n^2)`` pairwise-key-similarity
term. It implements two strategies and the primitives to diff-test one against
the other:

* **Route A — one-shot** (:meth:`RKVPrefill.oneshot_keep`): score the *whole*
  prompt once, against the true final observation window (the last
  ``window_size`` prompt tokens), then keep ``budget`` tokens. This is the
  faithful R-KV semantics — no token is evicted before the entire prompt has
  been seen — so it is the **accuracy oracle**. The only cost is the ``O(n^2)``
  similarity; :func:`cal_similarity_tiled` computes it in row blocks so peak
  memory is ``O(n)`` instead of ``O(n^2)``.

* **Route B — buffered / chunked** (:meth:`RKVPrefill.buffered_keep`): emulate
  chunked prefill. Feed the prompt in chunks; whenever the physical KV length
  exceeds ``budget + buffer``, compress back to ``budget``. The similarity
  matrix then never exceeds ``(budget + buffer + chunk_size)`` on a side, so
  memory and per-compaction compute are bounded regardless of prompt length —
  this is what the R-KV serving ports mean by "the buffer suppresses the
  quadratic growth". The price is *premature eviction*: a token scored low
  against an early, mid-prefill window is dropped irreversibly, even if a later
  chunk would have attended to it.

Both routes reduce the per-head / per-layer scores into a single global
per-token decision exactly the way the serving integration must (one
``req_to_token`` slot per token, shared across heads and layers):

* cross-head:  **mean** of the per-head joint score.
* cross-layer: **sum** over every layer's score.

so their kept-index sets are directly comparable. See
``test/srt/mem_cache/test_rkv_prefill.py`` for the parity + diff-test harness.

Tensor conventions (per layer, batch size 1 is implied — one request at a time):

* ``keys[l]``:   ``(kv_heads, n, head_dim)``
* ``queries[l]`` / ``window_q[l]``: ``(q_heads, m, head_dim)``

Grouped-query attention (``q_heads`` a multiple of ``kv_heads``) is handled the
same way as the reference: per-group logits are pooled down to ``kv_heads``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

__all__ = ["cal_similarity_tiled", "RKVPrefill"]


def _attention_logits(query_states, key_states, pooling="max"):
    """Scaled dot-product logits ``q @ k^T / sqrt(d)``, GQA-pooled to kv heads.

    ``query_states``: ``(1, q_heads, q_len, head_dim)``.
    ``key_states``:   ``(1, kv_heads, kv_len, head_dim)``.
    Returns ``(1, kv_heads, q_len, kv_len)``.

    Identical math to ``rkv.algo.compute_attention_scores`` — duplicated here so
    this module stays importable by file path (GPU-free unit tests), matching
    the standalone style of ``rkv/algo.py``.
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    group = q_heads // kv_heads

    if group == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            head_dim
        )

    query_states = query_states.view(batch_size, kv_heads, group, q_len, head_dim)
    key_states = key_states.unsqueeze(2)
    attn = torch.matmul(query_states, key_states.transpose(3, 4)) / math.sqrt(head_dim)
    if pooling == "mean":
        return attn.mean(dim=2)
    if pooling == "max":
        return attn.max(dim=2).values
    raise ValueError("Pooling method not supported")


def cal_similarity_tiled(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_direction: str = "last",
    row_block: int = 2048,
) -> torch.Tensor:
    """Memory-bounded redundancy score, parity-equal to ``algo.cal_similarity``.

    The reference materialises the full ``(1, kv_heads, n, n)`` cosine-similarity
    matrix, zeroes the diagonal and (per row) the most-recent above-threshold
    neighbour, then reduces ``mean`` over the row axis and ``softmax`` over the
    column axis. That is ``O(n^2)`` memory. Here we accumulate the per-column
    sum in blocks of at most ``row_block`` rows, so peak memory is
    ``O(kv_heads * row_block * n)``. The result is bit-parity (up to float
    reduction order) with the reference for ``retain_direction='last'`` — the
    only direction the R-KV serving path uses.

    ``key_states``: ``(1, kv_heads, n, head_dim)``. Returns ``(1, kv_heads, n)``.
    """
    if retain_direction != "last":
        # The reference supports other directions but the serving default (and
        # everything we diff-test) is "last"; fall back to the exact reference
        # for the rare other cases to avoid silently diverging.
        from sglang.srt.mem_cache.rkv.algo import cal_similarity

        return cal_similarity(
            key_states, threshold=threshold, retain_direction=retain_direction
        )

    _, kv_heads, seq_len, _ = key_states.shape
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    col_idx = torch.arange(seq_len, device=key_states.device)

    col_sum = torch.zeros(
        (1, kv_heads, seq_len), dtype=torch.float32, device=key_states.device
    )
    for r0 in range(0, seq_len, row_block):
        r1 = min(r0 + row_block, seq_len)
        block = k_norm[:, :, r0:r1, :]  # (1, kv_heads, B, d)
        sim = torch.matmul(block, k_norm.transpose(-1, -2))  # (1, kv_heads, B, n)

        # Zero the diagonal entries sim[:, :, li, r0 + li].
        local = torch.arange(r1 - r0, device=key_states.device)
        sim[:, :, local, r0 + local] = 0.0

        # Per row, find the most-recent (largest-index) neighbour above the
        # threshold and exempt it (scatter it back to zero), matching the
        # reference's ``similarity_retain`` for retain_direction="last".
        mask = sim > threshold
        idx = torch.where(mask, col_idx.view(1, 1, 1, seq_len), 0)
        retain = idx.max(dim=-1).values  # (1, kv_heads, B)
        sim.scatter_(-1, retain.unsqueeze(-1), 0.0)

        col_sum += sim.to(torch.float32).sum(dim=-2)  # (1, kv_heads, n)

    redundancy = (col_sum / seq_len).to(key_states.dtype)
    return redundancy.softmax(dim=-1)


class RKVPrefill:
    """Prefill-phase R-KV compressor with a one-shot and a buffered strategy.

    Scores are reduced to one global per-token decision (head-mean, layer-sum)
    so route A and route B produce directly comparable kept-index sets.
    """

    def __init__(
        self,
        budget: int = 1024,
        window_size: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.1,
        retain_ratio: float = 0.1,
        retain_direction: str = "last",
        sim_threshold: float = 0.5,
        row_block: int = 2048,
    ) -> None:
        assert budget > window_size, "budget must exceed window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.sim_threshold = sim_threshold
        self.row_block = row_block

    # ------------------------------------------------------------------
    # Per-layer scoring (heads reduced by mean -> one score per past token)
    # ------------------------------------------------------------------
    def _importance_past(
        self, keys: torch.Tensor, window_q: torch.Tensor
    ) -> torch.Tensor:
        """Attention importance of each past token, averaged over kv heads.

        ``keys``: ``(1, kv_heads, n, d)``; ``window_q``: ``(1, q_heads, w, d)``
        where ``w == window_size`` and the last ``window_size`` KV entries are
        the observation window. Returns ``(n - window_size,)``.
        """
        attn = _attention_logits(window_q, keys)  # (1, kv_heads, w, n)
        attn_sum = (
            F.softmax(
                attn[:, :, -self.window_size :, : -self.window_size],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(keys.dtype)
        )  # (1, kv_heads, n - window)
        attn_cache = F.max_pool1d(
            attn_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        return attn_cache

    def _redundancy_past(self, keys: torch.Tensor) -> torch.Tensor:
        """Tiled redundancy of each past token, averaged over kv heads.

        ``keys``: ``(1, kv_heads, n, d)``. Returns ``(1, kv_heads, n - window)``.
        """
        redundancy = cal_similarity_tiled(
            keys,
            threshold=self.sim_threshold,
            retain_direction=self.retain_direction,
            row_block=self.row_block,
        )
        return redundancy[:, :, : -self.window_size]

    def layer_past_score(
        self, keys: torch.Tensor, window_q: torch.Tensor
    ) -> torch.Tensor:
        """Joint R-KV score per past token for one layer, averaged over kv heads.

        Returns ``(n - window_size,)``. This mirrors
        ``rkv.integration.RKVCompressor._layer_score`` (which does
        ``final_score.mean(dim=1).squeeze(0)``) so serving and this diff-test
        harness agree token-for-token.
        """
        importance = self._importance_past(keys, window_q)  # (1, kv_heads, n-w)
        redundancy = self._redundancy_past(keys)  # (1, kv_heads, n-w)
        final = importance * self.mix_lambda - redundancy * (1 - self.mix_lambda)
        return final.mean(dim=1).squeeze(0)  # (n - window,)

    def _select_from_score(
        self, score_past: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Top past tokens + trailing window, as ascending indices (``budget``).

        ``score_past`` covers past tokens ``[0, seq_len - window)``.
        """
        num_past = self.budget - self.window_size
        past_idx = score_past.topk(num_past).indices
        window_idx = torch.arange(
            seq_len - self.window_size, seq_len, device=score_past.device
        )
        kept = torch.cat([past_idx, window_idx])
        return torch.sort(kept).values

    # ------------------------------------------------------------------
    # Route A — one-shot
    # ------------------------------------------------------------------
    def oneshot_keep(
        self,
        keys: Sequence[torch.Tensor],
        window_q: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Score the whole prompt once vs the true final window (accuracy oracle).

        ``keys[l]``: ``(kv_heads, n, d)`` for every layer ``l``.
        ``window_q[l]``: ``(q_heads, window_size, d)`` — the last ``window_size``
        prompt queries of layer ``l``.

        Returns ascending original-token indices to keep, length ``budget`` (or
        all ``n`` indices when ``n <= budget``, i.e. nothing to drop).
        """
        n = keys[0].shape[-2]
        device = keys[0].device
        if n <= self.budget:
            return torch.arange(n, device=device)

        score = None
        for k_l, wq_l in zip(keys, window_q):
            k = k_l.unsqueeze(0)  # (1, kv_heads, n, d)
            wq = wq_l.unsqueeze(0)  # (1, q_heads, window, d)
            s = self.layer_past_score(k, wq)  # (n - window,)
            score = s if score is None else score + s
        return self._select_from_score(score, n)

    # ------------------------------------------------------------------
    # Route B — buffered / chunked
    # ------------------------------------------------------------------
    def buffered_keep(
        self,
        keys: Sequence[torch.Tensor],
        queries: Sequence[torch.Tensor],
        chunk_size: int,
        buffer: int,
    ) -> torch.Tensor:
        """Emulate chunked prefill with buffer-bounded compaction (route B).

        Feeds the prompt in ``chunk_size``-token chunks. After a chunk lands, if
        the physical KV length exceeds ``budget + buffer`` it compresses back to
        ``budget`` — scoring against the window of the *current* physical KV (its
        last ``window_size`` tokens), not the final prompt window. Peak physical
        length (and thus the similarity side length) is bounded by
        ``budget + buffer + chunk_size``.

        ``keys[l]``:    ``(kv_heads, n, d)``; ``queries[l]``: ``(q_heads, n, d)``
        — the full per-layer prompt keys/queries (post-rotary), indexed by
        original token position.

        Returns ascending original-token indices to keep, length ``budget`` (or
        all ``n`` when ``n <= budget``).
        """
        n = keys[0].shape[-2]
        device = keys[0].device
        if n <= self.budget:
            return torch.arange(n, device=device)

        # Original indices currently resident in the (physical) KV, temporal.
        kept = torch.arange(0, min(chunk_size, n), device=device)
        pos = int(kept.numel())
        while True:
            if kept.numel() > self.budget + buffer:
                kept = self._compress_buffered(keys, queries, kept)
            if pos >= n:
                break
            nxt = min(pos + chunk_size, n)
            kept = torch.cat([kept, torch.arange(pos, nxt, device=device)])
            pos = nxt
        # Final compaction to exactly ``budget`` before decode (real prefill
        # always ends at budget). This last score is against the *true* final
        # window, so ``buffer >= n`` — no mid-prefill compaction — recovers
        # route A exactly.
        if kept.numel() > self.budget:
            kept = self._compress_buffered(keys, queries, kept)
        return torch.sort(kept).values

    def _compress_buffered(
        self,
        keys: Sequence[torch.Tensor],
        queries: Sequence[torch.Tensor],
        kept: torch.Tensor,
    ) -> torch.Tensor:
        """One buffered compaction: score current physical KV, keep ``budget``.

        ``kept`` are the ascending original indices currently resident. The
        observation window is the last ``window_size`` of them. Returns the new
        ascending original indices (length ``budget``).
        """
        phys = int(kept.numel())
        window_orig = kept[-self.window_size :]  # (window,)

        score = None
        for k_l, q_l in zip(keys, queries):
            k = k_l.index_select(-2, kept).unsqueeze(0)  # (1, kv_heads, phys, d)
            wq = q_l.index_select(-2, window_orig).unsqueeze(0)  # (1, q_heads, w, d)
            s = self.layer_past_score(k, wq)  # (phys - window,)
            score = s if score is None else score + s

        local_keep = self._select_from_score(score, phys)  # local idx into kept
        return kept.index_select(0, local_keep)
