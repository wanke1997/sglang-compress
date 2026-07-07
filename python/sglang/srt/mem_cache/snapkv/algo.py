"""Pure SnapKV compression algorithm (device-agnostic).

This is a faithful port of the reference implementation in the SnapKV
repository (``snapkv/monkeypatch/snapkv_utils.py``, class ``SnapKVCluster``). It
intentionally has **no** SGLang dependencies so it can be unit-tested on CPU and
compared bit-for-bit against the original, and it runs on whatever device the
input tensors live on (GPU in production).

SnapKV is a **prompt-phase** (prefill-time) KV-cache compressor: once, right
after the prompt has been processed, it looks at the attention that the last
``window_size`` "observation" query tokens pay to the earlier prompt tokens,
pools it, and keeps only the ``max_capacity_prompt - window_size`` highest-scoring
past tokens plus the trailing observation window. All later decode steps then run
against this shrunken prompt KV.

Tensor conventions (matching the reference):

* ``query_states``: ``(bsz, q_heads, q_len, head_dim)``
* ``key_states`` / ``value_states``: ``(bsz, kv_heads, kv_len, head_dim)``

SnapKV supports grouped-query attention: ``q_heads`` may be a multiple of
``kv_heads``, in which case the observation-window attention is pooled across
each query group (``compute_snap_attention``). The reference monkeypatch instead
calls :meth:`SnapKVCluster.update_kv` on already-``repeat_kv``-expanded tensors
(``q_heads == kv_heads``); that path is preserved for bit-level parity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SnapKVCluster", "compute_snap_attention", "observation_attn_cache"]


def _causal_window_mask(window_size: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Lower-triangular additive mask for the window's self-attention block.

    Matches the reference: the ``window_size x window_size`` bottom-right block
    of the attention logits is masked so an observation query cannot attend to
    later observation queries (they are "future" tokens for it).
    """
    mask = torch.full(
        (window_size, window_size),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return mask


def compute_snap_attention(query_states, key_states, window_size, pooling="max"):
    """Observation-window importance signal with grouped-query pooling.

    Computes, for the last ``window_size`` queries, the softmax attention over
    all keys, then sums the attention mass each *past* key receives across the
    observation rows. Grouped-query attention is handled by reshaping queries
    into their kv groups and pooling the per-group logits (``max`` or ``mean``)
    so the result is indexed by kv head.

    Returns ``attn_weights_sum`` of shape
    ``(bsz, kv_heads, kv_len - window_size)``.
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    window_q = query_states[:, :, -window_size:, :]

    if query_group_size == 1:
        attn_weights = torch.matmul(window_q, key_states.transpose(2, 3)) / math.sqrt(
            head_dim
        )
    else:
        window_q = window_q.view(
            batch_size, kv_heads, query_group_size, window_size, head_dim
        )
        keys = key_states.unsqueeze(2)
        attn_weights = torch.matmul(window_q, keys.transpose(3, 4)) / math.sqrt(
            head_dim
        )
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")

    # Causal mask on the trailing window-vs-window block.
    mask = _causal_window_mask(window_size, attn_weights.dtype, attn_weights.device)
    attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    # Sum the attention each past key receives across observation rows.
    attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim=-2)
    return attn_weights_sum


def observation_attn_cache(
    query_states, key_states, window_size, kernel_size, pooling="avgpool"
):
    """Smoothed per-past-token SnapKV importance ``attn_cache``.

    Applies the observation-window attention (:func:`compute_snap_attention`,
    query-group pooled with ``max``) followed by 1-D pooling over the sequence
    dimension (the ``avgpool`` / ``maxpool`` clustering that gives SnapKV its
    name — it clusters attention mass over neighbouring tokens).

    Returns a tensor of shape ``(bsz, kv_heads, kv_len - window_size)``.
    """
    attn_weights_sum = compute_snap_attention(
        query_states, key_states, window_size, pooling="max"
    )
    if pooling == "avgpool":
        attn_cache = F.avg_pool1d(
            attn_weights_sum,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    elif pooling == "maxpool":
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
    else:
        raise ValueError("Pooling method not supported")
    return attn_cache


class SnapKVCluster:
    """SnapKV prompt-phase KV cache compressor.

    Keeps the ``max_capacity_prompt - window_size`` highest-scoring past prompt
    tokens (by pooled observation-window attention) plus the trailing
    ``window_size`` observation tokens (always retained).
    """

    def __init__(
        self,
        max_capacity_prompt=1024,
        window_size=32,
        kernel_size=5,
        pooling="avgpool",
        **kwargs,
    ):
        assert (
            max_capacity_prompt - window_size > 0
        ), "max_capacity_prompt must be greater than window_size"
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling

    def select_indices(self, key_states, query_states, sort=True):
        """Return the token indices to keep, per (batch, kv head).

        This is the primitive the SGLang integration layer needs: it answers
        "which prompt tokens survive compression" without materialising new K/V
        tensors. The result has shape ``(bsz, kv_heads, max_capacity_prompt)``
        with indices into ``[0, kv_len)``.

        The kept set is the ``max_capacity_prompt - window_size`` top-scoring
        past tokens plus the trailing ``window_size`` window tokens. When
        ``sort`` is True the indices are returned in ascending (temporal) order,
        which is what the paged-cache compaction wants; ordering is otherwise
        semantically irrelevant because rotary position information is already
        baked into the stored keys.

        Returns ``None`` when ``kv_len <= max_capacity_prompt`` (no compression).
        """
        kv_len = key_states.shape[-2]
        if kv_len <= self.max_capacity_prompt:
            return None

        attn_cache = observation_attn_cache(
            query_states,
            key_states,
            self.window_size,
            self.kernel_size,
            self.pooling,
        )
        past_indices = attn_cache.topk(
            self.max_capacity_prompt - self.window_size, dim=-1
        ).indices

        bsz, kv_heads = past_indices.shape[0], past_indices.shape[1]
        window_indices = (
            torch.arange(
                kv_len - self.window_size,
                kv_len,
                device=past_indices.device,
            )
            .view(1, 1, -1)
            .expand(bsz, kv_heads, -1)
        )

        kept = torch.cat([past_indices, window_indices], dim=-1)
        if sort:
            kept, _ = torch.sort(kept, dim=-1)
        return kept

    def update_kv(self, key_states, query_states, value_states):
        """Faithful reference-compatible compression.

        Returns the compressed ``(key_states, value_states)``. When the cache is
        at or below ``max_capacity_prompt`` the inputs are returned unchanged.
        Token order matches the reference implementation (top-scoring past
        tokens in score order, followed by the observation window).

        Mirrors ``SnapKVCluster.update_kv`` from the reference repo, which is
        invoked on already-``repeat_kv``-expanded tensors
        (``query_states`` and ``key_states`` share the head count). The
        grouped-query path lives in :meth:`select_indices` /
        :func:`observation_attn_cache` for the SGLang integration.
        """
        head_dim = query_states.shape[-1]
        kv_len = key_states.shape[-2]

        if kv_len <= self.max_capacity_prompt:
            return key_states, value_states

        attn_cache = observation_attn_cache(
            query_states,
            key_states,
            self.window_size,
            self.kernel_size,
            self.pooling,
        )
        indices = attn_cache.topk(
            self.max_capacity_prompt - self.window_size, dim=-1
        ).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states
