"""Fused Triton redundancy kernel for R-KV prefill (Stage 1: no retain-exemption).

Computes the R-KV redundancy signal::

    redundancy = softmax_j( (1/n) * sum_i simcos(i, j) )

where ``simcos`` is the normalized-key cosine similarity with the diagonal
zeroed, *without* the per-row "retain the most-recent above-threshold
neighbour" exemption. The ``n x n`` similarity matrix is never materialised:
each ``(row-block, col-block)`` tile is formed in registers via ``tl.dot`` and
reduced into a per-column running sum, so HBM traffic is ``O(n*d)`` reads of the
(pre-normalized) keys plus an ``O(n)`` write of the column sums — instead of the
``O(n^2)`` write+reread of the materialised matrix in ``cal_similarity_tiled``.

The kernel is **col-parallel**: one program owns ``BLOCK_N`` output columns and
loops over all rows, accumulating the column sum locally (no atomics). Stage 2
adds the retain exemption as a cheap ``O(n)``-atomics correction to reach
bit-parity with ``algo.cal_similarity`` / ``cal_similarity_tiled``.

This module imports ``triton`` at load time, so it is only imported on the CUDA
serving path; the CPU-by-path unit tests exercise ``cal_similarity_tiled``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _colsum_kernel(
    K_ptr,  # (H, N, D) normalized keys
    OUT_ptr,  # (H, N) fp32 column sums
    N,
    D,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_oh,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    col_pid = tl.program_id(1)

    cols = col_pid * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    col_mask = cols < N
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    # Columns this program owns: K[h, cols, :] -> (BLOCK_N, BLOCK_D)
    kcol = tl.load(
        K_ptr + h * stride_kh + cols[:, None] * stride_kn + d[None, :] * stride_kd,
        mask=col_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    kcol_t = tl.trans(kcol)  # (BLOCK_D, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row0 in range(0, N, BLOCK_M):
        rows = row0 + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        row_mask = rows < N
        krow = tl.load(
            K_ptr + h * stride_kh + rows[:, None] * stride_kn + d[None, :] * stride_kd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # (BLOCK_M, BLOCK_D)
        sim = tl.dot(krow, kcol_t, input_precision="ieee")  # (BLOCK_M, BLOCK_N) fp32
        # Zero the diagonal (self-similarity) and any padded rows/cols.
        diag = rows[:, None] == cols[None, :]
        drop = diag | (~row_mask[:, None]) | (~col_mask[None, :])
        sim = tl.where(drop, 0.0, sim)
        acc += tl.sum(sim, axis=0)  # (BLOCK_N,)

    tl.store(OUT_ptr + h * stride_oh + cols * stride_on, acc, mask=col_mask)


def cal_similarity_fused_noretain(
    key_states: torch.Tensor,
    block_n: int = 128,
    block_m: int = 64,
) -> torch.Tensor:
    """Fused redundancy WITHOUT the retain exemption.

    ``key_states``: ``(bsz, kv_heads, n, d)``. Returns ``(bsz, kv_heads, n)``.
    Matches a reference that zeroes only the diagonal (no ``similarity_retain``).
    """
    bsz, kv_heads, n, d = key_states.shape
    h = bsz * kv_heads
    k = key_states.reshape(h, n, d)
    k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-8)).contiguous()
    out = torch.empty((h, n), dtype=torch.float32, device=k.device)

    block_d = triton.next_power_of_2(d)
    grid = (h, triton.cdiv(n, block_n))
    _colsum_kernel[grid](
        k_norm,
        out,
        n,
        d,
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    redundancy = (out / n).softmax(dim=-1)
    return redundancy.reshape(bsz, kv_heads, n).to(key_states.dtype)
