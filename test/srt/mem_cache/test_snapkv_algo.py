"""CPU parity tests for the pure SnapKV algorithm port.

These tests are GPU-free and self-contained: an inline *reference* copy of the
original SnapKV algorithm (from ``snapkv/monkeypatch/snapkv_utils.py``, class
``SnapKVCluster.update_kv``) is compared against
``sglang.srt.mem_cache.snapkv.algo``.

The reference operates on already-``repeat_kv``-expanded tensors
(``q_heads == kv_heads``); parity is therefore checked in that head-matched
setting. A separate grouped-query sanity check exercises the pooled path.

Run directly::

    python test/srt/mem_cache/test_snapkv_algo.py

or under a test runner (unittest / pytest).
"""

import importlib.util
import math
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the pure algorithm module directly by file path. It is intentionally free
# of SGLang dependencies, so we bypass ``sglang/__init__.py`` (which pulls in the
# full serving stack) to keep this test GPU-free and dependency-light.
_ALGO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "python",
    "sglang",
    "srt",
    "mem_cache",
    "snapkv",
    "algo.py",
)
_spec = importlib.util.spec_from_file_location(
    "snapkv_algo", os.path.abspath(_ALGO_PATH)
)
_algo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algo)
SnapKVCluster = _algo.SnapKVCluster


# --------------------------------------------------------------------------- #
# Reference implementation (verbatim from the SnapKV repo, MHA / post-repeat)  #
# --------------------------------------------------------------------------- #
def ref_update_kv(
    key_states,
    query_states,
    value_states,
    window_size,
    max_capacity_prompt,
    kernel_size,
    pooling,
):
    assert key_states.shape[-2] == query_states.shape[-2]
    bsz, num_heads, q_len, head_dim = query_states.shape
    if q_len < max_capacity_prompt:
        return key_states, value_states

    attn_weights = torch.matmul(
        query_states[..., -window_size:, :], key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)
    mask = torch.full(
        (window_size, window_size),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
    )
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    attention_mask = mask[None, None, :, :]
    attn_weights[:, :, -window_size:, -window_size:] += attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim=-2)
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
    indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]
    key_states = torch.cat([k_past_compress, k_cur], dim=2)
    value_states = torch.cat([v_past_compress, v_cur], dim=2)
    return key_states, value_states


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
class TestSnapKVParity(unittest.TestCase):
    def _run(
        self, bsz, heads, seq_len, head_dim, window, budget, kernel, pooling, dtype
    ):
        torch.manual_seed(0)
        q = torch.randn(bsz, heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(bsz, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(bsz, heads, seq_len, head_dim, dtype=dtype)

        ref_k, ref_v = ref_update_kv(k, q, v, window, budget, kernel, pooling)
        cluster = SnapKVCluster(
            max_capacity_prompt=budget,
            window_size=window,
            kernel_size=kernel,
            pooling=pooling,
        )
        out_k, out_v = cluster.update_kv(k, q, v)

        self.assertEqual(out_k.shape, ref_k.shape)
        self.assertTrue(
            torch.equal(out_k, ref_k), "compressed K differs from reference"
        )
        self.assertTrue(
            torch.equal(out_v, ref_v), "compressed V differs from reference"
        )

    def test_parity_mha_avgpool(self):
        self._run(
            1,
            4,
            128,
            16,
            window=8,
            budget=32,
            kernel=5,
            pooling="avgpool",
            dtype=torch.float32,
        )

    def test_parity_mha_maxpool(self):
        self._run(
            1,
            4,
            128,
            16,
            window=8,
            budget=32,
            kernel=7,
            pooling="maxpool",
            dtype=torch.float32,
        )

    def test_parity_batch(self):
        self._run(
            3,
            2,
            96,
            8,
            window=16,
            budget=48,
            kernel=5,
            pooling="avgpool",
            dtype=torch.float32,
        )

    def test_below_budget_is_noop(self):
        torch.manual_seed(1)
        q = torch.randn(1, 4, 20, 16)
        k = torch.randn(1, 4, 20, 16)
        v = torch.randn(1, 4, 20, 16)
        cluster = SnapKVCluster(max_capacity_prompt=64, window_size=8, kernel_size=5)
        out_k, out_v = cluster.update_kv(k, q, v)
        self.assertTrue(torch.equal(out_k, k))
        self.assertTrue(torch.equal(out_v, v))
        self.assertIsNone(cluster.select_indices(k, q))


class TestSelectIndices(unittest.TestCase):
    def test_select_indices_keeps_window_and_budget(self):
        torch.manual_seed(2)
        bsz, heads, seq_len, head_dim = 1, 2, 100, 16
        window, budget = 8, 32
        q = torch.randn(bsz, heads, seq_len, head_dim)
        k = torch.randn(bsz, heads, seq_len, head_dim)
        cluster = SnapKVCluster(
            max_capacity_prompt=budget, window_size=window, kernel_size=5
        )
        kept = cluster.select_indices(k, q)
        self.assertEqual(tuple(kept.shape), (bsz, heads, budget))
        # ascending order
        self.assertTrue(torch.equal(kept, torch.sort(kept, dim=-1).values))
        # trailing window always retained
        for h in range(heads):
            tail = set(range(seq_len - window, seq_len))
            self.assertTrue(tail.issubset(set(kept[0, h].tolist())))

    def test_grouped_query_shapes(self):
        torch.manual_seed(3)
        bsz, kv_heads, group, seq_len, head_dim = 1, 2, 4, 80, 16
        q_heads = kv_heads * group
        window, budget = 8, 32
        q = torch.randn(bsz, q_heads, seq_len, head_dim)
        k = torch.randn(bsz, kv_heads, seq_len, head_dim)
        cluster = SnapKVCluster(
            max_capacity_prompt=budget, window_size=window, kernel_size=5
        )
        kept = cluster.select_indices(k, q)
        self.assertEqual(tuple(kept.shape), (bsz, kv_heads, budget))


if __name__ == "__main__":
    unittest.main(verbosity=2)
