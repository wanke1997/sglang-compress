"""GPU parity tests for the fused R-KV redundancy kernel.

These require CUDA + Triton and are skipped otherwise (the CPU-by-path suite in
``test_rkv_prefill.py`` covers the reference ``cal_similarity_tiled``). Stage 1
checks the fused kernel WITHOUT the retain exemption against a no-retain torch
reference; Stage 2 will add the retain exemption and bit-parity vs the full
reference.

Run directly (on a GPU box)::

    python test/srt/mem_cache/test_rkv_redundancy_fused.py
"""

import importlib.util
import os
import unittest

import torch

_HERE = os.path.dirname(__file__)
_RKV_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "python", "sglang", "srt", "mem_cache", "rkv")
)

_HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_RKV_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_noretain(key: torch.Tensor) -> torch.Tensor:
    """Redundancy with only the diagonal zeroed (no retain exemption)."""
    k_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    n = key.shape[-2]
    diag = torch.eye(n, dtype=torch.bool, device=key.device)
    sim.masked_fill_(diag.view(1, 1, n, n), 0.0)
    return sim.to(torch.float32).mean(dim=-2).softmax(dim=-1).to(key.dtype)


@unittest.skipUnless(
    _HAS_CUDA and _HAS_TRITON, "fused redundancy kernel needs CUDA + Triton"
)
class TestFusedRedundancyNoRetain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fused = staticmethod(
            _load(
                "rkv_redundancy_fused", "redundancy_fused.py"
            ).cal_similarity_fused_noretain
        )

    def test_parity_fp32(self):
        for n in (37, 128, 512, 2000):
            torch.manual_seed(0)
            key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.float32)
            ref = _ref_noretain(key)
            got = self.fused(key)
            self.assertEqual(ref.shape, got.shape)
            self.assertTrue(
                torch.allclose(ref, got, atol=2e-4, rtol=1e-3),
                f"fp32 fused diverged at n={n}: "
                f"max {(ref - got).abs().max().item():.3e}",
            )

    def test_parity_bf16(self):
        for n in (128, 512, 2000):
            torch.manual_seed(1)
            key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.bfloat16)
            ref = _ref_noretain(key)
            got = self.fused(key)
            self.assertTrue(
                torch.allclose(ref.float(), got.float(), atol=2e-2, rtol=1e-3),
                f"bf16 fused diverged at n={n}: "
                f"max {(ref.float() - got.float()).abs().max().item():.3e}",
            )

    def test_multi_layer_leading_dims(self):
        # Leading (layers, kv_heads) dims are flattened and reduced independently.
        torch.manual_seed(2)
        key = torch.randn(3, 4, 200, 128, device="cuda", dtype=torch.float32)
        got = self.fused(key)
        self.assertEqual(got.shape, (3, 4, 200))
        # Each (layer, head) row is a probability distribution over n.
        sums = got.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


@unittest.skipUnless(
    _HAS_CUDA and _HAS_TRITON, "fused redundancy kernel needs CUDA + Triton"
)
class TestFusedRedundancyRetain(unittest.TestCase):
    """Fused kernel WITH the retain exemption must match cal_similarity_tiled."""

    @classmethod
    def setUpClass(cls):
        cls.fused = staticmethod(
            _load("rkv_redundancy_fused", "redundancy_fused.py").cal_similarity_fused
        )
        cls.tiled = staticmethod(
            _load("rkv_prefill", "prefill.py").cal_similarity_tiled
        )

    def test_bitparity_fp32(self):
        for thr in (0.0, 0.3, 0.5, 0.9):
            for n in (37, 128, 512, 1500):
                torch.manual_seed(n)
                key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.float32)
                ref = self.tiled(key, threshold=thr, retain_direction="last")
                got = self.fused(key, threshold=thr)
                self.assertTrue(
                    torch.allclose(ref, got, atol=5e-4, rtol=1e-3),
                    f"fp32 retain diverged thr={thr} n={n}: "
                    f"max {(ref - got).abs().max().item():.3e}",
                )

    def test_parity_bf16(self):
        for thr in (0.3, 0.5):
            for n in (128, 512, 1500):
                torch.manual_seed(n)
                key = torch.randn(1, 4, n, 128, device="cuda", dtype=torch.bfloat16)
                ref = self.tiled(key, threshold=thr, retain_direction="last")
                got = self.fused(key, threshold=thr)
                self.assertTrue(
                    torch.allclose(ref.float(), got.float(), atol=3e-2, rtol=1e-3),
                    f"bf16 retain diverged thr={thr} n={n}: "
                    f"max {(ref.float() - got.float()).abs().max().item():.3e}",
                )

    def test_no_neighbor_defaults_to_col0(self):
        # A very high threshold -> no above-threshold neighbours -> retain=col 0
        # for every row (the reference's default). Must still match tiled.
        torch.manual_seed(7)
        key = torch.randn(1, 4, 300, 128, device="cuda", dtype=torch.float32)
        ref = self.tiled(key, threshold=0.99, retain_direction="last")
        got = self.fused(key, threshold=0.99)
        self.assertTrue(torch.allclose(ref, got, atol=5e-4, rtol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
