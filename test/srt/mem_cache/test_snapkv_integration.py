"""CPU unit tests for the SnapKV integration layer's physical compaction.

GPU-free and do not require an installed ``sglang`` package. The pure algorithm
module and the integration module are loaded directly by file path (the
integration module's absolute ``from sglang...algo import observation_attn_cache``
is satisfied by pre-registering the algo module under its real dotted name in
``sys.modules``), so we bypass ``sglang/__init__.py`` and the serving stack.

Run directly::

    python test/srt/mem_cache/test_snapkv_integration.py

or under a test runner (unittest / pytest).
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SNAPKV_DIR = os.path.abspath(
    os.path.join(
        _HERE, "..", "..", "..", "python", "sglang", "srt", "mem_cache", "snapkv"
    )
)


def _load_by_path(dotted_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.snapkv",
):
    if _pkg not in sys.modules:
        _placeholder = types.ModuleType(_pkg)
        _placeholder.__path__ = []  # mark as package
        sys.modules[_pkg] = _placeholder

_load_by_path("sglang.srt.mem_cache.snapkv.algo", os.path.join(_SNAPKV_DIR, "algo.py"))
_integration = _load_by_path(
    "sglang.srt.mem_cache.snapkv.integration",
    os.path.join(_SNAPKV_DIR, "integration.py"),
)

SnapKVConfig = _integration.SnapKVConfig
SnapKVRequestState = _integration.SnapKVRequestState
SnapKVCompressor = _integration.SnapKVCompressor


# --------------------------------------------------------------------------- #
# Minimal mocks for the SGLang pools / allocator / forward batch              #
# --------------------------------------------------------------------------- #
class MockReqToTokenPool:
    def __init__(self, num_reqs: int, max_ctx: int, device: torch.device):
        self.req_to_token = torch.zeros(
            (num_reqs + 1, max_ctx), dtype=torch.int32, device=device
        )


class MockKVPool:
    """Per-layer NHD buffers: (num_slots, head_num, head_dim)."""

    def __init__(self, num_layers, num_slots, head_num, head_dim, device, dtype):
        self.k_buffers = [
            torch.zeros((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffers = [
            torch.zeros((num_slots, head_num, head_dim), device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffers[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffers[layer_id]


class MockAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor) -> None:
        self.freed.append(free_index.clone())


class _MockReq:
    def __init__(self, origin_len: int, output_len: int, req_pool_idx: int = 1):
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.kv_committed_len = origin_len + output_len
        self.kv_allocated_len = origin_len + output_len
        self.req_pool_idx = req_pool_idx

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)


class _MockLayer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id


class _MockForwardBatch:
    def __init__(self, req_pool_indices, seq_lens, extend_seq_lens, positions=None):
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
        self.seq_lens_cpu = self.seq_lens.clone()
        self.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int64)
        self.extend_seq_lens_cpu = extend_seq_lens
        self.positions = positions


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
class TestAssembleKept(unittest.TestCase):
    def _compressor(self, budget, window):
        cfg = SnapKVConfig(max_capacity_prompt=budget, window_size=window)
        return SnapKVCompressor(
            config=cfg,
            req_to_token_pool=MockReqToTokenPool(1, 64, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                1, 64, 1, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
        )

    def test_kept_is_top_past_plus_window_sorted(self):
        budget, window, seq_len = 4, 2, 6
        comp = self._compressor(budget, window)
        score_accum = torch.tensor([0.1, 9.0, 0.2, 8.0])  # past indices [0,4)
        kept = comp._assemble_kept(score_accum, seq_len)
        self.assertEqual(kept.numel(), budget)
        self.assertTrue(torch.equal(kept, torch.sort(kept).values))
        self.assertEqual(sorted(kept.tolist()), [1, 3, 4, 5])


class TestCompactRequest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_layers = 2
        self.head_num = 2
        self.head_dim = 4
        self.budget = 4
        self.window = 2
        self.seq_len = 6
        self.req_pool_idx = 1
        self.num_slots = 32

        self.r2t_pool = MockReqToTokenPool(4, 64, self.device)
        self.kv_pool = MockKVPool(
            self.num_layers,
            self.num_slots,
            self.head_num,
            self.head_dim,
            self.device,
            self.dtype,
        )
        self.alloc = MockAllocator()
        cfg = SnapKVConfig(max_capacity_prompt=self.budget, window_size=self.window)
        self.comp = SnapKVCompressor(
            config=cfg,
            req_to_token_pool=self.r2t_pool,
            token_to_kv_pool=self.kv_pool,
            kv_allocator=self.alloc,
            start_layer=0,
            end_layer=self.num_layers,
            device=self.device,
        )

        self.slots = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        self.r2t_pool.req_to_token[self.req_pool_idx, : self.seq_len] = self.slots
        for layer in range(self.num_layers):
            kb = self.kv_pool.get_key_buffer(layer)
            vb = self.kv_pool.get_value_buffer(layer)
            for s in self.slots.tolist():
                base = s * 100 + layer
                pattern = base + 0.01 * (
                    torch.arange(self.head_num).view(-1, 1)
                    + torch.arange(self.head_dim).view(1, -1)
                )
                kb[s] = pattern
                vb[s] = pattern + 0.5

        self.state = SnapKVRequestState(req_pool_idx=self.req_pool_idx)
        self.state.req = _MockReq(
            origin_len=self.seq_len, output_len=0, req_pool_idx=self.req_pool_idx
        )

    def test_relocation_free_and_rewrite(self):
        kept_local = torch.tensor([0, 2, 4, 5])
        src = self.slots[kept_local]  # [10, 12, 14, 15]
        dst = self.slots[: self.budget]  # [10, 11, 12, 13]
        expected_k = [
            self.kv_pool.get_key_buffer(l)[src].clone() for l in range(self.num_layers)
        ]
        expected_v = [
            self.kv_pool.get_value_buffer(l)[src].clone()
            for l in range(self.num_layers)
        ]

        self.comp._compact_request(self.state, self.seq_len, kept_local)

        for l in range(self.num_layers):
            self.assertTrue(
                torch.equal(self.kv_pool.get_key_buffer(l)[dst], expected_k[l])
            )
            self.assertTrue(
                torch.equal(self.kv_pool.get_value_buffer(l)[dst], expected_v[l])
            )
        # tail freed once with [14, 15]
        self.assertEqual(len(self.alloc.freed), 1)
        self.assertEqual(sorted(self.alloc.freed[0].tolist()), [14, 15])
        # req_to_token tail cleared, physical length bookkeeping updated
        self.assertTrue(
            torch.all(
                self.r2t_pool.req_to_token[
                    self.req_pool_idx, self.budget : self.seq_len
                ]
                == 0
            )
        )
        self.assertEqual(self.state.req.kv_committed_len, self.budget)
        self.assertEqual(self.state.req.kv_allocated_len, self.budget)
        self.assertTrue(self.state.compressed)
        self.assertEqual(
            self.comp.pending_length_updates[self.req_pool_idx], self.budget
        )


class TestObserveAndCompactEndToEnd(unittest.TestCase):
    """Drive observe_prefill_layer over a full prompt then compact."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_layers = 2
        self.kv_heads = 2
        self.q_heads = 2
        self.head_dim = 8
        self.budget = 16
        self.window = 4
        self.seq_len = 40  # prompt length > budget -> compresses
        self.req_pool_idx = 1
        self.num_slots = 128

        self.r2t_pool = MockReqToTokenPool(4, 256, self.device)
        self.kv_pool = MockKVPool(
            self.num_layers,
            self.num_slots,
            self.kv_heads,
            self.head_dim,
            self.device,
            self.dtype,
        )
        self.alloc = MockAllocator()
        cfg = SnapKVConfig(
            max_capacity_prompt=self.budget,
            window_size=self.window,
            kernel_size=5,
            pooling="avgpool",
        )
        self.comp = SnapKVCompressor(
            config=cfg,
            req_to_token_pool=self.r2t_pool,
            token_to_kv_pool=self.kv_pool,
            kv_allocator=self.alloc,
            start_layer=0,
            end_layer=self.num_layers,
            device=self.device,
        )
        # request occupies slots [0, seq_len)
        self.slots = torch.arange(self.seq_len, dtype=torch.int32)
        self.r2t_pool.req_to_token[self.req_pool_idx, : self.seq_len] = self.slots
        torch.manual_seed(7)
        for layer in range(self.num_layers):
            self.kv_pool.get_key_buffer(layer)[: self.seq_len] = torch.randn(
                self.seq_len, self.kv_heads, self.head_dim
            )
            self.kv_pool.get_value_buffer(layer)[: self.seq_len] = torch.randn(
                self.seq_len, self.kv_heads, self.head_dim
            )

    def test_prefill_observe_then_compact(self):
        req = _MockReq(
            origin_len=self.seq_len, output_len=0, req_pool_idx=self.req_pool_idx
        )
        self.comp.on_request_begin(req)

        fb = _MockForwardBatch(
            req_pool_indices=[self.req_pool_idx],
            seq_lens=[self.seq_len],
            extend_seq_lens=[self.seq_len],
        )
        torch.manual_seed(11)
        for layer_id in range(self.num_layers):
            q = torch.randn(self.seq_len, self.q_heads, self.head_dim)
            self.comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)

        state = self.comp.states[self.req_pool_idx]
        self.assertIsNotNone(state.score_accum)
        self.assertEqual(state.score_accum.numel(), self.seq_len - self.window)

        self.comp.maybe_compact(fb)

        self.assertTrue(state.compressed)
        # freed exactly seq_len - budget slots
        self.assertEqual(len(self.alloc.freed), 1)
        self.assertEqual(self.alloc.freed[0].numel(), self.seq_len - self.budget)
        # physical length shrunk to budget
        self.assertEqual(req.kv_committed_len, self.budget)
        self.assertEqual(
            self.comp.pending_length_updates[self.req_pool_idx], self.budget
        )
        # armed set cleared
        self.assertEqual(len(self.comp._armed), 0)

    def test_reprefill_after_retract_frees_regenerated_slots(self):
        # A retracted request keeps its output_ids and re-prefills
        # origin_input_ids + output_ids, allocating (origin + output) KV slots.
        # Compaction must free the FULL physical tail, not orphan the
        # regenerated output slots (KV-pool leak regression).
        origin, output = self.seq_len, 12
        n = origin + output  # physical prefill length after re-prefill
        slots = torch.arange(n, dtype=torch.int32)
        self.r2t_pool.req_to_token[self.req_pool_idx, :n] = slots

        req = _MockReq(
            origin_len=origin, output_len=output, req_pool_idx=self.req_pool_idx
        )
        self.comp.on_request_begin(req)
        # prompt_len tracks the full physical length (origin + output).
        self.assertEqual(self.comp.states[self.req_pool_idx].prompt_len, n)

        fb = _MockForwardBatch(
            req_pool_indices=[self.req_pool_idx],
            seq_lens=[n],
            extend_seq_lens=[n],
        )
        torch.manual_seed(11)
        for layer_id in range(self.num_layers):
            q = torch.randn(n, self.q_heads, self.head_dim)
            self.comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        self.comp.maybe_compact(fb)

        # Frees the full tail (n - budget); no output slot left orphaned.
        self.assertEqual(len(self.alloc.freed), 1)
        self.assertEqual(self.alloc.freed[0].numel(), n - self.budget)
        r2t = self.r2t_pool.req_to_token
        self.assertTrue(bool((r2t[self.req_pool_idx, self.budget : n] == 0).all()))
        self.assertEqual(req.kv_committed_len, self.budget)
        self.assertEqual(req.kv_allocated_len, self.budget)

    def test_below_budget_prompt_is_untouched(self):
        short = 10  # < budget
        req = _MockReq(origin_len=short, output_len=0, req_pool_idx=self.req_pool_idx)
        self.comp.on_request_begin(req)
        fb = _MockForwardBatch(
            req_pool_indices=[self.req_pool_idx],
            seq_lens=[short],
            extend_seq_lens=[short],
        )
        for layer_id in range(self.num_layers):
            q = torch.randn(short, self.q_heads, self.head_dim)
            self.comp.observe_prefill_layer(q, None, None, _MockLayer(layer_id), fb)
        self.comp.maybe_compact(fb)
        self.assertFalse(self.comp.states[self.req_pool_idx].compressed)
        self.assertEqual(len(self.alloc.freed), 0)


class TestLifecycleAndPositions(unittest.TestCase):
    def _comp(self):
        cfg = SnapKVConfig(max_capacity_prompt=8, window_size=2)
        return SnapKVCompressor(
            config=cfg,
            req_to_token_pool=MockReqToTokenPool(4, 64, torch.device("cpu")),
            token_to_kv_pool=MockKVPool(
                1, 64, 1, 4, torch.device("cpu"), torch.float32
            ),
            kv_allocator=MockAllocator(),
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
        )

    def test_begin_end(self):
        comp = self._comp()
        req = _MockReq(origin_len=20, output_len=0, req_pool_idx=2)
        comp.on_request_begin(req)
        self.assertIn(2, comp.states)
        comp.on_request_end(req)
        self.assertNotIn(2, comp.states)

    def test_override_decode_positions_logical(self):
        comp = self._comp()
        req = _MockReq(origin_len=30, output_len=5, req_pool_idx=3)
        comp.on_request_begin(req)
        fb = _MockForwardBatch(
            req_pool_indices=[3],
            seq_lens=[8],  # physical (post-compaction budget)
            extend_seq_lens=[0],
            positions=torch.tensor([7], dtype=torch.int64),  # baseline seq_lens-1
        )
        comp.override_decode_positions(fb)
        # logical position = origin(30) + output(5) - 1 = 34
        self.assertEqual(int(fb.positions[0].item()), 34)


class TestRequestWantsCompression(unittest.TestCase):
    """The per-request gate: only task_type == 'summarization' opts in."""

    def _req(self, task_type):
        return types.SimpleNamespace(task_type=task_type, req_pool_idx=1)

    def test_summarization_opts_in(self):
        self.assertTrue(
            SnapKVCompressor.request_wants_compression(self._req("summarization"))
        )

    def test_case_and_whitespace_insensitive(self):
        for val in (" Summarization ", "SUMMARIZATION", "summarization\n"):
            self.assertTrue(
                SnapKVCompressor.request_wants_compression(self._req(val)),
                f"should match: {val!r}",
            )

    def test_other_or_missing_stays_full_kv(self):
        for val in (None, "", "classification", "choose", "summarize"):
            self.assertFalse(
                SnapKVCompressor.request_wants_compression(self._req(val)),
                f"should NOT match: {val!r}",
            )

    def test_req_without_task_type_attr(self):
        # A Req lacking the attribute entirely must not raise -> full KV.
        self.assertFalse(
            SnapKVCompressor.request_wants_compression(
                types.SimpleNamespace(req_pool_idx=1)
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
