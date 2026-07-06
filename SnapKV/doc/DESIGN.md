# SnapKV on SGLang v0.5.14 — Design & Porting Notes

Hand-off reference for the SnapKV port. It captures the design decisions that
are not obvious from the code alone. SnapKV reuses most of the paged-pool
eviction machinery pioneered by the R-KV port, so read
[`../../R-KV/doc/DESIGN.md`](../../R-KV/doc/DESIGN.md) first — this document only
covers what is *different* for SnapKV.

## 1. Goal

Port **SnapKV** ([Li et al., 2024](https://arxiv.org/abs/2404.14469)) onto this
SGLang v0.5.14 tree. SnapKV is a **prompt-phase** (prefill-time) KV-cache
compressor: after a long prompt is processed, it keeps only the
`max_capacity_prompt` prompt tokens that the trailing observation window attends
to most, and frees the rest, so a long prompt costs a fixed small KV budget.

**Source of truth for the algorithm** is the reference repo, file
`snapkv/monkeypatch/snapkv_utils.py` (class `SnapKVCluster.update_kv`).

## 2. The SnapKV algorithm (what `algo.py` implements)

Given the per-request `key/query_states` of shape
`(bsz, heads, seq_len, head_dim)`, once `seq_len > max_capacity_prompt`:

1. **Observation window** — take the last `window_size` prompt queries.
2. **Importance** — attend those queries over *all* prompt keys
   (`q_window @ kᵀ / √d`), apply a causal mask on the window-vs-window block,
   softmax over the key axis, and **sum** the attention each past key receives
   across the observation rows → `attn_weights_sum`, shape
   `(bsz, kv_heads, seq_len - window)`. Grouped-query attention is handled by
   pooling per query group (`compute_snap_attention`).
3. **Clustering** — smooth with `avg_pool1d` / `max_pool1d` of width
   `kernel_size` (`observation_attn_cache`) so informative *spans* survive, not
   isolated tokens.
4. **Selection** — keep the top `max_capacity_prompt - window_size` past tokens
   **plus** the trailing `window_size` observation tokens (always kept).

Defaults (reference): `max_capacity_prompt=1024`, `window_size=32`,
`kernel_size=5`, `pooling="avgpool"`.

`algo.py` exposes both `update_kv` (returns compacted K/V, reference-compatible
for bit-level parity) and `select_indices` (returns only the kept indices —
what the integration needs to relocate slots in a paged pool).

## 3. Two-layer design

Same split as R-KV: a pure, device-agnostic algorithm layer (`algo.py`,
CPU-testable, bit-parity vs the reference) and an integration layer
(`integration.py`) that bridges to SGLang's paged pool and FlashInfer.

## 4. Injection point — **prefill**, not decode

This is the one real structural difference from R-KV.

- **Observe** — `SnapKVCompressor.observe_prefill_layer` is called from
  `FlashInferAttnBackend.forward_extend`, *after* `set_kv_buffer` (the prompt K/V
  is in the pool) and *before* the attention wrapper runs. For each request it
  slices the last `window_size` queries out of the extend query block (via
  `extend_seq_lens` / `extend_start_loc`), reads the prompt keys back from the
  pool (`req_to_token` slots), computes this layer's `attn_cache`, reduces it
  across kv-heads (**mean**) and accumulates it across layers (**sum**).
- **Compact** — after the *full* prefill forward, `ModelRunner` calls
  `maybe_compact` (guarded by `forward_mode.is_extend()`). For every request
  whose prompt exceeds the budget it selects `top-k past + window`, relocates the
  surviving KV slots to the front `max_capacity_prompt` slots of every layer,
  `free()`s the tail, rewrites `req_to_token`, and shrinks the physical length.

Compression fires **once** per request (a `compressed` flag guards against
re-triggering). R-KV, by contrast, hooks `forward_decode` and fires every
`buffer_size` steps.

## 5. Position / rotary handling (shared "scheme A")

Identical to R-KV: after compaction `seq_lens` tracks the *physical* (shrunk) KV
length so new decode tokens append at the physical tail and attention reads the
right length. Rotary stays *logical*: `override_decode_positions` sets each
SnapKV request's decode position to `len(origin_input_ids) + len(output_ids) - 1`
(the 0-based position of the token being decoded) instead of
`clamp_position(seq_lens)`, so decode tokens keep absolute positions consistent
with the retained prompt keys (whose rotary was baked in at their original
positions). The scheduler applies the pending physical-length shrink to the
batch `seq_lens` before `prepare_for_decode` (`_apply_snapkv_pre_decode`).

## 6. Why chunked prefill must be off

SnapKV scores the prompt with the trailing observation-window queries. Those
queries only exist in the forward pass that processes the tail of the prompt. To
keep phase 1 simple and correct we require `--chunked-prefill-size -1` so the
whole prompt (queries + keys) is seen in a single forward; `observe_prefill_layer`
skips any request whose `extend_len < seq_len` as a guard.

## 7. Wiring (5 files, additive alongside R-KV)

- `server_args.py` — `--enable-snapkv` + `--snapkv-*` flags +
  `_handle_snapkv_validation` (mirrors `_handle_rkv_validation`, plus the
  chunked-prefill-off check and an rkv/snapkv mutual-exclusion check).
- `model_executor/model_runner.py` — construct `SnapKVCompressor` in
  `alloc_memory_pool`; call `maybe_compact` after an extend forward.
- `managers/scheduler.py` — bind the compressor; register requests right after
  `prepare_for_extend` (so `observe_prefill_layer` finds their state); drain
  pending physical-length shrinks in `_apply_snapkv_pre_decode`.
- `layers/attention/flashinfer_backend.py` — hold the compressor; observe in
  `forward_extend`; override decode positions in `init_forward_metadata`.
- `managers/scheduler_components/batch_result_processor.py` — `on_request_end`
  cleanup at the two request-finished points.

## 8. Status

- **[DONE]** Algorithm layer (`algo.py`) + CPU parity tests (bit-for-bit vs the
  reference `SnapKVCluster`, incl. GQA and below-budget no-op).
- **[DONE]** Integration layer + wiring, GPU-validated on Qwen2.5-0.5B
  (FlashInfer, `page_size=1`, radix/CUDA-graph/overlap/chunked-prefill off): a
  16 405-token prompt compresses to 256 (64×) and the needle is retrieved at
  every depth, matching baseline. See
  [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md).
- **[LATER]** TP ≥ 2, chunked-prefill support (score on the final chunk),
  batch > 1 accuracy sweeps on LongBench, performance (avoid the score-time
  read-back, reduce host/device syncs).
