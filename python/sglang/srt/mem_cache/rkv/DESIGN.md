# R-KV on SGLang v0.5.14 — Design & Porting Notes

This document is the hand-off reference for anyone (human or AI agent) continuing
the R-KV port. It captures the architecture research and design decisions that
are **not** obvious from the code alone. Read this first before touching the
integration layer.

## 1. Goal

Port **R-KV** (Redundancy-aware KV Cache Compression for reasoning models,
NeurIPS 2025) onto a clean **SGLang v0.5.14** baseline. R-KV is a
**decoding-time** KV-cache compressor: while a reasoning model generates long
chain-of-thought output, it keeps only the *important* and *non-redundant*
tokens, giving large memory savings and throughput gains at near-full accuracy.

**Source of truth for the algorithm** is the R-KV reference repo, file
`rkv/compression/r1_kv.py` (+ `rkv/utils.py`). We port that faithfully rather
than reworking a previous (buggy, slow) integration.

## 2. Branch layout (repo: wanke1997/sglang-compress)

| Branch | Purpose |
| --- | --- |
| `dev-v0.5.14` | Active development. Clean SGLang v0.5.14 baseline + this port. |
| `release/sglang-v0.5.14` | Pristine v0.5.14 reference. **Do not modify.** |
| `dev` / `main` | Old CustomKV/SnapKV implementation on SGLang v0.4.3 (buggy + slow). Kept only for reference. |

`upstream` remote points to official `sgl-project/sglang` for future syncs.

## 3. The R-KV algorithm (what `algo.py` implements)

Given the per-request `key/query/value_states` of shape
`(bsz, heads, seq_len, head_dim)` (GQA: `q_heads` may be a multiple of
`kv_heads`), compression triggers once `kv_cache_len >= budget`:

1. **Importance** — take the last `window_size` "observation" queries, attend
   over all earlier keys, softmax, average across the observation rows, then
   smooth with `max_pool1d(kernel_size)`. → `attn_cache`.
2. **Redundancy** — pairwise key cosine similarity (`cal_similarity`): mask the
   diagonal, threshold, exempt the most-recent similar neighbour per key,
   aggregate the rest into a per-key redundancy distribution.
3. **Joint score** — `score = importance * mix_lambda - redundancy * (1 - mix_lambda)`.
4. **Selection** — keep the top `budget - window_size` past tokens **plus** the
   trailing `window_size` observation tokens (always retained).

Default hyper-parameters: `budget=1024`, `window_size=8`, `kernel_size=7`,
`mix_lambda≈0.1`, `retain_ratio=0.1`, `retain_direction="last"`. Compression is
triggered every `B_buffer=128` newly generated tokens (the staging buffer size).

## 4. Two-layer design

The port is deliberately split so the algorithm can be validated without a GPU
or the full serving stack.

```
┌───────────────────────────────────────────────────────────┐
│  Pure algorithm layer  —  algo.py                          │
│  • R1KV, compute_attention_scores, cal_similarity          │
│  • update_kv(): reference-compatible (returns compacted KV)│
│  • select_indices(): returns ONLY kept token indices       │
│      shape (bsz, kv_heads, budget); None when below budget │
│  • device-agnostic, ZERO sglang deps → runs on GPU in prod,│
│      CPU-testable in isolation                             │
└───────────────────────────────────────────────────────────┘
                         ▲ called by
┌───────────────────────────────────────────────────────────┐
│  Integration layer  (NOT YET WRITTEN — next phase, GPU)    │
│  • per-request R-KV state: query cache, trigger counter,   │
│      dropped-token count                                   │
│  • read K/V back from the paged pool for a request         │
│  • compact: relocate kept slots, rewrite req_to_token,     │
│      free dropped slots                                    │
│  • hook into FlashInfer forward_decode                     │
└───────────────────────────────────────────────────────────┘
```

`select_indices()` exists because the integration layer must **relocate slots**
in a paged pool, not `torch.cat` new tensors — it needs the surviving indices,
not a rebuilt KV tensor.

## 5. SGLang v0.5.14 architecture (verified findings)

File paths are relative to the repo root.

- **KV pool**: `MHATokenToKVPool` in
  `python/sglang/srt/mem_cache/memory_pool.py` (~L1068). Per-layer buffers
  `k_buffer[layer]` / `v_buffer[layer]`, shape `(size + page, head_num, head_dim)`,
  NHD layout.
- **Request → token map**: `ReqToTokenPool` (memory_pool.py ~L231),
  `req_to_token[req_pool_idx, :seq_len]` gives the pool slot indices for a
  request. So per-request K for a layer is
  `k_buffer[layer_id][req_to_token[req_pool_idx, :seq_len]]`.
- **Attention is fully fused** (FlashInfer / FlashAttention). Attention weights
  are **not** exposed, so R-KV importance needs a **separate scoring pass**
  (recompute `q @ kᵀ` over the observation window) — this is exactly what
  `compute_attention_scores` does.
- **No built-in mid-generation token dropping.** Eviction (`evict_from_tree_cache`
  in `mem_cache/common.py` ~L298) only fires when the allocator runs out of
  space. R-KV compaction must be added explicitly.
- **Per-request state**: `Req` in `managers/schedule_batch.py` —
  `req_pool_idx` (~L787), `kv_committed_len` (~L737), `kv_allocated_len` (~L738).
- **ForwardBatch**: `model_executor/forward_batch_info.py` (~L322) exposes
  `req_pool_indices`, `seq_lens`, `seq_lens_cpu`, `out_cache_loc`.

## 6. Chosen injection point

`FlashInferAttnBackend.forward_decode()` in
`python/sglang/srt/layers/attention/flashinfer_backend.py` (~L1086–1120),
**after** `set_kv_buffer()` (new K/V written) and **before**
`decode_wrapper.forward()`. At that point we have `q`, `k`, `v`, `layer`, and
`forward_batch` in hand. Backend for phase 1 is **FlashInfer only**.

## 7. Known challenges (where the old impl went wrong)

1. **Paged pool vs dense concat** — compression means physically relocating kept
   slots, rewriting `req_to_token`, and freeing dropped slots. Getting this
   wrong corrupts the cache; doing it inefficiently is the main source of
   slowness.
2. **Query cache** — decode produces only 1 query/step, but R-KV needs the last
   `window_size` observation queries. The integration layer must cache them
   per request.
3. **Trigger cadence** — compress every `B_buffer` generated tokens per request,
   not every step; needs a per-request counter.
4. **Position / rotary handling** — rotary position info is already baked into
   stored keys, so kept keys keep their embeddings; be careful with any code
   that assumes contiguous positions after tokens are dropped.
5. **O(budget²) similarity** — `cal_similarity` builds a `budget × budget`
   matrix per layer per trigger. Fine for correctness; a target for phase-2
   optimization (chunking / cheaper redundancy estimate).

## 8. Status & roadmap

- **[DONE] Phase 1 · step 1** — pure algorithm layer (`algo.py`) + CPU parity
  tests (`test/srt/mem_cache/test_rkv_algo.py`). Verified bit-for-bit against an
  inline reference (4 configs incl. GQA and below-budget). Run:
  `python3 test/srt/mem_cache/test_rkv_algo.py` (no GPU, no PYTHONPATH needed —
  the test loads `algo.py` by file path to bypass the heavy `sglang/__init__.py`).
- **[NEXT] Phase 1 · step 2** — integration layer on a GPU machine: per-request
  state + query cache, paged-pool read-back, slot compaction + `req_to_token`
  rewrite + slot free, and the `forward_decode` hook. Target: single backend
  (FlashInfer), `batch=1`, correctness first (naive read-back/write-back).
- **[LATER] Phase 2** — performance: batching, avoid redundant read-back,
  optimize the O(budget²) similarity, CUDA-graph compatibility, reduce
  host/device syncs. Then accuracy validation on MATH-500 / AIME-24.
