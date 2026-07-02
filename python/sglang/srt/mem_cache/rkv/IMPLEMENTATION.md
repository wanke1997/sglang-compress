# R-KV Integration Layer — Implementation Notes

This document describes **how** R-KV (Redundancy-aware KV Cache Compression) is
implemented on top of a clean SGLang v0.5.14 baseline. For the *why* (research
findings, rejected alternatives, the `sparsity/` framework evaluation), read
[`DESIGN.md`](./DESIGN.md) first. This file is the practical map of the code:
the components, the per-step data flow, the exact wiring points, and the
decisions baked into them.

## 1. Overview

R-KV is a **decoding-time** KV-cache compressor. While a model generates a long
output, R-KV periodically evicts the *unimportant* and *redundant* past tokens,
keeping only a fixed `budget` of KV entries per request — freeing GPU memory
while preserving generation quality.

The port is split into two layers:

| Layer | File | Responsibility |
| --- | --- | --- |
| **Algorithm** | [`algo.py`](./algo.py) | Pure, device-agnostic R-KV scoring & selection. Zero SGLang deps. CPU-testable. |
| **Integration** | [`integration.py`](./integration.py) | Bridges the algorithm to SGLang's paged KV pool, FlashInfer decode path, and scheduler lifecycle. |

Phase-1 scope: **FlashInfer backend, `batch=1`, `page_size=1`, correctness
first.**

## 2. The core tension (and how we resolve it)

The algorithm is **per-head / per-layer**: `R1KV.select_indices` returns a kept
set of shape `(bsz, kv_heads, budget)` — different heads may keep different
tokens. But SGLang's physical layout is **per-token / global**: one
`req_to_token` slot is shared across *all* layers and *all* heads, so dropping a
token drops it everywhere.

We therefore **reduce** the algorithm's scores into a single global per-token
decision:

- **cross-head:** mean of the per-head joint scores.
- **cross-layer:** sum over every layer's scores (each layer contributes its own
  `q·kᵀ` importance + key-similarity redundancy).

The result is one score vector per request; we keep the top `budget - window`
past tokens plus the trailing `window` observation tokens.

## 3. Integration components (`integration.py`)

- **`RKVConfig`** — `budget`, `window_size`, `kernel_size`, `mix_lambda`,
  `buffer_size` (compress every N generated tokens), `min_seq_len`.
- **`RKVRequestState`** — per-request bookkeeping: trigger counter
  (`steps_since_compact`), a **per-layer ring buffer of the last `window_size`
  decode queries** (queries are per-layer, allocated lazily from the first
  query's shape), the transient cross-layer score accumulator, and a
  back-reference to the owning `Req`.
- **`RKVCompressor`** — the coordinator. Holds `R1KV`, the pools/allocator, and
  the per-request states. Public surface:
  - `on_request_begin(req)` / `on_request_end(req)` — lifecycle.
  - `observe_decode_layer(q, k, v, layer, forward_batch)` — called per layer
    during decode: caches the query, and (when a compaction is armed) computes
    and accumulates this layer's per-token score.
  - `maybe_compact(forward_batch)` — called after the full forward pass; for any
    armed request, assembles the global kept set and physically compacts.
  - `override_decode_positions(forward_batch)` — replaces decode positions with
    *logical* positions (see §5).
  - `take_pending_length_updates()` — hands the scheduler the new physical
    lengths to apply to its batch tensors.

## 4. Per-decode-step data flow

```
scheduler.update_running_batch
  └─ _apply_rkv_pre_decode(batch)            # register new reqs; apply previous
        ├─ on_request_begin(req) if new        step's physical-length shrink to
        └─ apply take_pending_length_updates   batch.seq_lens BEFORE +1
  └─ batch.prepare_for_decode()              # seq_lens += 1, alloc out_cache_loc

FlashInferAttnBackend.init_forward_metadata(fb)   # fb = the REAL decode batch
  ├─ forward_batch.rkv_compressor = self.rkv_compressor   # bind onto real batch
  └─ override_decode_positions(fb)                        # logical positions

model.forward → per layer → RadixAttention → FlashInferAttnBackend.forward_decode
  └─ after set_kv_buffer:
       observe_decode_layer(q, k, v, layer, fb)   # cache query; accumulate score
                                                   # arm compaction every buffer_size
                                                   # steps once seq_len >= budget

model_runner.forward (after all layers)
  └─ maybe_compact(fb)                          # for armed reqs: compact
        ├─ assemble kept = top(budget-window) past + trailing window
        └─ _compact_request(...)                # see §6
```

## 5. Key design decision — position / rotary (“scheme A”)

SGLang treats `seq_lens` as **both** the physical KV length **and** the rotary
position source (`positions = clamp_position(seq_lens)`). R-KV breaks that
identity: after eviction the physical KV length shrinks to `budget`, but future
tokens must keep their **original absolute positions** so their rotary stays
consistent with the retained keys (rotary is baked into stored keys).

Resolution:

- **`seq_lens` / `kv_committed_len` / `kv_allocated_len` track the physical KV
  length** (drop to `budget` at compaction, `+1` per step after). This makes
  slot allocation and attention length automatically correct — new tokens append
  at the physical tail, attention reads exactly `budget (+n)` slots.
- **Positions stay logical.** `override_decode_positions` sets
  `positions[i] = len(origin_input_ids) + len(output_ids)` for each R-KV
  request, overriding the physical-length-derived value. For an un-compacted
  request this equals the normal value, so it is safe to always apply.

Scheduler and ModelRunner share the process, so `Req` length fields are updated
in place during compaction; the batch-level `seq_lens` tensor is updated by the
scheduler via `take_pending_length_updates()`.

## 6. Physical compaction (`_compact_request`, `page_size=1`)

Given the surviving logical indices `kept` (ascending, length `budget`) for a
request occupying physical slots `slots = req_to_token[idx, :seq_len]`:

1. `src = slots[kept]` (surviving physical slots), `dst = slots[:budget]`
   (target front slots).
2. For **every layer**: read `k/v_buffer[src]`, `.clone()` (avoids src/dst
   overlap corruption), write to `k/v_buffer[dst]`.
3. `free(slots[budget:seq_len])` — return the tail slots to the allocator
   (`page_size=1` ⇒ per-slot free is clean; the paged allocator would otherwise
   free at page granularity and clobber live neighbours).
4. `req_to_token[idx, budget:seq_len] = 0` (clear the tail); the front `budget`
   entries already point at `dst`.
5. Shrink `req.kv_committed_len / kv_allocated_len = budget`; publish the new
   physical length via `pending_length_updates`.

## 7. Wiring points (5 core files)

| File | Change |
| --- | --- |
| `server_args.py` | `--enable-rkv`, `--rkv-config '{...}'` |
| `model_executor/model_runner.py` | Build `RKVCompressor` in `alloc_memory_pool` (after pools exist); call `maybe_compact` after the decode forward pass |
| `layers/attention/flashinfer_backend.py` | Backend holds `rkv_compressor`; in `init_forward_metadata` bind it onto the real decode batch + `override_decode_positions`; call `observe_decode_layer` in `forward_decode` after `set_kv_buffer` |
| `managers/scheduler.py` | Re-bind `rkv_compressor` in `init_memory_pools` (see §8); `_apply_rkv_pre_decode` before `prepare_for_decode` |
| `managers/scheduler_components/batch_result_processor.py` | Call `on_request_end` at the two real-finished points (beside `hisparse.request_finished`, **not** at retract points) |

## 8. Bugs fixed during bring-up

1. **`observe` never fired — compressor bound too early.** The scheduler
   snapshotted `rkv_compressor` in `__init__`, *before* `alloc_memory_pool`
   constructs it → it was `None`. Fix: re-bind in `init_memory_pools`, after the
   pools (and the compressor) exist.
2. **`observe` still saw `rkv=False` — wrong `forward_batch` object.** Binding
   `forward_batch.rkv_compressor` in `model_runner.forward` was useless because
   `_prepare_eager_forward_batch` rebuilds the batch before the model runs. Fix:
   the FlashInfer backend holds the compressor itself and binds it (and overrides
   positions) inside `init_forward_metadata`, which receives the *actual* decode
   batch; `forward_decode` uses `self.rkv_compressor`.
3. **`on_request_end` and frozen dataclass.** `SchedulerBatchResultProcessor` is
   a frozen dataclass built *before* `init_memory_pools` binds the compressor, so
   a snapshot field would capture `None`. Fix: look it up dynamically via
   `model_worker.model_runner.rkv_compressor` at the finished-request points.

## 9. Environment (dev-v0.5.14 needs a newer stack than v0.5.3-era wheels)

- `transformers==5.8.1`, `flashinfer{,-cubin}==0.6.12`,
  `sglang-kernel==0.4.4+cu129` (from `https://docs.sglang.ai/whl/cu129/` so the
  torch C10 ABI matches), `torch==2.11.0+cu129`.
- HF **Xet** transfer can hang on large files behind some networks; download
  with `HF_HUB_DISABLE_XET=1` or `curl -L` on the resolved CDN URL.

## 10. Running & validation

Launch (phase-1 flags are required — R-KV runs only on the **eager decode**
path, so decode CUDA graph must be off; `page_size=1` for clean slot free;
overlap off for simple timing):

```bash
PYTHONPATH=$PWD/python HF_HUB_DISABLE_XET=1 python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-0.5B-Instruct \
  --attention-backend flashinfer \
  --disable-decode-cuda-graph --disable-prefill-cuda-graph \
  --disable-overlap-schedule --page-size 1 \
  --enable-rkv --rkv-config '{"budget":64,"window_size":8,"buffer_size":16}' \
  --mem-fraction-static 0.6 --host 127.0.0.1 --port 30000
```

Verified end-to-end on Qwen2.5-0.5B-Instruct: with `budget=64, buffer_size=16`,
a long generation triggers compaction repeatedly (e.g. `phys 80 -> 64, frees 16
slots`), output stays coherent (no garbage, no loops), and `on_request_end`
clears per-request state to 0 after each request.

CPU unit tests (no GPU, no installed `sglang` needed — modules are loaded by
path):

```bash
python3 test/srt/mem_cache/test_rkv_integration.py
```

7 cases: kept-set assembly, slot relocation, overlap safety, physical-length
bookkeeping, logical-position decoupling, and request lifecycle
(begin/end/idempotent).

## 11. Known limitations / next

- **`batch=1` fast paths.** `observe_decode_layer` / `maybe_compact` index
  request 0; extend to per-request loops for batched decode.
- **O(budget²) similarity** in `cal_similarity` — a phase-2 perf target.
- **No CUDA-graph decode** yet (dynamic eviction can't live in a captured
  graph). Phase-2 would need a graph-compatible compaction scheme.
- Accuracy validation (MATH-500 / AIME-24) is the next milestone.
