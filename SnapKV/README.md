# SnapKV — Prompt-Phase KV Cache Compression for SGLang v0.5.14

**SnapKV** is a *prompt-phase* (prefill-time) KV-cache compressor. Right after a
long prompt is prefilled, SnapKV looks at the attention that the last
`window_size` "observation" query tokens pay to the earlier prompt tokens,
clusters it with a small pooling kernel, and keeps only the
`max_capacity_prompt` most-attended prompt tokens — **physically freeing the
rest of the prompt KV**. Every subsequent decode step then runs against this
shrunken prompt, so a long prompt costs a fixed, small amount of KV memory.

This directory ports SnapKV
([Li et al., 2024](https://arxiv.org/abs/2404.14469);
[FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)) onto this
SGLang v0.5.14 tree, alongside the decoding-time [R-KV](../R-KV/) compressor.

- **Algorithm** —
  [`python/sglang/srt/mem_cache/snapkv/algo.py`](../python/sglang/srt/mem_cache/snapkv/algo.py):
  a faithful, device-agnostic port of the reference `SnapKVCluster`
  (observation-window attention → pooling → top-k selection), bit-for-bit
  checked against the original.
- **Integration** —
  [`python/sglang/srt/mem_cache/snapkv/integration.py`](../python/sglang/srt/mem_cache/snapkv/integration.py):
  scores the prompt during the FlashInfer prefill forward, then performs true
  physical eviction in SGLang's paged KV pool (relocate surviving slots, `free()`
  the rest, rewrite `req_to_token`) and keeps rotary positions consistent after
  the prompt physically shrinks.
- **Docs** — design notes in [`doc/`](doc/) (`DESIGN.md`).
- **Benchmark** — the long-input/short-output compression example in
  [`benchmark/`](benchmark/) (`launch_server.sh`, `eval_needle.py`) and the
  measured numbers in [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

## Headline result — Qwen2.5-0.5B-Instruct (single NVIDIA H100)

Needle-in-a-haystack: a **16 405-token** article is fed as the prefill, a unique
passkey is injected at a chosen depth, and the model is asked for it at the end.

| Config | Compression | Passkey retrieved (depths 0.1–0.9) |
| --- | --- | --- |
| baseline (SnapKV off) | 1× | ✅ all |
| **SnapKV, `max_capacity_prompt=256`** | **64×** | **✅ all** |
| SnapKV, `max_capacity_prompt=1024` | 16× | ✅ all |

SnapKV shrinks the prompt KV **64×** and still retrieves the needle at every
depth, matching the uncompressed baseline. Full report and the compression limit
(`budget=128` starts to degrade) in [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

---

## How SnapKV works (and how it differs from R-KV)

Both compressors keep a fixed KV budget per request and physically free the
rest, but they fire at different points in a request's life:

| | **SnapKV** (this dir) | **R-KV** ([`../R-KV`](../R-KV/)) |
| --- | --- | --- |
| When | **once**, at the end of prefill | **repeatedly**, during decode |
| Target | the long **prompt** | the long generated **output** (CoT) |
| Score | observation-window attention + pooling | importance (attention) − redundancy (key similarity) |
| Best for | long-context QA / summarisation (long input, short output) | long reasoning (short input, long output) |

The SGLang plumbing is shared: both reduce the algorithm's per-head/per-layer
scores to a single global per-token decision (one KV slot per token is shared
across all layers), evict at `page_size=1`, and decouple the *physical* KV
length (which shrinks) from the *logical* rotary position (which does not).

## Usage

SnapKV runs from source; no install needed (`launch_server.sh` sets
`PYTHONPATH` to this repo's `python/`).

```bash
cd SnapKV/benchmark
./prepare_data.sh                                # fetch the demo article

# SnapKV on, keep 1024 prompt tokens per request:
MODEL=/data/model/Qwen2.5-0.5B-Instruct ./launch_server.sh snapkv 1024

# then, in another shell:
python3 eval_needle.py --port 30000 --mode passkey --depth 0.5
```

To enable SnapKV on your own launch command, add `--enable-snapkv` plus the
required flags (see **Constraints** below):

```bash
python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-0.5B-Instruct \
  --attention-backend flashinfer \
  --enable-snapkv \
  --snapkv-config '{"max_capacity_prompt": 1024, "window_size": 32, "kernel_size": 5, "pooling": "avgpool"}' \
  --disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph \
  --disable-overlap-schedule --page-size 1 --chunked-prefill-size -1
```

## Parameters

All parameters are exposed as `ServerArgs` flags. Per-field flags set the base
config; `--snapkv-config` (JSON) overrides any of them and takes priority.

| Flag | Type / default | Meaning |
| --- | --- | --- |
| `--enable-snapkv` | bool, `False` | Turn SnapKV on. |
| `--snapkv-max-capacity-prompt` | int, `1024` | **The budget.** Number of prompt KV entries kept per request after compression. A prompt with `seq_len > max_capacity_prompt` is compressed to exactly this many tokens; shorter prompts are left untouched. Smaller = more memory saved, more aggressive. |
| `--snapkv-window-size` | int, `32` | Size of the trailing **observation window**: the last `window_size` prompt tokens, always kept, whose queries score all earlier tokens. Must be `< max_capacity_prompt`. The other `max_capacity_prompt - window_size` kept tokens are the top-scoring earlier prompt tokens. |
| `--snapkv-kernel-size` | int, `5` | Width of the 1-D pooling applied to the attention scores before top-k. This "clusters" attention mass over neighbouring tokens so SnapKV keeps informative *spans*, not isolated tokens. |
| `--snapkv-pooling` | `avgpool` \| `maxpool`, `avgpool` | Pooling used for that clustering step. |
| `--snapkv-config` | JSON string, `None` | Overrides the per-field flags, e.g. `'{"max_capacity_prompt": 512, "window_size": 32}'`. Also accepts alias `--snapkv-extra-config`. |

**Rule of thumb.** Set `max_capacity_prompt` to the KV budget you can afford per
request; keep `window_size` at 16–64 (large enough that the trailing question /
instruction is fully in the window). `kernel_size=5`, `pooling=avgpool` are the
reference defaults and rarely need changing.

## Constraints (required flags)

`--enable-snapkv` is rejected at startup (`ServerArgs._handle_snapkv_validation`)
unless the following hold, because SnapKV physically frees prompt KV slots:

- `--disable-radix-cache` — SnapKV frees KV slots the prefix cache would still
  reference.
- `--disable-decode-cuda-graph` (or `--disable-cuda-graph`) — after eviction the
  physical KV length and decode positions are dynamic, so decode runs eager.
- `--disable-overlap-schedule` — phase-1 timing simplification.
- `--page-size 1` — per-slot free.
- `--chunked-prefill-size -1` — SnapKV needs the whole prompt (and its trailing
  observation window) in a single prefill forward to score it.
- `--tp-size 1` — tensor parallelism is not yet supported.
- Cannot be combined with `--enable-rkv` (both own the physical-length / rotary
  bookkeeping).

## Tests

GPU-free CPU unit tests live at
[`test/srt/mem_cache/test_snapkv_algo.py`](../test/srt/mem_cache/test_snapkv_algo.py)
and
[`test/srt/mem_cache/test_snapkv_integration.py`](../test/srt/mem_cache/test_snapkv_integration.py):
the algorithm is bit-for-bit checked against the reference `SnapKVCluster`, and
the integration's paged-pool compaction, lifecycle, and decode-position override
are covered with mock pools. Run:

```bash
python3 test/srt/mem_cache/test_snapkv_algo.py
python3 test/srt/mem_cache/test_snapkv_integration.py
```
