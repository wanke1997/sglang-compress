# R-KV — Decoding-Time KV Cache Compression for SGLang v0.5.14

**R-KV** is a *decoding-time* KV-cache compressor. While a model generates a
long output (e.g. chain-of-thought), R-KV periodically evicts the **unimportant**
and **redundant** past tokens, keeping only a fixed `budget` of KV entries per
request — freeing GPU memory while preserving generation quality.

This directory holds the docs and benchmark for the R-KV port; the code lives in
[`python/sglang/srt/mem_cache/rkv/`](../python/sglang/srt/mem_cache/rkv/).

- **Algorithm** —
  [`python/sglang/srt/mem_cache/rkv/algo.py`](../python/sglang/srt/mem_cache/rkv/algo.py):
  joint scoring of *importance* (attention over a recent observation window) and
  *redundancy* (key cosine-similarity); keep the top `budget` tokens per request.
- **Integration** —
  [`python/sglang/srt/mem_cache/rkv/integration.py`](../python/sglang/srt/mem_cache/rkv/integration.py):
  true physical eviction in SGLang's paged KV pool (relocate surviving slots,
  `free()` the rest, rewrite `req_to_token`) with rotary positions kept
  consistent after the sequence physically shrinks. Runs on the FlashInfer
  decode path.
- **Docs** — design notes in [`doc/`](doc/) (`DESIGN.md`, `IMPLEMENTATION.md`,
  `RETRO_old_vs_new.md`).
- **Benchmark** — the accuracy/speed suite in [`benchmark/`](benchmark/) and the
  measured numbers in [`benchmark/RESULTS.md`](benchmark/RESULTS.md),
  [`benchmark/RESULTS_math7b.md`](benchmark/RESULTS_math7b.md),
  [`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md).

## Headline result — Qwen2.5-Math-7B-Instruct (single NVIDIA H100)

GSM8K (200 questions, 5-shot, `max_new_tokens=512`), **decode CUDA graph ON**,
`budget=512, window=8, buffer=16`:

| Config | Accuracy (200) | Total time | Throughput | Compactions |
| --- | --- | --- | --- | --- |
| Full-KV baseline | 90.0% | 12.1 s | 1817 tok/s | — |
| **R-KV decode** | **91.5%** | 43.0 s | 510 tok/s | 1263 |

Accuracy is **lossless** (91.5% vs 90.0%, within n=200 judge noise) and R-KV runs
**with CUDA graph** (1263 physical compactions, zero crashes) — confirming the
hybrid eager/graph decode path is correct. The **3.5× throughput cost here is the
worst case**: GSM8K has short outputs and no memory pressure, so R-KV is pure
overhead with no memory payoff to recoup. R-KV wins on throughput only when the
server is memory-bound with long decodes. Full analysis (and the earlier
eager-path numbers): [`benchmark/RESULTS_math7b.md`](benchmark/RESULTS_math7b.md).

---

## How R-KV works (and how it differs from SnapKV)

Both keep a fixed KV budget per request and physically free the rest, but fire
at different points in a request's life:

| | **R-KV** (this dir) | **SnapKV** ([`../SnapKV`](../SnapKV/)) |
| --- | --- | --- |
| When | **repeatedly**, during decode | **once**, at the end of prefill |
| Target | the long generated **output** (CoT) | the long **prompt** |
| Score | importance (attention) − redundancy (key similarity) | observation-window attention + pooling |
| Best for | long reasoning (short input, long output) | long-context QA / summarisation (long input, short output) |

## Usage

R-KV runs from source; no install needed (`launch_server.sh` sets `PYTHONPATH`
to this repo's `python/`).

```bash
cd R-KV/benchmark
./prepare_data.sh                                       # fetch the eval set

# R-KV on, keep 512 KV entries per request:
MODEL=/data/model/Qwen2.5-Math-7B-Instruct ./launch_server.sh rkv 512

# then, in another shell:
python3 eval.py --n 20 --label rkv_b512
```

To enable R-KV on your own launch command, add `--enable-rkv` plus the required
flags (see **Constraints** below):

```bash
python3 -m sglang.launch_server \
  --model-path /data/model/Qwen2.5-Math-7B-Instruct \
  --attention-backend flashinfer \
  --enable-rkv \
  --rkv-config '{"budget": 512, "window_size": 8, "buffer_size": 16, "mix_lambda": 0.1}' \
  --disable-radix-cache --disable-overlap-schedule --page-size 1
```

## Parameters

Per-field flags set the base config; `--rkv-config` (JSON) overrides any of them
and takes priority.

| Flag | Type / default | Meaning |
| --- | --- | --- |
| `--enable-rkv` | bool, `False` | Turn R-KV on. |
| `--rkv-budget` | int, `1024` | **The budget.** Number of KV entries kept per request after each compression. |
| `--rkv-window-size` | int, `8` | Trailing **observation window** (most recent tokens), always retained; its queries score the past tokens. |
| `--rkv-kernel-size` | int, `7` | Pooling kernel size used to smooth the importance (attention) scores over neighbouring tokens. |
| `--rkv-mix-lambda` | float, `0.1` | Mixing weight for the joint score `importance·λ − redundancy·(1−λ)`. `1` = importance only, `0` = redundancy only. |
| `--rkv-retain-ratio` | float, `0.1` | Fraction of most-recent similar neighbours exempted from the redundancy penalty. |
| `--rkv-retain-direction` | `last` \| `first` \| `last_percent` \| `first_percent`, `last` | Which end of the sequence the retain ratio protects. |
| `--rkv-buffer-size` | int, `128` | Compress once every this many **newly generated** tokens per request (the trigger cadence `B_buffer`). |
| `--rkv-min-seq-len` | int, `= budget` | Minimum KV length before compression is ever considered. |
| `--rkv-config` | JSON string, `None` | Overrides the per-field flags, e.g. `'{"budget": 512, "window_size": 8, "buffer_size": 16}'`. Alias `--rkv-extra-config`. |

## Constraints (required flags)

`--enable-rkv` is rejected at startup unless these hold, because R-KV physically
frees KV slots mid-generation:

- `--disable-radix-cache` — R-KV frees KV slots the prefix cache would still
  reference.
- `--disable-overlap-schedule`, `--page-size 1` — phase-1 simplifications.
- `--tp-size 1` — tensor parallelism is not yet supported (plain data
  parallelism `--dp-size N --tp-size 1` **is** supported; see
  [`benchmark/RESULTS_dp.md`](benchmark/RESULTS_dp.md)).

**Decode CUDA graph is supported** (and recommended): a per-step hook forces the
`window_size` steps ending at each compaction — plus the compaction step — to run
eager, while every other decode step replays the captured graph. Logical rotary
positions are restored at `ForwardBatch` construction so graph-replay steps stay
correct. (Prefill CUDA graph is fine too; R-KV only acts during decode.)

A **fair** speed comparison uses the `baseline` mode (same flags, CUDA graph on,
no `--enable-rkv`); `baseline-production` additionally re-enables the radix cache
for a full-production reference.
