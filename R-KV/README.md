# R-KV — Importance + Redundancy KV Cache Compression (SGLang v0.5.14)

**R-KV** scores every past KV token by **importance** (attention over a recent
observation window) and **redundancy** (key cosine-similarity), keeps only a
fixed `budget` of tokens per request, and **physically frees** the rest from
SGLang's paged KV pool. It ships in **two modes**:

- **Decode-time** (`--enable-rkv`) — the original R-KV. While a model generates a
  long output (e.g. chain-of-thought), R-KV re-scores and evicts every
  `buffer_size` steps, holding the decode KV at a constant `budget`. Best for
  **long reasoning**: short prompt, long output.
- **Prefill-time** (`--enable-rkv-prefill`) — compress the **prompt** once at the
  end of prefill (`oneshot`) or in bounded chunks during prefill (`buffered`),
  then decode against the smaller cache, using R-KV's importance+redundancy
  scoring. Best for **long context**: long prompt, short output (e.g.
  summarisation).

This directory's docs and benchmark below focus on the **decode-time** mode; the
**prefill-time** mode's design, results and roadmap live in
[`doc/FINDINGS_AND_ROADMAP.md`](doc/FINDINGS_AND_ROADMAP.md). Code for both is in
[`python/sglang/srt/mem_cache/rkv/`](../python/sglang/srt/mem_cache/rkv/):

- **Decode-time algorithm** —
  [`algo.py`](../python/sglang/srt/mem_cache/rkv/algo.py): joint scoring of
  *importance* (attention over a recent observation window) and *redundancy* (key
  cosine-similarity); keep the top `budget` tokens per request.
- **Decode-time integration** —
  [`integration.py`](../python/sglang/srt/mem_cache/rkv/integration.py): true
  physical eviction in SGLang's paged KV pool (relocate surviving slots, `free()`
  the rest, rewrite `req_to_token`) with rotary positions kept consistent after
  the sequence physically shrinks. Runs on the FlashInfer decode path.
- **Prefill-time algorithm + integration** —
  [`prefill.py`](../python/sglang/srt/mem_cache/rkv/prefill.py) /
  [`prefill_integration.py`](../python/sglang/srt/mem_cache/rkv/prefill_integration.py):
  the same joint score applied to the **prompt** at prefill end — `oneshot`
  (route A, the accuracy oracle) or `buffered` (route B, bounds the O(n²)
  similarity). A/B diff-test: [`benchmark/rkv_prefill_ab.py`](benchmark/rkv_prefill_ab.py).
- **Docs** — decode design notes in [`doc/`](doc/) (`DESIGN.md`,
  `IMPLEMENTATION.md`, `RETRO_old_vs_new.md`); **prefill** design, results &
  roadmap in [`doc/FINDINGS_AND_ROADMAP.md`](doc/FINDINGS_AND_ROADMAP.md).
- **Benchmark** — decode accuracy/speed suite in [`benchmark/`](benchmark/)
  ([`RESULTS.md`](benchmark/RESULTS.md),
  [`RESULTS_dp.md`](benchmark/RESULTS_dp.md),
  [`RESULTS_tp.md`](benchmark/RESULTS_tp.md)); the prefill mode is benchmarked on
  a summarisation task (see
  [`doc/FINDINGS_AND_ROADMAP.md`](doc/FINDINGS_AND_ROADMAP.md)).

## Headline result — decode mode, Qwen2.5-Math-7B-Instruct (single NVIDIA H100)

GSM8K (200 questions, `max_new_tokens=512`, `--concurrency 32`), **decode CUDA
graph ON**, **batched scoring**, `budget=512, window=8, buffer=16`:

| Config | Accuracy (200) | Throughput | Compactions |
| --- | --- | --- | --- |
| Full-KV baseline | 92.0% | 2363 tok/s | — |
| **R-KV decode (batched)** | **90.5%** | 987 tok/s | 1980 |

Accuracy is **lossless** (90.5% vs 92.0%, within n=200 judge noise) and R-KV runs
**with CUDA graph** (1980 physical compactions, zero crashes) — the hybrid
eager/graph decode path is correct. Throughput here is a **~2.4× cost**, down from
~4.4× before **batched scoring** (which fused the per-layer scoring GEMMs for
**+80%**, 546→987 tok/s): GSM8K has short outputs and no memory pressure, so R-KV is
pure overhead with no memory payoff to recoup. R-KV wins on throughput only when the
server is memory-bound with long decodes. Full sweep + before/after:
[`benchmark/RESULTS.md`](benchmark/RESULTS.md).

---

## R-KV modes: decode vs prefill

Both keep a fixed KV budget per request and physically free the rest; they
differ in **when** they fire and **what** they compress:

| | **R-KV decode** (`--enable-rkv`) | **R-KV prefill** (`--enable-rkv-prefill`) |
| --- | --- | --- |
| When | repeatedly, during decode | once at prefill end (or chunked) |
| Target | the generated **output** (CoT) | the **prompt** |
| Score | importance − redundancy | importance − redundancy |
| Best for | long reasoning (short in, long out) | long context (long in, short out) |

See [`doc/FINDINGS_AND_ROADMAP.md`](doc/FINDINGS_AND_ROADMAP.md) for the
prefill-mode design, results and roadmap.

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
