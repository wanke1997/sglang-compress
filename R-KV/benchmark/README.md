# R-KV Benchmark

Reproducible accuracy + speed benchmark for the R-KV KV-cache compression port
(see [`python/sglang/srt/mem_cache/rkv/`](../../python/sglang/srt/mem_cache/rkv/)).

It compares an SGLang server **with R-KV on vs off** on a GSM8K-style math
harness (few-shot MATH prompts with a `#### N` gold answer), talking to the
server over the `/generate` HTTP API and judging with simple numeric matching.

> **Scope: decode-time R-KV.** This suite benchmarks the *decoding-time* mode
> (`--enable-rkv`) on math reasoning. The **prefill-time** mode
> (`--enable-rkv-prefill`) is benchmarked on a summarisation task — see
> [`../doc/FINDINGS_AND_ROADMAP.md`](../doc/FINDINGS_AND_ROADMAP.md).

## Layout

| File | Purpose |
| --- | --- |
| `launch_server.sh` | Start a server in `rkv <budget>` / `fullkv` / `constrained` mode; set `DP=N` or `TP=N` |
| `prepare_data.sh` | Pull the eval dataset (`test.jsonl`) from the `dev` branch |
| `eval.py` | Drive the server over `/generate`, extract answers, report accuracy + throughput (use `--concurrency N` for server-side batch>1) |
| `RESULTS.md` | Qwen2.5-Math-7B GSM8K `budget`×`buffer_size` sweep (two Full-KV baselines) |
| `RESULTS_dp.md` | Data-parallel (`--dp-size N`) correctness + throughput scaling (up to 8× H100) |
| `RESULTS_tp.md` | Tensor-parallel (`--tp-size N`) scaling + cross-rank correctness |

## Prerequisites

- The dev-v0.5.14 dependency stack is installed (torch 2.11.0+cu129,
  flashinfer 0.6.12, sglang-kernel 0.4.4+cu129, transformers 5.8.1).
- A rotary model downloaded locally, e.g. `/data/model/Qwen2.5-0.5B-Instruct`.
- `sglang` is used from source via `PYTHONPATH` (no install needed); the launch
  script sets it automatically.

## Quick start

```bash
cd R-KV/benchmark

# 1. Fetch the dataset (1319 GSM8K-style items) into ./data/
./prepare_data.sh

# 2a. Terminal A — start a server (pick one):
./launch_server.sh baseline             # R-KV OFF, same flags (fair compare; CUDA graph ON)
./launch_server.sh rkv 512              # R-KV ON,  budget=512
./launch_server.sh baseline-production  # R-KV OFF, full production (radix cache + CUDA graph ON)

# 2b. Terminal B — run the eval (after the server prints "Uvicorn running"):
python3 eval.py --n 100 --label rkv_b512

# Add --concurrency N to send requests in parallel, forcing the server to batch
# (exercises the R-KV batch>=2 per-request path):
python3 eval.py --n 20 --concurrency 8 --label rkv_b512_batch8

# Data parallel: set DP=N to run N R-KV replicas (plain DP, tp=1). Use a high
# --concurrency so requests fan out across all replicas. See RESULTS_dp.md.
DP=4 ./launch_server.sh rkv 512
python3 eval.py --n 128 --concurrency 64 --label rkv_b512_dp4
```

## Important: why the flags differ

R-KV **requires** a specific server configuration, and the benchmark encodes it:

- **`--disable-radix-cache`** — R-KV frees KV slots that the radix/prefix cache
  would still reference; leaving radix on double-counts the pool and crashes the
  server's leak checker at idle. R-KV and prefix caching are fundamentally
  incompatible (prefix reuse assumes KV is immutable; R-KV evicts it).
- **`--page-size 1`** — per-slot free is clean at page_size=1.
- **`--disable-overlap-schedule`** — the benchmark pins overlap OFF so the R-KV
  vs Full-KV timing comparison is a controlled A/B. Decode R-KV **supports
  overlap** now (it de-overlaps only the compaction steps; enabling it adds ~2–7%
  throughput on a single card) — it is left off here only for the equal-flags
  comparison.

**Decode CUDA graph is supported** (hybrid path, and left ON): the `window_size`
steps ending at each compaction — plus the compaction step — run eager, while
every other decode step replays the captured graph (logical rotary positions are
restored at `ForwardBatch` construction so graph-replay steps stay correct). So a
**fair** speed comparison uses the `baseline` mode (same flags, CUDA graph on, no
`--enable-rkv`); `baseline-production` additionally re-enables the radix cache for
a full-production reference.

## Tuning R-KV: `budget` vs `buffer_size`

Decode-time R-KV is controlled by the `--rkv-config` JSON (fields map 1:1 to
`RKVConfig` in [`rkv/integration.py`](../../python/sglang/srt/mem_cache/rkv/integration.py)).
`launch_server.sh` exposes the three knobs you actually tune and leaves the rest
at their reference defaults:

| Field | Set via `launch_server.sh` | Script default | Meaning |
| --- | --- | --- | --- |
| `budget` | positional: `./launch_server.sh rkv <budget>` | 512 | **How much** KV survives each compaction — compression *strength* |
| `buffer_size` | `BUFFER=<n>` env | 16 | **How often** compaction fires — once every `buffer_size` generated tokens |
| `window_size` | `WINDOW=<n>` env | 8 | Trailing observation-window queries used to score token importance |

Other `RKVConfig` fields (`kernel_size=7`, `mix_lambda=0.1`, `retain_ratio=0.1`)
keep the reference defaults; to change them, pass a full `--rkv-config` JSON
yourself instead of using the script.

> ⚠️ The script defaults `BUFFER=16`, but `RKVConfig.buffer_size` defaults to
> **128** — always set `BUFFER` explicitly if you care about throughput.

**Mental model — the two knobs are (almost) orthogonal:**

- **`budget` = compress to *how much*.** The steady KV footprint per request is
  `budget` tokens, so a smaller budget saves more memory but evicts more context
  (accuracy risk). *This is the memory / accuracy axis.*
- **`buffer_size` = compress *how often*.** The number of decode steps between
  compactions. Larger `buffer_size` = rarer compaction = fewer forced-eager steps
  = higher throughput, at the cost of a higher *peak* footprint
  (`budget + buffer_size`). *This is the throughput axis.*

**How compaction fires.** Once a request's KV length reaches `budget`, R-KV counts
decode steps and every `buffer_size` steps it scores the past tokens and frees KV
back down to `budget`. The physical KV length is therefore a sawtooth between
`budget` and `budget + buffer_size` — this is the `phys 528 -> 512` you see in the
server log at `budget=512, buffer_size=16`. The `window_size` steps ending at each
compaction run eager (they collect the scoring queries); constraint:
`buffer_size >= window_size`.

**Choosing values** (Qwen2.5-Math-7B, GSM8K, ~700-token prompt; throughput below is
the current **batched-scoring** path — full grid + pre-optimization comparison in
[`RESULTS.md`](./RESULTS.md)):

- **Throughput ← `buffer_size`** (not budget): at budget 512, `BUFFER=16` ≈ 990 tok/s
  vs `BUFFER=128` ≈ 1510 tok/s — a **1.5× gap** (was 2.5× before batched scoring, which
  cut the per-compaction cost). Larger `buffer_size` is still faster, but small buffers
  are now practical rather than a loss-leader.
- **Accuracy / memory ← `budget`**: `budget=512` is lossless (−41% KV),
  `budget=256` near-lossless (90% vs 92%, −70% KV), `budget=128` collapses. For
  this prompt length `budget=256` is the lossless wall.
- **Sweet spot:** `BUFFER=128 ./launch_server.sh rkv 256` — near-lossless (90%), ~61%
  of baseline throughput, ~70% less KV per request. For the most timely memory reclaim
  (lower peak KV), `BUFFER=16` is now viable too (~40% of baseline).
- **Note:** on this short-output GSM8K workload, `budget ≥ 1024` or `buffer_size ≥ 256`
  barely compacts (peak KV ≈863 < trigger `prompt + buffer`), so those settings fall
  back to full-KV throughput/accuracy; use a long-output workload to exercise them.

```bash
BUFFER=128 ./launch_server.sh rkv 256   # sweet spot: budget=256, buffer_size=128
```

## Judging

`eval.py` extracts the gold answer from `#### N` and the prediction from
`\boxed{...}` / "the final answer is ..." / trailing number, then compares
numerically. This is adequate for GSM8K integer answers; it is **not** a full
latex2sympy grader, so absolute accuracy is approximate — the point is the
**relative** on/off comparison under identical judging.
