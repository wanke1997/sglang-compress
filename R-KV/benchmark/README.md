# R-KV Benchmark

Reproducible accuracy + speed benchmark for the R-KV KV-cache compression port
(see [`python/sglang/srt/mem_cache/rkv/`](../../python/sglang/srt/mem_cache/rkv/)).

It compares an SGLang server **with R-KV on vs off** on a GSM8K-style math
harness (few-shot MATH prompts with a `#### N` gold answer), talking to the
server over the `/generate` HTTP API and judging with simple numeric matching.

## Layout

| File | Purpose |
| --- | --- |
| `launch_server.sh` | Start a server in `baseline` / `rkv <budget>` / `baseline-cudagraph` mode; set `DP=N` for N-way data parallelism |
| `prepare_data.sh` | Pull the eval dataset (`test.jsonl`) from the `dev` branch |
| `eval.py` | Drive the server over `/generate`, extract answers, report accuracy + throughput (use `--concurrency N` for server-side batch>1) |
| `RESULTS.md` | Numbers we measured on Qwen2.5-0.5B-Instruct (H100) |
| `RESULTS_math7b.md` | Numbers we measured on Qwen2.5-Math-7B-Instruct (H100) |
| `RESULTS_dp.md` | Data-parallel (`--dp-size N`) correctness + throughput scaling (up to 8× H100) |

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
./launch_server.sh baseline            # R-KV OFF, eager config (fair compare)
./launch_server.sh rkv 512             # R-KV ON,  budget=512
./launch_server.sh baseline-cudagraph  # R-KV OFF, CUDA graph on (production-like)

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
- **`--disable-decode-cuda-graph`** — dynamic eviction cannot live inside a
  captured CUDA graph, so R-KV runs on the eager decode path.
- **`--disable-overlap-schedule`, `--page-size 1`** — phase-1 simplifications
  (simple timing; per-slot free is clean at page_size=1).

Because R-KV must run eager, a **fair** speed comparison uses the `baseline`
(eager, same flags, no `--enable-rkv`) mode, not `baseline-cudagraph`.

## Judging

`eval.py` extracts the gold answer from `#### N` and the prediction from
`\boxed{...}` / "the final answer is ..." / trailing number, then compares
numerically. This is adequate for GSM8K integer answers; it is **not** a full
latex2sympy grader, so absolute accuracy is approximate — the point is the
**relative** on/off comparison under identical judging.
