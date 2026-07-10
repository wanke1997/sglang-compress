# R-KV Benchmark Results — Qwen2.5-Math-7B-Instruct

Dedicated report for the R-KV port on a **strong math model**, where accuracy
differences are signal rather than noise. Companion to [`RESULTS.md`](./RESULTS.md)
(which also covers the weak Qwen2.5-0.5B sanity check).

Two campaigns:
- **§A — GSM8K with decode CUDA graph ON (2026-07-10, current).** n=200, the
  standard [`benchmark/gsm8k/bench_sglang.py`](../../benchmark/gsm8k/bench_sglang.py)
  harness, decode CUDA graph enabled — the production-relevant run.
- **§B — phase-1 eager-path sweep (historical).** The original n=20 correctness
  sweep on the eager decode path (predates CUDA-graph support), kept for
  reference.

---

## A. GSM8K with decode CUDA graph ON (current)

**Setup.** `Qwen2.5-Math-7B-Instruct`, single H100 (`--mem-fraction-static 0.85`),
GSM8K test set, **200 questions**, 5-shot, `max_new_tokens=512`, `temperature=0`,
`--parallel 32`. R-KV `budget=512, window=8, buffer=16`. **Decode CUDA graph ON
for both** (R-KV uses the hybrid eager/graph path); the baseline is the same
flags without `--enable-rkv` (`--disable-radix-cache --disable-overlap-schedule
--page-size 1`).

| Config | Accuracy (200) | Total time | Throughput | Compactions |
| --- | ---: | ---: | ---: | ---: |
| Full-KV baseline | 90.0% | 12.1 s | 1817 tok/s | — |
| **R-KV decode** | **91.5%** | 43.0 s | 510 tok/s | 1263 |

**Findings.**

1. **Accuracy is lossless** — R-KV 91.5% vs full-KV 90.0% (within n=200 judge
   noise; R-KV nominally higher), even though `budget=512 < prompt (~900)` so R-KV
   evicts part of the few-shot prompt itself.
2. **Decode CUDA graph works with R-KV.** 1263 physical compactions, zero
   crashes; startup captured both the prefill and decode graphs and the run
   logged `cuda graph: True`. The hybrid path — the `window_size` steps ending at
   each compaction plus the compaction step run eager, every other decode step
   replays the graph — is correct. **This supersedes the old "R-KV must run
   eager" constraint.**
3. **But throughput costs 3.5× here — the worst case.** GSM8K has **short outputs
   (~110 tok/req)** and **no memory pressure** (sequences fit easily), so R-KV is
   *pure overhead*: it pays the per-`buffer_size` compaction (O(budget²)
   key-similarity + slot relocation) and, with `buffer_size=16` / `window_size=8`,
   forces ~9 of every 16 decode steps eager — losing the graph on the majority of
   steps. In the CUDA-graph regime the **baseline speeds up far more than R-KV**
   (full ~101→1817 tok/s vs R-KV ~74→510 across the eager→graph move), so the
   *relative* gap widens beyond the old eager-vs-eager ~28%.
4. **Two knobs shrink it:** a larger `buffer_size` (rarer compactions → fewer
   forced-eager steps) and cross-layer score subsampling (the O(budget²) scoring
   dominates). And the *benefit* side (memory-bound, long CoT) is not exercised
   here at all — see the §B caveat.

---

## B. Phase-1 eager-path sweep (historical, n=20)

> These numbers predate decode-CUDA-graph support; R-KV and the baseline both ran
> **eager** here. Kept for the correctness sweep and the eager-vs-eager overhead.

## Setup

- **Model**: `Qwen/Qwen2.5-Math-7B-Instruct` (dense, GQA, rotary; `bf16`).
  Downloaded to `/data/model/Qwen2.5-Math-7B-Instruct` via direct CDN `curl -L`
  (HF Xet transfer hangs on large files).
- **GPU**: single NVIDIA H100 80GB, `--mem-fraction-static 0.85`
  (weights ≈14.3 GB, KV pool ≈52 GB → ~981k tokens).
- **Dataset**: GSM8K-style few-shot MATH harness (`test.jsonl` from the `dev`
  branch), first **20 items**, `max_new_tokens=512`, `temperature=0`.
  Few-shot prompt is ~700 tokens.
- **Judging**: numeric match (`eval.py`), identical across all runs.
- **Config**: all R-KV runs use the benchmark's `rkv` mode
  (`--disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph
  --disable-overlap-schedule --page-size 1`); the eager baseline uses the same
  flags without `--enable-rkv`. `window_size=8`, `buffer_size=16` throughout.

> ⚠️ **Small sample (20 items).** Each item is ±5%, so treat these as trend
> indicators. Re-run with `--n 100`+ for tighter figures. The trend is clean.
>
> **Re-run 2026-07-02, post rotary off-by-one fix.** Earlier numbers (budget=512
> 100%, budget=256 90%) were n=20 noise; with the fix accuracy is a flat 95% at
> every budget — i.e. *identical* to the baseline.

## Accuracy vs budget

| Config | Accuracy (20) | Compactions | avg tok | Notes |
| --- | --- | --- | --- | --- |
| baseline (R-KV off) | 95% (19/20) | — | 181 | reference |
| **R-KV budget=512** | **95% (19/20)** | ~230 | 197 | on par with baseline, heavy eviction |
| R-KV budget=256 | 95% (19/20) | 227 | 190 | most aggressive, still on par |

**Takeaway.** Accuracy is **lossless at every budget tested** — R-KV matches the
baseline exactly (95%, 19/20):

- At `budget=512` accuracy is identical to baseline — **even though
  `budget=512 < prompt (~700)`**, so R-KV is evicting part of the few-shot prompt
  itself. The "keep important + recent" selection retains what matters.
- Even at the aggressive `budget=256` there is **no drop** (still 95%); the one
  miss is the same item the baseline also misses.
- The server ran **227–230 physical compactions per sweep with zero crashes**,
  confirming eviction / `free` / `req_to_token` rewrite / rotary decoupling are
  correct under sustained load on a 7B model.

## Speed (fair, same eager config)

| Config | Wall (20) | avg tok | Throughput | Notes |
| --- | --- | --- | --- | --- |
| baseline (eager, fair) | 35.6s | 181 | 101.8 tok/s | serial batch=1, reference |
| R-KV budget=512 | 53.4s | 197 | 73.7 tok/s | serial, ~230 compactions |
| R-KV budget=256 | 49.1s | 190 | 77.3 tok/s | serial, 227 compactions |
| R-KV budget=512, concurrency=8 | 14.3s | 194 | **272.2 tok/s** | 8 in-flight (batch path) |

**R-KV compaction overhead** (same eager path, only `--enable-rkv` differs):
≈102 → 74 tok/s at batch=1, **~28% slower** — consistent with the 0.5B
measurement (~26%). The overhead comes from the per-`buffer_size` compaction
(read back all layers' KV, O(budget²) key-similarity, slot relocation), all
synchronous on the eager path.

**Batching recovers it and then some.** At `concurrency=8`, R-KV budget=512
sustains **272 tok/s** — ~2.7× the eager serial baseline and ~3.7× serial R-KV —
because the batch>=2 per-request path amortizes the compaction work across
in-flight requests. As with 0.5B, the larger *implicit* cost is that R-KV must
run eager and so gives up CUDA graph relative to a production baseline.

## Caveat: benefit side not measured here

This is **short sequences (~1k tokens), batch=1** — the KV cache is not the
bottleneck, so only R-KV's overhead shows, not its benefit (memory saved →
longer context / larger batch under fixed VRAM, and cheaper late-decode
attention on long CoT). A long-sequence / memory-pressure throughput test is
still needed to show the upside.

## Reproduce

**§A — GSM8K with CUDA graph (current):**

```bash
# R-KV decode (CUDA graph ON):
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 PORT=30030 \
  bash R-KV/benchmark/launch_server.sh rkv 512
PYTHONPATH=$PWD/python python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 --num-shots 5 --parallel 32 --host http://127.0.0.1 --port 30030

# Full-KV baseline (same flags, CUDA graph ON):
MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 PORT=30030 \
  bash R-KV/benchmark/launch_server.sh baseline
PYTHONPATH=$PWD/python python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 --num-shots 5 --parallel 32 --host http://127.0.0.1 --port 30030
```

**§B — phase-1 eager sweep (historical):**

```bash
cd R-KV/benchmark
./prepare_data.sh   # or use an existing test.jsonl via --data

MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh baseline
python3 eval.py --n 20 --label base7b

MODEL=/data/model/Qwen2.5-Math-7B-Instruct MEM_FRAC=0.85 ./launch_server.sh rkv 512
python3 eval.py --n 20 --label rkv7b_b512
```
