# R-KV on SGLang — Findings & Optimization Roadmap

Hand-off reference (human or AI agent) for continuing the prompt-phase KV
compression work. Captures the **measured findings** and the **future
optimization directions**, ROI-ordered. Read `DESIGN.md` and
`IMPLEMENTATION.md` first for the architecture; this doc is the "where we are
and what's worth doing next" layer.

Branch: `main`.
Model for quality: Qwen3-30B-A3B-Instruct-2507 (dp=8). Model for fast
perf/repro: Qwen2.5-0.5B-Instruct. Data: 136 summarization records
(`benchmark/data/stage1_summarization.jsonl`, which also carries the gold as a
trailing assistant turn; the runner strips it so it never leaks into the prompt).

---

## 1. What exists today (committed)

Two R-KV compressors: R-KV-prefill compresses the prompt at the end of prefill
(then decode normally), while decode R-KV compresses mid-generation:

| Compressor | File | Compaction timing | Scoring |
|---|---|---|---|
| **R-KV-prefill** (oneshot=A, buffered=B) | `rkv/prefill_integration.py` | prefill end | importance + O(n²) redundancy |
| **decode R-KV** | `rkv/integration.py` | every `buffer_size` decode steps | importance + redundancy |

Key commits (this line of work):
- `3253dbe3c` R-KV as prefill compressor (oneshot + buffered)
- `b8fe8b297` decode R-KV hybrid CUDA graph (non-compaction steps on graph)
- `5b43ea81b` decode R-KV compression-aware admission
- `17c05bee5` decode R-KV retract state hook
- `a7620c8d4` overlap enablement + **retract KV-leak fix** (`prompt_len = req.seqlen`) + retract hooks for R-KV-prefill
- `43a41c131` **compression-aware admission** for prompt-phase (R-KV-prefill)
- `fe1b26d29` **re-gate overlap off** for prompt-phase (allocator race fix)

Constraints (both R-KV modes): `--disable-radix-cache`,
`--disable-overlap-schedule`, `--page-size 1`, `tp==1` (R-KV-prefill also needs
`--disable-prefill-cuda-graph`). Decode CUDA graph **is** supported and stays on.

---

## 2. Measured findings

### 2.1 Quality — lossless at budget 4096
Full 136 summarization, GPT-5.2 judge vs gold, no leak:
- **Full KV: 75/136 = 55.1%**, **R-KV oneshot budget 4096: 75/136 = 55.1%** — identical.
- Budget scan (earlier, same judge): R-KV(A) best near budget 4096
  (near-lossless); R-KV(B, buffered) best at mid budget (1024–2048). n=136
  single-pass judge is noisy (~±2 rows).

### 2.2 Compression-aware admission — the throughput unlock
Memory-bound regime (64K pool, ~5k prompts, strict mem-check):
- Peak concurrent running-req: **Full KV 10 → R-KV-prefill 65 (6.5×)**.
- Throughput (conc 64, 1024 out): R-KV-prefill ≈ even at 5k prompts (the
  concurrency headroom is offset by R-KV's O(n²) prefill scoring at this budget).
- Zero leak / OOM / crash across all runs. Non-compressed path unaffected.

### 2.3 The prompt-length trend — R-KV wins more as prompts grow
Equal-work (`--ignore-eos`, both generate exactly 1024 tokens):

| prompt | Full KV conc | throughput |
|---|---|---|
| ~5k | 10 | R-KV 3200 vs Full 3166 (**even**) |
| ~20k | **3** | R-KV 2185 vs Full 1480 (**1.48×**) |

Mechanism: Full KV footprint scales with prompt length (P + output), so its
concurrency collapses as prompts grow; R-KV's decode footprint is constant
(budget + output). The longer the prompt, the bigger R-KV's concurrency edge.

### 2.4 When R-KV does NOT win (large-pool regime)
Full 136 at budget 4096 on 30B with a **large** KV pool (not memory-bound),
natural EOS: Full 1998 tok/s / 4.27s avg-lat vs R-KV 1558 tok/s / 5.45s. R-KV
loses because there is no memory pressure to exploit and it still pays the
O(n²) prefill scoring. **R-KV wins on throughput only when memory-bound.**

> Methodology note: with natural EOS the two configs generate different output
> lengths (compression changes the model's text), so `gen_throughput` is
> confounded. Use `--ignore-eos` for clean equal-work throughput A/B.

### 2.5 Overlap — small benefit, re-gated off
- Measured (small model, prefill-dominant): overlap = **+16.6%** non-compressed,
  **+7.5%** R-KV-prefill. Benefit shrinks with scale (GPU forward dwarfs the CPU
  scheduling overlap hides) → estimated **~2–5% at 30B** on this prefill-bound
  workload.
- **Crash at scale** (30B, dp=8, budget 4096, ~23k prompts): the compressor's
  `maybe_compact` frees KV *inside* the forward (on `forward_stream`);
  `kv_allocator.free()` does `free_pages = cat(free_pages, freed)` on that
  stream, while overlap's next-batch `alloc()` reads `free_pages` on the default
  stream without waiting → torn read → garbage slot indices → CUDA illegal
  memory access. Same reason decode R-KV requires overlap off. Re-gated
  (`fe1b26d29`); fails fast at startup now.

### 2.6 CUDA graph
Decode graph works for both R-KV modes (logical rotary positions restored at
ForwardBatch construction; decode R-KV uses a hybrid eager/graph). Prefill
graph must stay off (prompt-phase scoring/compaction are dynamic shapes) and it
gives ~0% benefit on long prompts anyway (compute-bound).

---

## 3. Optimization roadmap (ROI-ordered)

### P1 — Optimize R-KV-prefill's O(n²) prefill scoring  ← highest value

> **Update (2026-07-11): partly done.** Per-layer scoring is now **batched across
> layers** (one `batched_past_score` call instead of `num_layers` GEMMs) for both
> prefill and decode — 8× on a 2174-token prompt (94.4 ms → 11.9 ms). The items
> below are the *remaining* work on top of that batched baseline.

This is still a bottleneck: it is why R-KV-prefill only ties Full KV at 5k
(instead of winning), and why its per-request latency was higher (11.2s vs 9.4s
at 20k — measured pre-batching; likely narrower now). Killing the rest lets R-KV's
*quality* edge (redundancy term) also translate into throughput/latency wins.
- **P1a. Cross-layer subsampling** — on top of the batched pass, score only K of N
  layers (a further compute ÷ N/K). Easy, high impact. Validate accuracy A/B
  (subsampled vs all-layers keep-set Jaccard + judge accuracy) before/after.
- **P1b. Redundancy-term approximation** — the `k_norm @ k_norm.T` is the O(n²)
  cost. Try local-window / blocked similarity or clustering instead of full
  pairwise. Bounds compute to O(n·w).
- **P1c. fp8 / lower-precision scoring** — the scores only need to rank tokens.
- **P1d. Fused scoring+relocate kernel** (sgl-kernel) — hard, highest ceiling.

Start with P1a (cheapest, biggest immediate win), measure quality + throughput.

### P2 — Systematic benchmark + TECH_REPORT
Consolidate the wins into reproducible artifacts so results are defensible:
- Fold in the 20k prompt-length sweep, the admission concurrency/throughput
  table, the lossless-at-4096 quality result, and the overlap crash finding.
- **Commit the benchmark-side helpers**: `run_bench.py --ignore-eos` and the
  per-request `latency_s` field are currently uncommitted in the `kwa-microsoft`
  benchmark repo — commit them so the A/B is reproducible.
- Add a prompt-length × budget × (Full / R-KV-A / R-KV-B) table with
  throughput + quality.

### P3 — True mid-prefill physical KV release  (unlocks input > KV pool)
R-KV-prefill compresses at prefill **end**, so the *full* prompt KV is
resident during prefill → max input length is capped by the KV pool (~50k
practical). Buffered mode already does a *logical* mid-prefill compaction but
not physical release. True mid-prefill physical release would allow prompts
longer than the pool, but requires decoupling chunked-prefill's
`len(prefix_indices) == logical processed len` invariant (touches core prefill).
High value for very-long-context; hard.

### P4 — Async-safe overlap (re-enable overlap for prompt-phase)  ← low priority
Proper fix for §2.5: defer the compaction `free()` out of the forward to the
scheduler's synced default-stream point (like `cache_finished_req`) — plumb the
freed slots from the compressor to the scheduler. Then remove the overlap guard.
Low priority: benefit is only ~2–5% at scale on the prefill-bound workload, and
it needs expensive large-scale race re-validation.

### P5 — Admission refinements
- Chunked-prefill reservation is currently conservative (only the non-chunked
  commit path reserves `min(P, budget)`; chunked path still reserves full P per
  chunk). Make chunked prefill compression-aware too.
- Consider an adaptive `new_token_ratio` for compressed requests.

### P6 — Decode R-KV throughput (different use case)
Cross-layer score **batching is done** (+80% throughput at `buffer_size=16`).
Remaining: adaptive compaction frequency, cross-layer score *subsampling*,
relocate on a separate CUDA stream, and letting the forced-eager window steps
replay the graph. These target the *decode* R-KV (short prompt / long CoT
output), not the summarization workload.

### P7 — Quality robustness
n=136 single judge pass is noisy (~±2 rows). Multi-seed judging or a larger
sample for stronger quality claims (esp. A-vs-B, which flips within noise at
some budgets).

---

## 4. Known limitations / gotchas
- Overlap OFF is **required** for both R-KV modes (allocator race — §2.5).
- `--ignore-eos` is required for clean throughput A/B (EOS confound — §2.4).
- Repro of the overlap race needs the real scale (30B, dp=8, async);
  `CUDA_LAUNCH_BLOCKING=1` and small models **hide** it (serialization).
- Launch servers detached (`setsid ... </dev/null & disown`) so they survive the
  agent terminal cleanup.
- `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` defaults True → any idle KV leak
  raises `ValueError` and crashes the scheduler (use this to prove no-leak).
