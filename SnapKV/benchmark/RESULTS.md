# SnapKV Benchmark Results

Long-context prefill-compression validation of the SnapKV port
(see [`python/sglang/srt/mem_cache/snapkv/`](../../python/sglang/srt/mem_cache/snapkv/)).

SnapKV is a **prompt-phase** compressor: right after a long prompt is prefilled,
it keeps only the `max_capacity_prompt` prompt tokens that the trailing
observation window attends to most and physically frees the rest. The natural
test is therefore the SnapKV notebook demo (`notebooks/example.ipynb`) hardened
into a **needle-in-a-haystack**: a long article (the SnapKV paper source,
`data/snapkv.txt`) is fed as the prefill, a unique passkey is injected at a
chosen depth, and the model is asked for it at the very end (**long input, short
output**). If SnapKV keeps the right tokens, the passkey survives compression
and the model still answers.

## Setup

- **Model**: `Qwen2.5-0.5B-Instruct` (a small, weak model — a conservative
  stress test; a stronger model retrieves more robustly).
- **GPU**: single NVIDIA H100 80GB.
- **Prompt**: full `snapkv.txt` article + injected passkey + question,
  **16 405 tokens** (via `/v1/chat/completions`, so the chat template is
  applied). `max_tokens=48`, `temperature=0`.
- **SnapKV config**: `window_size=32`, `kernel_size=5`, `pooling=avgpool`
  throughout; only `max_capacity_prompt` (the budget) is swept.
- **Flags**: all runs use `--disable-radix-cache --disable-decode-cuda-graph
  --disable-prefill-cuda-graph --disable-overlap-schedule --page-size 1
  --chunked-prefill-size -1`; the baseline uses the same flags without
  `--enable-snapkv`.
- **Judging**: exact match of the injected 5-digit passkey in the answer.

Reproduce:

```bash
cd SnapKV/benchmark
./prepare_data.sh                                  # fetch snapkv.txt
MEM_FRAC=0.6 ./launch_server.sh snapkv 256         # terminal A
python3 eval_needle.py --port 30000 --mode passkey --depth 0.5   # terminal B
```

## Passkey retrieval vs budget (needle depth sweep)

Each cell is "retrieved the exact passkey? (Y/N)". The prompt is 16 405 tokens;
"compression" is `16405 / max_capacity_prompt`. Depth is the fraction through
the article where the passkey is injected.

| Config | Compression | d=0.1 | d=0.3 | d=0.5 | d=0.7 | d=0.9 | Compaction (log) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | --- |
| baseline (SnapKV off) | 1× (no evict) | ✅ | — | ✅ | — | ✅ | — |
| **SnapKV, budget=1024** | **16×** | ✅ | ✅ | ✅ | ✅ | ✅ | `prompt 16405 -> 1024 (freed 15381)` |
| **SnapKV, budget=256** | **64×** | ✅ | ✅ | ✅ | ✅ | ✅ | `prompt 16405 -> 256 (freed 16149)` |
| SnapKV, budget=128 | 128× | ❌ | — | ❌ | — | ❌ | `prompt 16405 -> 128 (freed 16277)` |

**Headline takeaway.** SnapKV compresses a **16 405-token** prompt down to a
**256-token** KV budget (**64×**) and the injected passkey is **still retrieved
at every depth**, matching the uncompressed baseline. Even at 16× (budget=1024)
retrieval is perfect. The server logged a physical compaction per request every
time (`SnapKV compacted req_pool_idx=… prompt 16405 -> N slots`), confirming the
eviction / `free` / `req_to_token` rewrite / logical-rotary path is correct.

**Where it breaks (the compression limit).** At `budget=128` (128×, only
`128 - 32 = 96` scored past tokens kept from 16 405) retrieval degrades — the
0.5B model emits a *partial* passkey (`93`, `9388` vs gold `93810`), i.e. the
5-digit needle no longer reliably survives such aggressive pruning on this weak
model. This is the expected behaviour of a KV compressor: lossless while the
budget is large enough, degrading only once it is pushed far below what the task
needs. A budget of **256 already gives full retrieval** here.

## Notes

- `budget` here is `max_capacity_prompt` — the number of prompt KV entries kept
  per request. `window_size=32` of those are always the most recent prompt
  tokens (the observation window); the other `budget - window_size` are the
  top-scoring earlier tokens.
- SnapKV only compresses prompts **longer than** `max_capacity_prompt`; shorter
  prompts are left untouched (verified: no compaction logged, output unchanged).
- The absolute retrieval numbers depend on the model; the point is the
  **relative** on-vs-off comparison under identical judging, and the large
  compression ratio at which SnapKV stays lossless.
