# SnapKV Benchmark

Reproducible long-context prefill-compression benchmark for the SnapKV port
(see [`python/sglang/srt/mem_cache/snapkv/`](../../python/sglang/srt/mem_cache/snapkv/)).

It runs SnapKV's own compression example — the notebook demo
(`FasterDecoding/SnapKV notebooks/example.ipynb`) hardened into a
**needle-in-a-haystack**: feed a long article as the prefill (**long input**),
ask one question at the very end (**short output**), and check the answer. With
SnapKV on, the prompt KV is physically compressed to `max_capacity_prompt`
tokens right after prefill; a relevant needle should survive because the
question (the observation window) decides which prompt tokens are kept.

## Layout

| File | Purpose |
| --- | --- |
| `launch_server.sh` | Start a server in `snapkv <max_capacity_prompt>` / `snapkv-baseline` mode |
| `prepare_data.sh` | Download the SnapKV notebook article (`snapkv.txt`) into `./data/` |
| `eval_needle.py` | Feed a long article + a question, judge the answer (passkey / notebook modes) |
| `RESULTS.md` | Numbers we measured on Qwen2.5-0.5B-Instruct (H100) |

## Prerequisites

- The dev-v0.5.14 dependency stack is installed (torch 2.11.0+cu129,
  flashinfer 0.6.12, sglang-kernel 0.4.4+cu129, transformers 5.8.1).
- A rotary model downloaded locally, e.g. `/data/model/Qwen2.5-0.5B-Instruct`.
- `sglang` is used from source via `PYTHONPATH` (no install needed); the launch
  script sets it automatically to this repo's `python/`.

## Quick start

```bash
cd SnapKV/benchmark

# 1. Fetch the article (the SnapKV paper source) into ./data/snapkv.txt
./prepare_data.sh

# 2a. Terminal A — start a server (pick one):
./launch_server.sh snapkv 256          # SnapKV ON,  max_capacity_prompt=256
./launch_server.sh snapkv 1024         # SnapKV ON,  max_capacity_prompt=1024
./launch_server.sh snapkv-baseline     # SnapKV OFF, same eager flags (fair compare)

# 2b. Terminal B — run the compression example (after "Uvicorn running"):
#     passkey mode: inject a unique passkey at a depth, ask for it at the end
python3 eval_needle.py --port 30000 --mode passkey --depth 0.5

#     notebook mode: the original demo question against the SnapKV paper
python3 eval_needle.py --port 30000 --mode notebook
```

`eval_needle.py` options: `--mode {passkey,notebook}`, `--depth <0..1>` (where
in the article the passkey goes), `--max-words N` (truncate the article),
`--max-tokens N` (generation length), `--seed N`, `--model <path>`.

## Why the flags differ

SnapKV **requires** a specific server configuration, and the benchmark encodes
it (enforced at startup by `ServerArgs._handle_snapkv_validation`):

- **`--disable-radix-cache`** — SnapKV frees prompt KV slots that the
  radix/prefix cache would still reference.
- **`--disable-decode-cuda-graph`** — after prompt eviction the physical KV
  length shrinks and decode positions are set dynamically, which cannot live
  inside a captured CUDA graph, so decode runs eager.
- **`--chunked-prefill-size -1`** — SnapKV needs the whole prompt (and its
  trailing observation window) in a **single** prefill forward to score it.
- **`--disable-overlap-schedule`, `--page-size 1`** — phase-1 simplifications
  (simple timing; per-slot free is clean at page_size=1).

A **fair** comparison uses `snapkv-baseline` (eager, same flags, no
`--enable-snapkv`).

## Judging

`eval_needle.py` (passkey mode) injects a random 5-digit passkey sentence at the
chosen depth and checks the exact number appears in the answer — a definite
ground truth that directly tests whether the needle token survived compression.
Notebook mode has no strict ground truth (weak model, free-form answer); it is a
qualitative echo of the upstream demo.
