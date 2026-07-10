#!/usr/bin/env bash
# Launch an SGLang server for the R-KV benchmark.
#
# Usage:
#   ./launch_server.sh baseline               # R-KV OFF, same flags as R-KV (fair compare; CUDA graph ON)
#   ./launch_server.sh rkv 512                 # R-KV ON,  budget=512 (256/512/1024 ...)
#   ./launch_server.sh baseline-production     # R-KV OFF, full production (radix cache + CUDA graph ON)
#
# Data parallel (optional): set DP=N to run N R-KV replicas (plain DP, tp=1).
# Each replica keeps its own KV pool and runs R-KV independently; a router
# load-balances requests. Example:
#   DP=4 ./launch_server.sh rkv 512            # 4-way data parallel, R-KV on
#
# Env overrides: MODEL, PORT, WINDOW, BUFFER, MEM_FRAC, DP
set -euo pipefail

MODE="${1:-rkv}"
BUDGET="${2:-512}"
MODEL="${MODEL:-/data/model/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
WINDOW="${WINDOW:-8}"
BUFFER="${BUFFER:-16}"
MEM_FRAC="${MEM_FRAC:-0.6}"
DP="${DP:-1}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO/python"
export HF_HUB_DISABLE_XET=1  # HF Xet transfer can hang on large files

# R-KV requires: radix/prefix cache OFF (R-KV frees slots the radix tree still
# references -> pool double-count crash), overlap OFF (phase-1 simplification),
# page_size=1 (clean per-slot free). Decode CUDA graph IS supported and left ON:
# a per-step hook forces the `window_size` steps ending at each compaction (plus
# the compaction step) to run eager, while every other decode step replays the
# captured graph (logical rotary positions are restored at ForwardBatch
# construction so graph-replay steps see them too).
RKV_FLAGS=(--disable-overlap-schedule --disable-radix-cache --page-size 1)

COMMON=(--model-path "$MODEL" --attention-backend flashinfer
        --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$PORT")

# Optional plain data parallelism: N independent R-KV replicas (tp=1). R-KV does
# not support tp>1, but plain DP is fine -- each rank runs its own compressor
# over a disjoint set of requests (validated; see RESULTS_dp.md).
DP_FLAGS=()
if [[ "$DP" -gt 1 ]]; then
  DP_FLAGS=(--dp-size "$DP" --tp-size 1)
fi

case "$MODE" in
  rkv)
    echo ">> R-KV ON  | budget=$BUDGET window=$WINDOW buffer=$BUFFER dp=$DP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${RKV_FLAGS[@]}" "${DP_FLAGS[@]}" \
      --enable-rkv \
      --rkv-config "{\"budget\":$BUDGET,\"window_size\":$WINDOW,\"buffer_size\":$BUFFER}"
    ;;
  baseline)
    echo ">> BASELINE (same flags as R-KV, CUDA graph ON, no --enable-rkv) dp=$DP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${RKV_FLAGS[@]}" "${DP_FLAGS[@]}"
    ;;
  baseline-production)
    echo ">> BASELINE (full production: radix cache + CUDA graph ON)"
    exec python3 -m sglang.launch_server "${COMMON[@]}"
    ;;
  *)
    echo "unknown mode: $MODE (use: baseline | rkv <budget> | baseline-production)" >&2
    exit 1
    ;;
esac
