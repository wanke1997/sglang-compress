#!/usr/bin/env bash
# Launch an SGLang server for the SnapKV benchmark.
#
# SnapKV is a *prompt-phase* KV-cache compressor: right after a prompt is
# prefilled, it keeps only the `max_capacity_prompt` prompt tokens that the
# trailing observation window attends to most and physically frees the rest.
#
# Usage:
#   ./launch_server.sh snapkv 1024            # SnapKV ON,  max_capacity_prompt=1024
#   ./launch_server.sh snapkv-baseline        # SnapKV OFF, same eager flags (fair compare)
#
# Env overrides: MODEL, PORT, SNAP_WINDOW, SNAP_KERNEL, SNAP_POOLING, MEM_FRAC, DP
set -euo pipefail

MODE="${1:-snapkv}"
BUDGET="${2:-1024}"                              # max_capacity_prompt
MODEL="${MODEL:-/data/model/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
SNAP_WINDOW="${SNAP_WINDOW:-32}"
SNAP_KERNEL="${SNAP_KERNEL:-5}"
SNAP_POOLING="${SNAP_POOLING:-avgpool}"
MEM_FRAC="${MEM_FRAC:-0.6}"
DP="${DP:-1}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO/python"
export HF_HUB_DISABLE_XET=1  # HF Xet transfer can hang on large files

# SnapKV requires: eager decode (dynamic positions after eviction), radix/prefix
# cache OFF (SnapKV frees slots the radix tree still references), overlap OFF
# (simple timing), page_size=1 (clean per-slot free), and chunked prefill OFF so
# the whole prompt (and its observation window) is seen in a single forward.
SNAPKV_FLAGS=(--disable-decode-cuda-graph --disable-prefill-cuda-graph
              --disable-overlap-schedule --disable-radix-cache --page-size 1
              --chunked-prefill-size -1)

COMMON=(--model-path "$MODEL" --attention-backend flashinfer
        --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$PORT")

DP_FLAGS=()
if [[ "$DP" -gt 1 ]]; then
  DP_FLAGS=(--dp-size "$DP" --tp-size 1)
fi

case "$MODE" in
  snapkv)
    echo ">> SnapKV ON | max_capacity_prompt=$BUDGET window=$SNAP_WINDOW kernel=$SNAP_KERNEL pooling=$SNAP_POOLING dp=$DP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${SNAPKV_FLAGS[@]}" "${DP_FLAGS[@]}" \
      --enable-snapkv \
      --snapkv-config "{\"max_capacity_prompt\":$BUDGET,\"window_size\":$SNAP_WINDOW,\"kernel_size\":$SNAP_KERNEL,\"pooling\":\"$SNAP_POOLING\"}"
    ;;
  snapkv-baseline)
    echo ">> BASELINE (eager, same flags as SnapKV, no --enable-snapkv) dp=$DP"
    exec python3 -m sglang.launch_server "${COMMON[@]}" "${SNAPKV_FLAGS[@]}" "${DP_FLAGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE (use: snapkv <max_capacity_prompt> | snapkv-baseline)" >&2
    exit 1
    ;;
esac
