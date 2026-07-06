#!/usr/bin/env bash
# Fetch the SnapKV notebook article (snapkv.txt — the SnapKV paper source) into
# ./data/. This is the long-context prefill used by the compression example.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/data/snapkv.txt"
URL="https://raw.githubusercontent.com/FasterDecoding/SnapKV/main/notebooks/snapkv.txt"

mkdir -p "$DIR/data"
curl -fsSL "$URL" -o "$OUT"
echo "wrote $OUT ($(wc -w < "$OUT") words, $(wc -c < "$OUT") bytes)"
