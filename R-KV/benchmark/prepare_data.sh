#!/usr/bin/env bash
# Fetch the GSM8K-style few-shot eval set from the `dev` branch into ./data/.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
OUT="$DIR/data/gsm8k_fewshot.jsonl"

mkdir -p "$DIR/data"
cd "$REPO"
git fetch origin dev
git show origin/dev:evaluation/data/test.jsonl > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
