#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-LiquidAI/LFM2.5-350M}"
CONFIG="${CONFIG:-configs/agents.yaml}"
EVALS="${EVALS:-configs/evals.yaml}"
WORKSPACE="${WORKSPACE:-.}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "$EVALS" ]]; then
  echo "Eval set not found: $EVALS" >&2
  exit 1
fi

run_dir="runs/$(date +%Y%m%d-%H%M%S)-evals"
mkdir -p "$run_dir"
cp "$CONFIG" "$run_dir/agents.yaml"
cp "$EVALS" "$run_dir/evals.yaml"

"$PYTHON" -m root.evaluate \
  --evals "$EVALS" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --workspace "$WORKSPACE" \
  --output-dir "$run_dir" \
  "$@" \
  2>&1 | tee "$run_dir/evals.log"
