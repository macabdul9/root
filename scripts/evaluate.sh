#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-lfm2-350m}"
WORKSPACE="${WORKSPACE:-.}"
read -r -a run_with <<< "${RUN:-uv run}"

run_dir="runs/$(date +%Y%m%d-%H%M%S)-evals"
mkdir -p "$run_dir"


"${run_with[@]}" python -m root.evaluate \
  --model "$MODEL" \
  --workspace "$WORKSPACE" \
  --output-dir "$run_dir" \
  "$@" \
  2>&1 | tee "$run_dir/evals.log"
