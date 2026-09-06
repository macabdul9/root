#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-lfm2-350m}"
WORKSPACE="${WORKSPACE:-.}"
read -r -a run_with <<< "${RUN:-uv run}"

agent="${1:?usage: scripts/run_agent.sh AGENT [PROMPT...]}"
shift

# One directory per run holds the resolved environment and the transcript.
run_dir="runs/$(date +%Y%m%d-%H%M%S)-$agent"
mkdir -p "$run_dir"
uv export --no-hashes > "$run_dir/requirements.txt"

"${run_with[@]}" python -m root.run \
  --agent "$agent" \
  --model "$MODEL" \
  --workspace "$WORKSPACE" \
  --output-dir "$run_dir" \
  "$@" \
  2>&1 | tee "$run_dir/agent.log"
