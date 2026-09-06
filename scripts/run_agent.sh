#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-LiquidAI/LFM2.5-350M}"
CONFIG="${CONFIG:-configs/agents.yaml}"
WORKSPACE="${WORKSPACE:-.}"
PYTHON="${PYTHON:-python}"

agent="${1:?usage: scripts/run_agent.sh AGENT [PROMPT...]}"
shift

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# One directory per run holds the config, the environment and the transcript.
run_dir="runs/$(date +%Y%m%d-%H%M%S)-$agent"
mkdir -p "$run_dir"
cp "$CONFIG" "$run_dir/agents.yaml"
"$PYTHON" -m pip freeze > "$run_dir/requirements.txt"

"$PYTHON" -m root.run \
  --agent "$agent" \
  --model "$MODEL" \
  --config "$CONFIG" \
  --workspace "$WORKSPACE" \
  --output-dir "$run_dir" \
  "$@" \
  2>&1 | tee "$run_dir/agent.log"
