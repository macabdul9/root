#!/usr/bin/env bash
set -euo pipefail

# Terminal-Bench-Science through Harbor, with root as the agent.
#
# Task containers run in Modal; the model stays on this host, so the GPU here
# does the generating and the sandbox is somebody else's problem. Modal builds
# the task images from their Dockerfiles, which is what makes this work on a
# node that cannot build a container itself.
#
#   scripts/tbs_eval.sh                          # every task, root as the agent
#   TASKS=5 scripts/tbs_eval.sh                  # only the first five
#   AGENT=oracle scripts/tbs_eval.sh             # check the tasks, not the model
#   TASK=tess-transit-vetting scripts/tbs_eval.sh
#
# MODEL       root alias or Hugging Face id
# AGENT       harbor agent; the default runs root
# TASK        one task name from the dataset, for a cheap first run
# TASKS       run only the first N tasks of the dataset
# ATTEMPTS    trials per task
# CONCURRENT  tasks in flight at once
# ENGINE      what generates the tokens (transformers, vllm, ...)
# ENGINE_URL  base URL when ENGINE is a served one

MODEL="${MODEL:-qwen3.8-27b}"
AGENT="${AGENT:-root.harbor:RootAgent}"
ATTEMPTS="${ATTEMPTS:-1}"
CONCURRENT="${CONCURRENT:-8}"
ENGINE="${ENGINE:-transformers}"
DATASET="${DATASET:-terminal-bench-science/terminal-bench-science@latest}"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f "$HOME/.modal.toml" && -z "${MODAL_TOKEN_ID:-}" ]]; then
  echo "Modal is not authenticated. Run: uv run modal token new" >&2
  exit 1
fi

run_dir="runs/tbs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/git-commit.txt"
git diff          > "$run_dir/git-diff.patch"
nvidia-smi        > "$run_dir/nvidia-smi.txt"
cp uv.lock "$run_dir/uv.lock"

# Task names in a dataset are qualified, as in
# terminal-bench-science/tess-transit-vetting. Accept the bare slug too.
task="${TASK:-}"
if [[ -n "$task" && "$task" != */* ]]; then
  qualifier="${DATASET#*/}"
  task="${qualifier%@*}/$task"
fi

# Agent options only mean anything to root's adapter.
agent_kwargs=()
if [[ "$AGENT" == root.* ]]; then
  agent_kwargs=(--ak "engine=$ENGINE")
  if [[ -n "${ENGINE_URL:-}" ]]; then
    agent_kwargs+=(--ak "engine_url=$ENGINE_URL")
  fi
fi

uv run harbor run \
  --dataset "$DATASET" \
  --agent "$AGENT" \
  --model "$MODEL" \
  "${agent_kwargs[@]}" \
  ${task:+--include-task-name "$task"} \
  ${TASKS:+--n-tasks "$TASKS"} \
  --n-attempts "$ATTEMPTS" \
  --n-concurrent "$CONCURRENT" \
  --env modal \
  2>&1 | tee "$run_dir/harbor.log"

echo "wrote $run_dir"
