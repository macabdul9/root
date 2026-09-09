#!/usr/bin/env bash
set -euo pipefail

# IFEval and InFoBench over the sub-1B models: one process per GPU, then one
# judging pass, then one aggregate.
#
#   scripts/instruct_eval.sh                       # every sub-1B model
#   scripts/instruct_eval.sh lfm2-350m qwen3-06b   # only these
#
# GPUS              comma-separated device ids to spread across
# SIF               image to run in
# LIMIT             first N cases per benchmark, for a smoke run
# JUDGE             model that grades InFoBench
# BATCH_SIZE        prompts per generation batch
# JUDGE_BATCH_SIZE  questions per judging batch; lower it for a verbose model,
#                   whose answers make the judge prompts several times longer

GPUS="${GPUS:-0}"
SIF="${SIF:-root-eval.sif}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE="${JUDGE:-Qwen/Qwen3.8-27B}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-32}"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f "$SIF" ]]; then
  echo "Image not found: $SIF (build it with scripts/container.sh)" >&2
  exit 1
fi

cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$cache_dir"

run_dir="runs/instruct/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"

# What the numbers came from. uv.lock rather than a pip freeze: the packages
# live in the image, and the lock is what pinned them.
git rev-parse HEAD       > "$run_dir/git-commit.txt"
git diff                 > "$run_dir/git-diff.patch"
nvidia-smi               > "$run_dir/nvidia-smi.txt"
apptainer inspect "$SIF" > "$run_dir/image.txt"
cp uv.lock "$run_dir/uv.lock"

limit=()
if [[ -n "${LIMIT:-}" ]]; then
  limit=(--limit "$LIMIT")
fi
IFS=',' read -r -a gpus <<< "$GPUS"

# One invocation of the eval, inside the image, pinned to one GPU.
instruct() {
  local gpu="$1"
  shift
  APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" apptainer exec --cleanenv --nv \
    --bind "$repo_dir:/checkout" \
    --bind "$cache_dir:/cache/huggingface" \
    --env HF_HOME=/cache/huggingface \
    --env PYTHONPATH=/checkout/src \
    "$SIF" python -m root.instruct \
    --output-dir "/checkout/$run_dir" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${limit[@]}" \
    "$@"
}

# Joining with commas is how the eval takes a list of models.
commas() {
  local IFS=,
  echo "$*"
}

# Which models to run. Asking the eval keeps the sub-1B rule in one place.
if (( $# > 0 )); then
  models=("$@")
else
  read -r -a models <<< "$(instruct "${gpus[0]}" --list-models)"
fi
echo "models: ${models[*]}" | tee "$run_dir/models.txt"

# One process per GPU, each working through its own share. The eval loads one
# model at a time, so a share is a queue and no two processes touch a card.
for offset in "${!gpus[@]}"; do
  share=()
  for index in "${!models[@]}"; do
    if (( index % ${#gpus[@]} == offset )); then
      share+=("${models[index]}")
    fi
  done
  if (( ${#share[@]} == 0 )); then
    continue
  fi
  echo "gpu ${gpus[offset]}: ${share[*]}"
  instruct "${gpus[offset]}" \
    --models "$(commas "${share[@]}")" \
    --skip-judge \
    > "$run_dir/generate-gpu${gpus[offset]}.log" 2>&1 &
done
wait

# The judge is the largest set of weights in the run, so it is loaded once and
# grades every model's saved answers.
instruct "${gpus[0]}" \
  --models "$(commas "${models[@]}")" \
  --benchmark infobench \
  --judge "$JUDGE" \
  --judge-batch-size "$JUDGE_BATCH_SIZE" \
  --judge-only \
  2>&1 | tee "$run_dir/judge.log"

instruct "${gpus[0]}" \
  --models "$(commas "${models[@]}")" \
  --report-only \
  2>&1 | tee "$run_dir/report.log"

echo "wrote $run_dir/report.json"
