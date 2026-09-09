#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

action=run
case "${1:-}" in
  run|start|list|shell|stop) action="$1"; shift ;;
esac
name="${ROOT_CONTAINER_NAME:-root-dev}"
runtime="${ROOT_CONTAINER_RUNTIME:-auto}"
if [[ "$runtime" == auto ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
  else
    runtime=docker
  fi
fi
case "$runtime" in
  apptainer|docker) ;;
  *)
    echo "ROOT_CONTAINER_RUNTIME must be auto, apptainer, or docker." >&2
    exit 1
    ;;
esac
if ! command -v "$runtime" >/dev/null 2>&1; then
  echo "Load or install $runtime before running this script." >&2
  exit 1
fi

# Preserve the scheduler's GPU assignment even with --cleanenv.
if [[ "$runtime" == apptainer && ${CUDA_VISIBLE_DEVICES+x} ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

# These paths are inside the writable cache mount, including in an instance shell.
cache_env=(
  --env HF_HOME=/cache/huggingface
  --env HF_HUB_CACHE=/cache/huggingface/hub
  --env XDG_CACHE_HOME=/cache/huggingface/.root-runtime
  --env TRITON_CACHE_DIR=/cache/huggingface/.root-runtime/triton
)
case "$runtime:$action" in
  apptainer:list) exec apptainer instance list ;;
  apptainer:shell)
    exec apptainer shell --cleanenv "${cache_env[@]}" --pwd /workspace "instance://$name"
    ;;
  apptainer:stop) exec apptainer instance stop "$name" ;;
  docker:list) exec docker ps --filter "name=$name" ;;
  docker:shell) exec docker exec -it --workdir /workspace "$name" bash ;;
  docker:stop) exec docker stop "$name" ;;
esac

cache_dir="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
mkdir -p "$cache_dir"
cache_dir="$(cd -- "$cache_dir" && pwd)"

if [[ "$runtime" == apptainer ]]; then
  image="${ROOT_SIF:-$repo_dir/root.sif}"
  if [[ ! -f "$image" ]]; then
    apptainer build --fakeroot "$image" containers/root.def
  fi
  apptainer test --cleanenv "$image"

  launch=(run --pwd /workspace)
  target=("$image" "$@")
  if [[ "$action" == start ]]; then
    launch=(instance start)
    target=("$image" "$name")
  fi
  exec apptainer "${launch[@]}" --cleanenv --nv \
    --bind "$repo_dir:/workspace" \
    --bind "$cache_dir:/cache/huggingface" \
    "${cache_env[@]}" \
    "${target[@]}"
fi

image="${ROOT_DOCKER_IMAGE:-root:local}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --tag "$image" .
fi
docker run --rm --network none "$image" root-container-test

terminal_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  terminal_flags+=(-t)
fi
container_command=(root --engine transformers --device cuda "$@")
if [[ "$action" == start ]]; then
  terminal_flags=(-d --name "$name")
  container_command=(sleep infinity)
fi
exec docker run --rm \
  "${terminal_flags[@]}" \
  --gpus "${ROOT_DOCKER_GPUS:-all}" \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$repo_dir,dst=/workspace" \
  --mount "type=bind,src=$cache_dir,dst=/cache/huggingface" \
  "${cache_env[@]}" \
  --workdir /workspace \
  "$image" "${container_command[@]}"
