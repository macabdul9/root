#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

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

cache_dir="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
mkdir -p "$cache_dir"
cache_dir="$(cd -- "$cache_dir" && pwd)"

if [[ "$runtime" == apptainer ]]; then
    image="${ROOT_SIF:-$repo_dir/root.sif}"
    if [[ ! -f "$image" ]]; then
        apptainer build --fakeroot "$image" containers/root.def
    fi
    apptainer test --cleanenv "$image"

    # Preserve the scheduler's GPU assignment even with --cleanenv.
    if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
        export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
    exec apptainer run --cleanenv --nv \
        --bind "$repo_dir:/workspace" \
        --bind "$cache_dir:/cache/huggingface" \
        --env HF_HOME=/cache/huggingface \
        --pwd /workspace \
        "$image" "$@"
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
exec docker run --rm \
    "${terminal_flags[@]}" \
    --gpus "${ROOT_DOCKER_GPUS:-all}" \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$repo_dir,dst=/workspace" \
    --mount "type=bind,src=$cache_dir,dst=/cache/huggingface" \
    --env HF_HOME=/cache/huggingface \
    --workdir /workspace \
    "$image" root --engine transformers --device cuda "$@"
