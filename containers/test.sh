#!/usr/bin/env bash
set -euo pipefail

# Copy fixtures to writable storage, including when the image is a read-only SIF.
source_dir="${1:-/opt/root}"
if (( $# > 0 )); then
  shift
fi
run_dir="$(mktemp -d "${TMPDIR:-/tmp}/root-tests.XXXXXXXX")"
trap 'rm -rf "$run_dir"' EXIT
cp -R "$source_dir/src" "$source_dir/tests" "$source_dir/pyproject.toml" "$run_dir/"
cd "$run_dir"
# The git tool tests need a repository; no host history or credentials are copied.
git init --quiet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run_dir/src"
python -m pytest -q -p no:cacheprovider "$@"
