#!/usr/bin/env bash
# Install root: small tool-using agents on a local open-weight model.
#
#   curl -fsSL https://raw.githubusercontent.com/macabdul9/root/main/install.sh | bash
#
# uv builds and owns an isolated environment for root and puts launchers on your
# PATH. Your system and conda Pythons are untouched.
set -euo pipefail

REPO="${ROOT_REPO:-https://github.com/macabdul9/root}"
REF="${ROOT_REF:-main}"
BIN_DIR="${ROOT_BIN:-${UV_TOOL_BIN_DIR:-$HOME/.local/bin}}"

say() { printf '%s\n' "$*"; }
fail() { printf '%s\n' "$*" >&2; exit 1; }

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  done
}

uv_bin="$(find_uv)"
if [[ -z "$uv_bin" ]]; then
  say "installing uv, which manages the environment root runs in"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  uv_bin="$(find_uv)"
  [[ -n "$uv_bin" ]] || fail "uv installed but could not be found; add ~/.local/bin to PATH and rerun"
fi

export UV_TOOL_BIN_DIR="$BIN_DIR"
[[ -n "${ROOT_HOME:-}" ]] && export UV_TOOL_DIR="$ROOT_HOME"

if [[ "${1:-}" == "--uninstall" ]]; then
  "$uv_bin" tool uninstall root
  exit 0
fi

# Installing from a checkout only when this script is a real file in one: piped
# through bash, $0 is "bash" and the working directory is wherever the user was.
source_dir=""
if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
  candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  grep -qs '^name = "root"' "$candidate/pyproject.toml" && source_dir="$candidate"
fi

say "installing root with $("$uv_bin" --version)"
say "downloading dependencies, which includes PyTorch and takes a few minutes"
if [[ -n "$source_dir" ]]; then
  say "installing from the checkout at $source_dir"
  "$uv_bin" tool install --force "$source_dir"
else
  "$uv_bin" tool install --force "root @ git+$REPO@$REF"
fi

say ""
say "installed $("$BIN_DIR/root" --version)"
say "  launchers: $BIN_DIR/root, $BIN_DIR/root-eval"
say "  weights:   downloaded on first run, into ~/.cache/huggingface"

case ":$PATH:" in
  *":$BIN_DIR:"*) say "" ; say "run: root" ;;
  # uv has already printed the export line to add; this says how to persist it.
  *) say "" ; say "then run: uv tool update-shell   (or add $BIN_DIR to PATH yourself)" ;;
esac
