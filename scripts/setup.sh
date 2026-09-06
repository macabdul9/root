#!/usr/bin/env bash
set -euo pipefail

# uv owns the environment: it creates .venv, resolves against uv.lock, and keeps
# the two in step. Nothing here touches the system or conda Python.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it into ~/.local/bin" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra dev

cat <<'MESSAGE'

Run without activating:   uv run root
                          uv run pytest
Or activate as usual:     source .venv/bin/activate
MESSAGE
