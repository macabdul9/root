#!/usr/bin/env bash
set -euo pipefail

# Smoke-test every agent against one prompt each, on whatever MODEL is set.
run() {
  local agent="$1"
  shift
  echo "=== $agent ==="
  scripts/run_agent.sh "$agent" "$@"
  echo
}

run chat "Why is a 350M parameter model faster than a 7B one?"
run calc "A box holds 24 pens. I buy 7 boxes and give away 13 pens. How many are left?"
run python "What is the sum of the squares of the numbers 1 through 20?"
run files "What Python version does pyproject.toml require?"
run search "Which file defines the run_agent function?"
run extract "Invoice from Dana Ruiz, dated 2026-03-14, for 412.50 USD of GPU rental."
