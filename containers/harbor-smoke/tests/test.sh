#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
if [ "$(cat /app/answer.txt 2>/dev/null | tr -d '[:space:]')" = "42" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi
