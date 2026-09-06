#!/usr/bin/env bash
set -euo pipefail
python -m ranking.pipeline --config configs/rank.yaml
