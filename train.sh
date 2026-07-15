#!/usr/bin/env bash
# One-command training. Edit the flags or pass your own through.
#   ./train.sh                      # sensible demo defaults
#   ./train.sh --model phi-3 ...    # override anything
set -euo pipefail
cd "$(dirname "$0")"

python orchestrator.py train \
  --model "${MODEL:-phi-3}" \
  --dataset "${DATASET:-my-data}" \
  --epochs "${EPOCHS:-3}" \
  --strategy "${STRATEGY:-fsdp-ray}" \
  --quant "${QUANT:-4bit}" \
  --kernel-profile "${KERNEL_PROFILE:-default}" \
  "$@"
