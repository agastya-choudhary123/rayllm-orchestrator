#!/usr/bin/env bash
# One-command serving.
#   ./serve.sh                                  # serve ./checkpoints/phi-3
#   MODEL=./checkpoints/phi-3 PORT=8000 ./serve.sh
set -euo pipefail
cd "$(dirname "$0")"

python orchestrator.py serve \
  --model "${MODEL:-./checkpoints/phi-3}" \
  --quant "${QUANT:-4bit}" \
  --port "${PORT:-8000}" \
  --tensor-parallel "${TP:-1}" \
  --continuous-batching \
  --kernel-profile "${KERNEL_PROFILE:-default}" \
  "$@"
