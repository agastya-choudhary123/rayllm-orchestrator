#!/usr/bin/env bash
# Turnkey GPU run -- paste this on any CUDA box (Lambda / RunPod / GCP / EC2).
# Exercises the fast paths that need a real GPU: bitsandbytes 4-bit QLoRA,
# Ray + FSDP multi-GPU training, and vLLM continuous-batching serving.
#
# Requirements on the host: NVIDIA driver + nvidia-container-toolkit + Docker.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
DATASET="${DATASET:-examples/my-data.jsonl}"

echo ">> Building CUDA image (full stack: torch, ray, vllm, bitsandbytes)..."
docker build -f Dockerfile -t rayllm:gpu .

echo ">> QLoRA + FSDP training across all visible GPUs..."
docker run --rm --gpus all \
  -v rayllm-models:/models -v "$PWD/checkpoints:/app/checkpoints" \
  rayllm:gpu train --model "$MODEL" --dataset "$DATASET" \
    --epochs 3 --strategy fsdp-ray --quant 4bit

echo ">> Serving the merged checkpoint with vLLM (continuous batching)..."
docker run -d --name rayllm-vllm --gpus all -p 8000:8000 \
  -v rayllm-models:/models -v "$PWD/checkpoints:/app/checkpoints" \
  rayllm:gpu serve --model ./checkpoints/Phi-3-mini-4k-instruct \
    --quant 4bit --tensor-parallel "$(nvidia-smi -L | wc -l)" \
    --continuous-batching --port 8000 --host 0.0.0.0

echo ">> Waiting for vLLM to load..."
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do sleep 3; done

echo ">> Test request:"
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is FSDP?"}],"max_tokens":64}'
echo
echo ">> Endpoint:  http://localhost:8000/v1/chat/completions"
echo ">> Stop with: docker rm -f rayllm-vllm"
