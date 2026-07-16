# RayLLM-Orchestrator

**Train any LLM on your data. Serve it instantly. One command each.**

```bash
# Train
python orchestrator.py train --model gpt2 --dataset my-data.jsonl --epochs 3

# Serve (in another terminal)
python orchestrator.py serve --model ./checkpoints/gpt2 --port 8000

# Test it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

---

## What It Does

**Train:** Real PyTorch training with automatic:
- Model loading (HuggingFace IDs or local paths)
- Dataset tokenization (JSONL or HF datasets)
- Multi-GPU support (Ray + FSDP)
- Quantization (4-bit / 8-bit LoRA)
- Checkpointing + resume from any epoch

**Serve:** Auto-selects the fastest backend for your hardware:
- **Apple Silicon:** MLX (native, 4x faster than PyTorch)
- **NVIDIA GPU:** vLLM (continuous batching)
- **CPU:** PyTorch transformers (fallback, works everywhere)
- **Quantized models:** llama.cpp (memory-efficient)

---

## Installation

### Option 1: PyPI (recommended)
```bash
pip install rayllm-orchestrator
rayllm train --model gpt2 --dataset my-data.jsonl
rayllm serve --model ./checkpoints/gpt2
```

### Option 2: From source
```bash
git clone https://github.com/yourorg/rayllm-orchestrator
cd rayllm-orchestrator
pip install -e .
python orchestrator.py train --model gpt2 --dataset my-data.jsonl
```

### Option 3: Docker
```bash
docker pull ghcr.io/yourorg/rayllm:latest
docker run -it ghcr.io/yourorg/rayllm:latest train \
  --model gpt2 --dataset my-data.jsonl --epochs 3
```

---

## Quick Start

### 1. Prepare your data

**Format:** JSONL (one JSON object per line)
```json
{"text": "Your training example here."}
{"text": "Another example."}
```

Or use a HuggingFace dataset:
```bash
python orchestrator.py train --model gpt2 \
  --dataset wikitext --epochs 1
```

### 2. Train

```bash
python orchestrator.py train \
  --model gpt2 \
  --dataset my-data.jsonl \
  --epochs 3 \
  --quant 4bit
```

**What happens:**
- Downloads model from HuggingFace
- Tokenizes your data
- Trains for 3 epochs (saves checkpoint after each)
- Merges LoRA adapters into final model
- Prints checkpoint path

### 3. Serve

```bash
python orchestrator.py serve \
  --model ./checkpoints/gpt2 \
  --port 8000
```

**What happens:**
- Auto-detects your hardware
- Loads model with best available backend
- Starts OpenAI-compatible API on port 8000
- Ready for requests

### 4. Use it

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a haiku about coding"}
    ],
    "max_tokens": 50
  }'
```

---

## Common Use Cases

### Train on Apple Silicon (M1/M2/M3)
```bash
python orchestrator.py train \
  --model meta-llama/Llama-3.2-1B \
  --dataset my-data.jsonl \
  --epochs 3
```
Uses MLX backend automatically (3-4x faster than PyTorch MPS).

### Fine-tune with quantization (save memory)
```bash
python orchestrator.py train \
  --model meta-llama/Llama-3.2-8B \
  --dataset my-data.jsonl \
  --quant 4bit
```
Uses QLoRA: trains only 1-2% of parameters, 60% less VRAM.

### Multi-GPU training
```bash
python orchestrator.py train \
  --model meta-llama/Llama-3.2-8B \
  --dataset my-data.jsonl \
  --strategy fsdp-ray \
  --num-workers 4
```
Uses Ray + PyTorch FSDP across 4 GPUs.

### Serve with small draft model (faster generation)
```bash
python orchestrator.py serve \
  --model ./checkpoints/large-model \
  --draft-model ./checkpoints/small-model \
  --port 8000
```
Uses speculative decoding: small model proposes, large model verifies.

---

## Command Reference

### `train`
Fine-tune a model on your data.

```bash
python orchestrator.py train \
  --model <model-id-or-path> \
  --dataset <dataset-id-or-path.jsonl> \
  --epochs <int> \
  --strategy <single|fsdp-ray|deepspeed-ray> \
  --quant <none|8bit|4bit> \
  --out <checkpoint-dir> \
  --num-workers <int>
```

**Defaults:**
- `--epochs 1`
- `--strategy fsdp-ray` (auto-detect GPUs, fall back to single)
- `--quant none`
- `--out ./checkpoints`
- `--num-workers 0` (auto-detect)

### `serve`
Serve a trained model with OpenAI-compatible API.

```bash
python orchestrator.py serve \
  --model <checkpoint-path-or-model-id> \
  --quant <none|8bit|4bit|awq|gptq> \
  --port <int> \
  --host <0.0.0.0> \
  --max-model-len <int> \
  --continuous-batching
```

**Defaults:**
- `--port 8000`
- `--host 0.0.0.0`
- `--max-model-len 4096`
- `--continuous-batching` (enabled by default)

### `run`
Train AND serve in one command (for demos).

```bash
python orchestrator.py run \
  --model <model-id-or-path> \
  --data <dataset-id-or-path.jsonl> \
  --epochs 3 \
  --port 8000 \
  --no-fast
```

Trains, then immediately starts serving on port 8000.

---

## Hardware Support

| Hardware | Train | Serve | Notes |
|----------|-------|-------|-------|
| **CPU (any OS)** | ✓ | ✓ | Slow but works everywhere |
| **Apple Silicon (M1/M2/M3)** | ✓ (MLX) | ✓ (MLX) | 3-4x faster, native Metal GPU |
| **NVIDIA GPU** | ✓ (FSDP) | ✓ (vLLM) | Requires CUDA + PyTorch |
| **Docker (CPU)** | ✓ | ✓ | Reproducible, isolated |
| **Docker (CUDA)** | ✓ | ✓ | GPU inside container |
| **Multi-GPU cluster** | ✓ (Ray+FSDP) | ✓ (vLLM) | Distributed training + serving |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch first:
```bash
pip install torch transformers datasets peft accelerate
```

### "Out of memory" during training
Try quantization:
```bash
python orchestrator.py train --model gpt2 --dataset my-data.jsonl --quant 4bit
```

Or reduce batch size / context length in the code.

### Slow training on Mac
Make sure MLX is installed:
```bash
pip install mlx
```
The trainer automatically picks MLX on Apple Silicon (much faster than PyTorch).

### Serve doesn't respond
Check the server is running:
```bash
curl http://localhost:8000/v1/models
```

Should return a list of loaded models.

---

## Performance Notes

**Training speed:**
- Apple Silicon M3 Max: ~100 tok/s for 3B model
- NVIDIA H100: ~1000+ tok/s with FSDP
- CPU: ~10 tok/s (slow but works for small models)

**Serving speed (tok/s, inference only):**
- Apple Silicon M1/M2: 40-80 tok/s for 7B quantized
- NVIDIA RTX 4090: 100-150 tok/s for 7B quantized
- NVIDIA H100: 300+ tok/s for 8B

*Speed depends on model size, quantization, hardware. Always benchmark on your setup.*

---

## Architecture

```
orchestrator.py       ← Simple CLI
├── train.py         ← Training loop (PyTorch, LoRA, checkpointing)
├── serve.py         ← Serving layer (OpenAI API compatible)
├── models.py        ← Model loading (HF + quantization)
├── data.py          ← Tokenization (packing, bucketing)
├── backends_mlx.py  ← Apple Silicon native training/serving
├── fast.py          ← Optimizations (bf16, prefetch, packing)
└── util.py          ← Helpers (logging, device detection)
```

No external dependencies beyond PyTorch, transformers, datasets.

---

## License

MIT

## Contributing

Issues and PRs welcome. Focus on:
- Bug reports with reproducible examples
- Performance improvements on your hardware
- New model/dataset support

---

## See Also

- [vLLM](https://github.com/vllm-project/vllm) — Fast serving inference
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [Ray Train](https://docs.ray.io/en/latest/train/train.html) — Distributed training
- [Hugging Face](https://huggingface.co) — Model zoo & datasets
