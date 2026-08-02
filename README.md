# RayLLM-Orchestrator

Fine-tune an LLM on your data and serve it as an API. One command to train, one to serve.

```bash
rayllm train --model gpt2 --dataset my-data.jsonl --epochs 3
rayllm serve --model ./checkpoints/gpt2 --port 8000

# Test it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

---

## What it does

**Train**: PyTorch fine-tuning with LoRA/QLoRA, checkpointing, resume from epoch. On Apple Silicon, native MLX path. Falls back to PyTorch otherwise.

**Serve**: OpenAI-compatible API. Pick your backend (MLX, transformers, llama.cpp, or Ollama).

---

## Installation

```bash
pip install rayllm-orchestrator
```

This installs the core stack: `torch`, `transformers`, `datasets`, `peft`,
`accelerate`. The serving backends below are **optional** and installed
separately — you only need the one(s) you want:

| Backend | Extra install | Where it runs |
|---------|---------------|---------------|
| transformers | (included) | CPU, Apple MPS; works with CUDA if installed |
| MLX | `pip install 'rayllm-orchestrator[mlx]'` | Apple Silicon (arm64 macOS) |
| llama.cpp | `pip install 'rayllm-orchestrator[llama-cpp]'` | CPU, Metal on macOS — GGUF models |
| Ollama | install [Ollama](https://ollama.com) + `pip install 'rayllm-orchestrator[ollama]'` | uses the local Ollama daemon |

### From source
```bash
git clone https://github.com/agastya-choudhary123/rayllm-orchestrator
cd rayllm-orchestrator
pip install -e .
python orchestrator.py train --model gpt2 --dataset my-data.jsonl
```

---

## Quick Start

### 1. Prepare your data

JSONL, one object per line. Supported shapes: `{"text": ...}`,
`{"prompt": ..., "completion": ...}`, or `{"messages": [...]}`.
```json
{"text": "Your training example here."}
{"text": "Another example."}
```

Or point at a HuggingFace dataset id:
```bash
rayllm train --model gpt2 --dataset wikitext --epochs 1
```

### 2. Train
```bash
rayllm train --model gpt2 --dataset my-data.jsonl --epochs 3 --quant 4bit
```
Downloads the model, tokenizes your data, trains (saving a checkpoint after each
epoch), and prints the checkpoint path. With `--quant`, base weights load in
4/8-bit and only LoRA adapters train (QLoRA); the PyTorch path merges the
adapters into a plain HuggingFace checkpoint the serving layer loads directly.

### 3. Serve
```bash
rayllm serve --model ./checkpoints/gpt2 --port 8000
```
Prompts you to choose a backend from what's installed (or pass `--backend`),
then starts an OpenAI-compatible API on port 8000.

### 4. Use it
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about coding"}],"max_tokens":50}'
```

---

## Choosing a backend

List what's actually usable on your machine:
```bash
rayllm backends
```
Example output — availability is probed live (installed **and** runnable here),
and a format-aware recommendation is marked:
```
1. [✗] ollama         needs the ollama daemon running
2. [✓] mlx            Apple-Silicon-native            <- recommended for this model
3. [✗] llama_cpp      `pip install llama-cpp-python`
4. [✓] transformers   universal fallback
```

Then either let `serve`/`run` prompt you (it offers the recommended one as the
Enter-default), or force a choice:
```bash
rayllm serve --model ./checkpoints/gpt2 --backend transformers
```

---

## Common Use Cases

### Apple Silicon (MLX)
```bash
pip install mlx mlx-lm
rayllm serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --backend mlx
```
MLX is Apple's Metal-native framework and keeps 4-bit models memory-safe in
unified memory (a guard caps its cache so training can't starve the OS). See
[measured numbers](#measured-performance).

### Fine-tune with quantization (QLoRA)
```bash
rayllm train --model meta-llama/Llama-3.2-3B --dataset my-data.jsonl --quant 4bit
```
Loads base weights in 4-bit and trains only the LoRA adapters — far fewer
trainable parameters and lower memory than full fine-tuning.

### Speculative decoding (MLX)
```bash
rayllm serve --model <large-mlx-model> --draft-model <small-mlx-model> --backend mlx
```
A small draft model proposes tokens the large model verifies — faster generation
with identical output. MLX backend only.

---

## Command Reference

### `train`
```
rayllm train --model <id-or-path> --dataset <id-or-path.jsonl>
  [--epochs N] [--quant none|8bit|4bit] [--lr LR] [--out DIR]
```
Defaults: `--epochs 1`, `--quant none`, `--lr 2e-5`, `--out ./checkpoints`.
Raise `--lr` to learn faster from little data; lower it if fine-tuning makes the
model worse at the task (see [Does fine-tuning actually help?](#does-fine-tuning-actually-help)).

### `serve`
```
rayllm serve --model <checkpoint-or-id>
  [--backend ollama|mlx|llama_cpp|transformers]
  [--quant ...] [--port 8000] [--host 0.0.0.0]
  [--max-model-len 4096] [--draft-model <id>]
```
If `--backend` is omitted you're prompted to choose (interactive terminals only).

### `run`
```
rayllm run --model <id-or-path> --data <id-or-path.jsonl>
  [--epochs N] [--port 8000] [--backend ...] [--no-serve] [--no-fast]
```
Fine-tune, then serve in one shot.

### `backends`
```
rayllm backends
```
Scan this machine and list which serving backends are available.

---

## Hardware Support

| Hardware | Train | Serve | Notes |
|----------|-------|-------|-------|
| CPU (any OS) | ✓ | ✓ | Tested; ~20 tok/s on CPU |
| Apple Silicon | ✓ | ✓ | Optimized via MLX; ~150-300 tok/s |

---

## Does fine-tuning actually work?

Yes, but only if the base model is bad at the task. We measured this.

500 training examples, 1 epoch, tested on 200 held-out examples. Full reproducible
harness in [examples/eval/](examples/eval/).

| Model | Emotion (6 labels) | Banking77 (77 intents) |
|---|:---:|:---:|
| 0.135B | 20.5% → **67.0%** | 2.5% → *(pending)* |
| 0.5B | 31.0% → **74.5%** | 2.5% → *(pending)* |
| 1.5B | *(running)* | *(running)* |
| 3B | *(pending)* | *(pending)* |
| 8B | *(pending)* | — |

Majority baselines: emotion 37%, banking77 3%.

**The take:**

Fine-tuning gets you +46 points when the model starts at 20%. It does nothing when
the model already scores 90%. Run the base-model eval first to know which case
you're in.

Learning rate matters a lot. The old default (hardcoded, hidden, `1e-4`) made the
0.5B model 23 points worse on emotion. New default is `2e-5`, and it's now a
`--lr` flag you can change.

The `val_loss → healthy` line the trainer prints is not a quality signal. It
ranked a 75% run better than a 91.5% run. Use an eval script, not training curves.

---

## Measured Performance

Serving throughput on one Apple Silicon Mac (MLX, 4-bit weights, 120-token outputs):

| Model | Throughput | Memory |
|-------|-----------:|-------:|
| Qwen2.5-0.5B-Instruct-4bit | ~276 tok/s | 0.33 GB |
| Qwen3-0.6B-4bit | ~240 tok/s | 0.44 GB |
| Qwen3-8B-4bit | ~22 tok/s | 4.72 GB |

Your numbers will differ based on model, quantization, prompt length, and hardware.
Benchmark on your own setup.

---

## Troubleshooting

**`No module named 'torch'`** — install the core stack: `pip install rayllm-orchestrator` (or `pip install torch transformers datasets peft accelerate`).

**Out of memory during training** — use quantization: add `--quant 4bit`.

**"No serving backend is available"** — run `rayllm backends` to see what's
installed, then install one (e.g. `pip install mlx-lm` on a Mac) or pass
`--backend transformers`.

**Serve doesn't respond** — check it's up: `curl http://localhost:8000/v1/models`.

---

## Architecture

```
orchestrator.py            ← thin shim (python orchestrator.py ...)
orchestrator/
├── cli.py                ← the `rayllm` command (train / serve / run / backends)
├── train.py              ← training loop + MLX/torch dispatch, fallback
├── serve.py              ← serving + backend dispatch (OpenAI-compatible API)
├── models.py             ← model loading + backend detection/selection
├── data.py               ← tokenization, packing/bucketing
├── fast.py               ← bf16, prefetch, packing, LoRA optimizations
├── backends_mlx.py       ← Apple Silicon native train + serve (MLX)
└── util.py               ← logging, capability detection

examples/eval/            ← downstream accuracy eval (base vs fine-tuned)
```

Core dependencies: `torch`, `transformers`, `datasets`, `peft`, `accelerate`.
Backends (MLX, llama.cpp, Ollama) are optional and installed as needed.

---

## License

MIT. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## See Also

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [Hugging Face](https://huggingface.co) — models & datasets
