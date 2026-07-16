# Design Notes: Training & Serving on Any Hardware

The goal: **one CLI that trains and serves an LLM on whatever hardware you have**
— a laptop (CPU/MPS), Apple Silicon, or an NVIDIA GPU node — without the user
choosing backends or writing glue code. This doc explains the engineering
decisions behind that.

For the backend catalog and hardware matrix, see [BACKENDS.md](BACKENDS.md).

---

## 1. Backend abstraction with graceful fallback

Instead of committing to one engine, the orchestrator picks the fastest one
available for the model + hardware, and always keeps a working fallback:

```
Apple Silicon + safetensors  →  MLX
ollama:// model               →  Ollama
.gguf file                    →  llama.cpp
NVIDIA GPU (Linux)            →  vLLM
otherwise                     →  transformers (always works)
```

The key property is **no dead ends**. If the fast path can't load a model — for
example MLX can't read a repo that ships only PyTorch `.bin` weights — training
and serving fall back to the transformers path instead of crashing the run. The
user gets a clear log line, not a traceback.

Each backend is a self-contained module (`orchestrator/backends_mlx.py`,
`serve.py`, `models.py`), so adding one doesn't touch the others.

---

## 2. Apple Silicon as a first-class path (MLX)

On a Mac, the fastest and most memory-safe trainer is MLX — Apple's own
Metal-tuned framework — not the generic PyTorch MPS wrapper.

- 3-4x faster than PyTorch MPS for small-model training/inference
- 4-bit + LoRA fits an 8B model in unified memory on a 16 GB Mac
- A memory guard caps MLX's cache so training can't starve the OS/screen

This is why `train` and `serve` both prefer MLX when it's available and the model
is MLX-loadable, falling back to torch otherwise.

---

## 3. Adaptive data strategy (minimize wasted FLOPs)

The trainer analyzes the dataset's length distribution and picks the padding
strategy that wastes the fewest FLOPs, with no regression on any shape:

- **pack** — concatenate short examples into dense blocks (per-example
  `position_ids` reset). Best when even length-matched batches stay short.
- **bucket** — sort by length so each batch pads only to its own longest member.
  Caveat-free; recovers most of the padding waste.
- **pad** — plain padding (baseline).

In practice bucketing already recovers almost all the padding waste, so the
engine only escalates to packing when it's clearly worth it. See
`orchestrator/data.py` and `orchestrator/fast.py`.

---

## 4. Real training, honest limits

- All training is genuine PyTorch backprop (or MLX-LoRA): tokenized DataLoader,
  forward/backward, AdamW, gradient accumulation, checkpoints, resume.
- QLoRA is the default when quantizing — load base weights in 4/8-bit, train LoRA
  adapters, then merge them back so the checkpoint is a plain HF model any
  runtime can serve.
- The CUDA-only fast paths (vLLM GPU, bitsandbytes, FSDP) are **not sold as
  verified** — they need real NVIDIA hardware to exercise.

---

## Architecture

```
orchestrator.py            ← thin shim (python orchestrator.py ...)
orchestrator/
├── cli.py                ← the `rayllm` command (train / serve / run)
├── train.py              ← training loop + MLX/torch dispatch, fallback
├── serve.py              ← serving + backend dispatch (OpenAI-compatible API)
├── models.py             ← universal model loader + backend selection
├── data.py               ← tokenization, packing/bucketing
├── fast.py               ← bf16, prefetch, packing, LoRA optimizations
├── backends_mlx.py       ← Apple Silicon native train + serve
└── util.py               ← logging, capability detection
```

Each layer is small and single-responsibility, so it can be read and tested on
its own.
