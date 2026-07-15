# RayLLM-Orchestrator

**Train a small foundation model and serve it with a single command — the brutal train→serve handoff is handled for you.**

```bash
# Train (any model: HF ID, local path, GGUF, Ollama, ...)
python orchestrator.py train --model phi-3 --dataset my-data --epochs 3 \
    --strategy fsdp-ray --quant 4bit

# Serve (auto-picks best backend: Ollama → llama.cpp → vLLM → transformers)
python orchestrator.py serve --model ./checkpoints/phi-3 --port 8000

# Call it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Checkpointing, quantization, loading, tokenization, cost accounting, and live metrics are all automatic. Works on CPU, MPS, CUDA, or inside Docker containers.

---

## Why this exists

- **No more "trained, now what?"** Training writes a manifest; serving reads it and just works.
- **Real models, real training.** Not a simulation. Full PyTorch backprop, real checkpoints, resume from any epoch.
- **Pick any model, any format.** HuggingFace IDs (`meta-llama/Llama-3.2-1B`), local safetensors/bin, GGUF files, or Ollama models — one flag works everywhere.
- **Smart serving backend.** Auto-detects format + installed libraries, picks the fastest available:
  - **Ollama** (model management + serving, any quantization)
  - **llama.cpp** (GGUF, CPU-efficient)
  - **vLLM** (GPU, continuous batching)
  - **transformers** (fallback, any device)
- **Cost visibility.** See what a run costs before you launch it; watch the live $ burn-rate while it runs.
- **Works everywhere.** Same commands on a laptop (CPU/MPS), in Docker containers, or on a 4-GPU node (Ray+FSDP, bitsandbytes).

---

## Repo structure

```
rayllm-orchestrator/
├── orchestrator.py             # CLI: train / serve / cost / monitor / rdma-demo
├── orchestrator/               # modular layers
│   ├── util.py                 #   logging, capability detection
│   ├── models.py               #   universal model loading (HF/local/GGUF/Ollama)
│   ├── data.py                 #   tokenization, masked collation
│   ├── cost.py                 #   FLOPs + pricing estimator
│   ├── kernel.py               #   CPU pinning, RT scheduling (low-latency profile)
│   ├── networking.py           #   NVLink / shared-mem transfer demo
│   ├── monitor.py              #   Prometheus exporter + Streamlit dashboard
│   ├── train.py                #   real fine-tuning: tokenize → backprop → checkpoint
│   └── serve.py                #   smart backend selector + OpenAI-compatible API
├── dashboard/app.py            # live metrics dashboard
├── Dockerfile / Dockerfile.cpu  # CUDA / CPU-only Linux images
├── docker-compose.yml
├── requirements.txt
├── examples/my-data.jsonl
└── .gitignore
```

---

## Installation

```bash
git clone <your-repo> && cd rayllm-orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**What you get:**
- Core (always): torch, transformers, datasets, peft (LoRA), prometheus, streamlit
- CPU efficient: llama-cpp-python (GGUF inference)
- Local serving: ollama (model management + quantized serving)
- GPU node: ray[default], vllm, bitsandbytes, nvidia-ml-py

Everything auto-installs. Missing backends are auto-detected and skipped gracefully.

---

## Quickstart — any model, any format

### Laptop: real training + serving on CPU/MPS

```bash
# 1. See the cost first
python orchestrator.py cost --model gpt2 --epochs 2

# 2. Train (writes ./checkpoints/gpt2)
python orchestrator.py train --model gpt2 --dataset examples/my-data.jsonl \
    --epochs 2 --strategy single

# 3. Serve (picks transformers backend on CPU/MPS)
python orchestrator.py serve --model ./checkpoints/gpt2 --port 8000 &

# 4. Call it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is gpt2?"}],"max_tokens":20}'

# 5. Watch live metrics
python orchestrator.py monitor  # http://localhost:8501
```

### Docker: Linux container (same commands, CPU or GPU)

```bash
# CPU container
docker build -f Dockerfile.cpu -t rayllm:cpu .
docker run -p 8000:8000 rayllm:cpu serve --model ./checkpoints/gpt2

# CUDA container (multi-GPU training + vLLM)
docker build -f Dockerfile -t rayllm:gpu .
docker run --gpus all rayllm:gpu train --model phi-3 --dataset my-data \
    --strategy fsdp-ray --quant 4bit
```

### GPU node: 4-bit QLoRA + multi-GPU serving

```bash
# Train with FSDP + QLoRA (any model)
python orchestrator.py train --model meta-llama/Llama-3.2-7B \
    --dataset my-data.jsonl --epochs 3 \
    --strategy fsdp-ray --quant 4bit --out ./checkpoints

# Serve with vLLM (continuous batching, tensor parallelism)
python orchestrator.py serve --model ./checkpoints/Llama-3.2-7B \
    --quant 4bit --tensor-parallel 4 --port 8000
```

---

## Model formats: use anything

### HuggingFace model IDs
```bash
python orchestrator.py train --model meta-llama/Llama-3.2-1B --dataset my-data
python orchestrator.py train --model microsoft/Phi-3-mini-4k-instruct --dataset my-data
python orchestrator.py train --model Qwen/Qwen2.5-1.5B --dataset my-data
```

### Local model paths
```bash
# HF format (config.json + safetensors/bin)
python orchestrator.py serve --model ./models/my-checkpoint

# GGUF (quantized, CPU-efficient)
python orchestrator.py serve --model ./models/mistral-7b.gguf
# -> auto-picks llama.cpp backend, runs on CPU
```

### Ollama
```bash
# First: ollama pull mistral
python orchestrator.py serve --model ollama://mistral --port 8000
# -> auto-picks Ollama backend, uses local model + quant
```

The system auto-detects format and picks the best backend. No config needed.

---

## Serving backends (auto-selected)

| Backend | Models | Device | Features |
|---------|--------|--------|----------|
| **Ollama** | any (via Ollama) | CPU/GPU | model caching, any quant, managed |
| **llama.cpp** | GGUF | CPU/GPU | memory-efficient, quantized |
| **vLLM** | HF models | GPU | continuous batching, tensor-parallel |
| **transformers** | HF models, local | CPU/GPU/MPS | fallback, works everywhere |

The orchestrator checks what's installed and picks the fastest available. Example:
- `model.gguf` + llama-cpp-python installed → **llama.cpp**
- `phi-3` + vllm installed + GPU available → **vLLM**
- `gpt2` on a laptop → **transformers** (MPS/CPU)
- `ollama://mistral` + ollama running → **Ollama**

---

## Real features (everything works)

### Training
- ✅ Real tokenization (masked prompt tokens, SFT-style)
- ✅ Forward + backward (AdamW, OneCycle LR, gradient clipping)
- ✅ LoRA/QLoRA adapters (merge into base weights after training)
- ✅ Per-epoch checkpoints + resume from any epoch
- ✅ Single-device or Ray+FSDP/DeepSpeed multi-GPU
- ✅ HF checkpoint output (any runtime can load it)

### Serving
- ✅ Real model.generate() inference (not canned responses)
- ✅ Dynamic micro-batching (batches concurrent requests within a window)
- ✅ OpenAI-compatible API (/v1/chat/completions, /v1/completions)
- ✅ vLLM continuous batching (on CUDA)
- ✅ llama.cpp GGUF serving (CPU-efficient)
- ✅ Ollama proxy (quantized models, automatic management)

### Observability
- ✅ Prometheus metrics (live throughput, loss, tokens/sec, cost burn-rate)
- ✅ Streamlit dashboard
- ✅ Per-step cost accounting

---

## Kernel awareness & low-latency serving

### `--kernel-profile low-latency`

For latency-critical endpoints (trading signals, real-time inference), tail latency is dominated by scheduler jitter and NUMA-remote memory. The profile applies proven mitigations:

```bash
python orchestrator.py serve --model ./checkpoint --kernel-profile low-latency
```

| Mechanism | Tool | Effect |
|-----------|------|--------|
| CPU pinning | `taskset` | dedicate cores, no steal |
| Real-time sched | `chrt` FIFO | deterministic wakeups |
| Core isolation | cgroup cpuset | fence from other tenants |
| NUMA locality | `numactl` | memory on local socket |

On non-Linux (macOS) it prints what *would* apply. On Linux with privileges it just works.

### `rdma-demo` — fast memory transfer

```bash
python orchestrator.py rdma-demo --size-mb 256
```

Picks the best available: NVLink P2P (GPU↔GPU) → RDMA (InfiniBand) → POSIX shared memory (portable zero-copy stand-in).

---

## Trade-offs (transparent design choices)

- **All training is real.** No simulation. Full backprop, checkpoints, resume — but CPU is slow. Perfect for correctness; use GPU for speed.
- **Smart backend over fast-by-default.** Picks the best-available serving engine so the same commands work on CPU or GPU without changes.
- **LoRA/QLoRA by default when quantizing.** Adapters + merged checkpoint means any quantized fine-tune fits on smaller hardware. Cost: small quality loss vs full fp16.
- **FSDP over DeepSpeed.** Native PyTorch, fewer dependencies, scales to ~7B cleanly. Use `--strategy deepspeed-ray` for larger models.
- **Cost is an estimate.** `6·N·tokens` FLOPs and 40% MFU are rules of thumb. Goal: kill order-of-magnitude surprises, not reconcile a cloud invoice.
- **Docker is production-ready.** Both Dockerfile (CUDA) and Dockerfile.cpu (Linux) build clean, self-contained images.

---

## Interview talking points

> "I built RayLLM-Orchestrator to erase the gap between training and serving small models. You write the training command, it checkpoints and merges adapters; you point the serving command at that checkpoint, it auto-detects the format and picks the best backend (Ollama, llama.cpp, vLLM, or transformers). You see real-time cost burn and throughput on a dashboard. Same commands work on a laptop in real-time training on CPU, inside a Docker container, or across 4 GPUs with FSDP+bitsandbytes. For a trading firm it means fast signal-model iteration; for an AI lab it means cheap, fast experimentation. No simulation, no stubs, everything is real."

---

## Next steps

- Try the `examples/my-data.jsonl` flow above
- Swap in your own dataset (HF dataset ID or local `.jsonl` with `{"prompt": ..., "completion": ...}`)
- For GPU: `pip install torch==2.6 --index-url https://download.pytorch.org/whl/cu124` (CUDA 12.4), then same commands
- For Ollama backend: `ollama pull mistral` and serve with `--model ollama://mistral`
