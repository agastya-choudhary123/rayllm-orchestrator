# RayLLM-Orchestrator: Serving Backends

Serving auto-selects the fastest backend available on your hardware. You don't
pick one — the orchestrator detects the model format and installed libraries and
routes to the best option, always with a working fallback.

## The Backends (in priority order)

### 1. MLX — Apple Silicon Native

Training and inference engine built for M-series chips. Faster than PyTorch MPS
and memory-safe in unified memory.

```bash
python orchestrator.py serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

- 3-4x faster than PyTorch MPS on M-series Macs
- Quantization-aware (int4/int8/fp16), LoRA training + serving
- No NVIDIA dependency

**Hardware:** Apple Silicon (arm64 Darwin). **Code:** `orchestrator/backends_mlx.py`

Note: MLX requires safetensors weights. Models that ship only PyTorch `.bin`
files fall back to the transformers path automatically.

---

### 2. Ollama — Local Model Management + Serving

Downloads, caches, and serves quantized models locally with Metal GPU support.

```bash
ollama pull mistral
python orchestrator.py serve --model ollama://mistral --port 8000
```

- Handles quantization automatically (any GGUF model)
- Metal GPU on macOS, CUDA on Linux, CPU fallback
- Model cache management built-in

**Hardware:** CPU/GPU (Metal on macOS, CUDA on Linux, CPU fallback)

---

### 3. llama.cpp — GGUF + CPU Efficient

Loads GGUF (quantized) models and runs inference on CPU with Metal acceleration.

```bash
pip install llama-cpp-python
python orchestrator.py serve --model ./model.gguf
```

- Extremely memory-efficient (GGUF format)
- Metal backend on macOS, AVX2 on x86
- Runs on any CPU

**Hardware:** CPU/GPU (Metal on macOS, AVX2 on x86). **Quant:** GGUF (int4/int8/fp16)

---

### 4. vLLM — GPU Continuous Batching

Fastest serving under concurrent load. Requires a Linux/CUDA host.

```bash
pip install vllm
python orchestrator.py serve --model ./checkpoints/phi-3 --tensor-parallel 2
```

- Continuous batching + PagedAttention (highest throughput)
- Tensor-parallel sharding across GPUs

**Hardware:** NVIDIA GPU (Linux). Not exercised without real GPU hardware.

---

### 5. transformers — Universal Fallback

Plain PyTorch + HuggingFace. Works everywhere, any HF checkpoint.

```bash
python orchestrator.py serve --model ./checkpoints/gpt2
```

- Runs on CPU or GPU, any HF model
- The safety net when no faster backend applies

**Hardware:** CPU/GPU (anywhere torch runs)

---

## Smart Backend Selection

The orchestrator auto-picks based on hardware + model format:

```
Apple Silicon + safetensors  →  MLX
ollama:// model               →  Ollama
.gguf file                    →  llama.cpp
NVIDIA GPU (Linux)            →  vLLM
otherwise                     →  transformers (always works)
```

Examples:

```bash
# Apple Silicon, 4-bit MLX model → MLX
python orchestrator.py serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# GGUF file → llama.cpp
python orchestrator.py serve --model ./mistral.gguf

# Plain HF checkpoint on a Mac → transformers (or MLX if safetensors)
python orchestrator.py serve --model ./checkpoints/gpt2
```

All backends expose the same OpenAI-compatible API.

---

## Hardware Support Matrix

| Backend | CPU | NVIDIA GPU | Apple Silicon | Metal |
|---------|-----|-----------|---------------|-------|
| **MLX** | — | — | ✅ | ✅ |
| **Ollama** | ✅ | ✅ | ✅ | ✅ |
| **llama.cpp** | ✅ | ✅ | ✅ | ✅ |
| **vLLM** | — | ✅ | — | — |
| **transformers** | ✅ | ✅ | ✅ | ✅ (MPS) |

---

## Usage Examples

### Train once, serve any way

```bash
# Train (picks MLX on Apple Silicon, torch elsewhere)
python orchestrator.py train --model gpt2 --dataset examples/my-data.jsonl

# Serve the checkpoint (auto-selected backend)
python orchestrator.py serve --model ./checkpoints/gpt2

# Serve an Ollama-managed model
ollama pull neural-chat
python orchestrator.py serve --model ollama://neural-chat

# Serve a GGUF file via llama.cpp
python orchestrator.py serve --model ./mistral-7b.q4_K_M.gguf
```

### Any model format

```bash
python orchestrator.py serve --model microsoft/Phi-3-mini-4k-instruct  # HF id
python orchestrator.py serve --model ./my-checkpoint                    # local dir
python orchestrator.py serve --model ./mistral-7b.q4_K_M.gguf           # GGUF
python orchestrator.py serve --model ollama://mistral                   # Ollama
```

---

## Design

The backend abstraction lives in `orchestrator/models.py` and
`orchestrator/serve.py`:

- Single OpenAI-compatible interface over several serving engines
- Auto-selection from hardware + installed libraries
- Graceful fallback — training and serving never dead-end; if the fast path
  can't load a model, they fall back to the transformers path instead of
  crashing.
