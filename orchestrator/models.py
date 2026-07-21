"""Universal model loader -- supports any format and source.

Detects and loads:
  * local paths (.safetensors, .bin, GGUF)
  * HuggingFace model IDs (microsoft/Phi-3-mini, etc)
  * Ollama models (ollama://mistral, ollama://neural-chat, etc)
  * Paths in GGUF format (especially from huggingface-hub quantized models)

Training always uses transformers (standard fine-tuning pipeline).
Serving auto-picks the best engine based on format + what's installed.
"""

from __future__ import annotations

import os
from pathlib import Path

from .util import have, log


# --------------------------------------------------------------------------- #
# Model type detection
# --------------------------------------------------------------------------- #
class ModelSpec:
    """Metadata about a model: format, source, whether it's quantized, etc."""

    def __init__(self, path_or_id: str):
        self.original = path_or_id
        self.is_local = os.path.exists(path_or_id)
        self.is_gguf = str(path_or_id).endswith(".gguf")
        self.is_ollama = path_or_id.startswith("ollama://")
        self.is_huggingface = not (self.is_local or self.is_ollama)

        if self.is_ollama:
            self.model_name = path_or_id.replace("ollama://", "")
        elif self.is_local:
            self.model_name = Path(path_or_id).stem
        else:
            self.model_name = path_or_id.split("/")[-1]

    def __repr__(self):
        fmt = "gguf" if self.is_gguf else ("ollama" if self.is_ollama else
                                           ("local" if self.is_local else "hf"))
        return f"ModelSpec({self.model_name}, {fmt})"


# --------------------------------------------------------------------------- #
# Training loader -- always transformers, but tolerant of any input
# --------------------------------------------------------------------------- #
def load_for_training(model_id: str, quant: str = "none", device: str = "cpu"):
    """Load a model for training (fine-tuning). Always uses transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = ModelSpec(model_id)
    log(f"Loading for training: {spec}")

    # If it's a GGUF, we can't train it directly -- convert to transformers.
    if spec.is_gguf:
        raise ValueError(
            f"Cannot train on GGUF format (quantized for inference). "
            f"For training, use the base model (e.g., a safetensors checkpoint "
            f"or a HuggingFace model ID). If you have a GGUF, train against the "
            f"original base model and quantize after.")

    if spec.is_ollama:
        raise ValueError(
            f"Cannot train via Ollama. Ollama is for serving only. "
            f"For training, use a HuggingFace model ID or local safetensors/bin path.")

    # Load tokenizer (works for all transformers-compatible inputs).
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        raise ValueError(
            f"Could not load tokenizer from '{model_id}'. "
            f"Is it a valid HF model or local path? Error: {e}")

    # Load model with optional quantization config.
    kwargs = {"torch_dtype": torch.bfloat16}
    if quant in ("4bit", "8bit") and have("bitsandbytes") and device.startswith("cuda"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quant == "4bit"),
            load_in_8bit=(quant == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Could not load model from '{model_id}'. Error: {e}")

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Serving backend selector -- picks the best engine for the model
# --------------------------------------------------------------------------- #
def select_serving_backend(model_id: str):
    """Inspect model format and installed libraries, return the best serving backend.

    Priority (fastest → most compatible):
      1. Ollama (model management + serving, Metal on macOS)
      2. MLX (Apple Silicon native, faster than PyTorch MPS)
      3. llama.cpp (GGUF + CPU, Metal on macOS)
      4. vLLM (CPU backend via Ray, or GPU)
      5. transformers (fallback, CPU/GPU/MPS)
    """
    spec = ModelSpec(model_id)

    # Ollama: local model management + serving (has Metal support on macOS)
    if spec.is_ollama and have("ollama"):
        try:
            import ollama
            _ = ollama.list()
            return "ollama", spec
        except Exception:
            pass

    # MLX: Apple Silicon native (fastest M-series inference)
    try:
        from . import backends_mlx
        if backends_mlx.is_available() and not spec.is_gguf:
            log("MLX available: Apple Silicon native inference selected")
            return "mlx", spec
    except ImportError:
        pass

    # llama.cpp: GGUF + CPU (Metal support on macOS)
    if spec.is_gguf:
        if have("llama_cpp"):
            return "llama_cpp", spec
        else:
            log("GGUF found but llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python")

    # vLLM: its real value is GPU continuous batching. Only auto-select it when
    # CUDA is present -- on a CPU/MPS laptop it's slow to boot and transformers
    # gives far better UX. (Force it anywhere with --backend vllm.)
    if have("vllm") and not spec.is_gguf:
        try:
            import torch
            if torch.cuda.is_available():
                log("vLLM + CUDA available: continuous-batching backend selected")
                return "vllm", spec
        except Exception:
            pass

    # transformers: fast, reliable fallback for any device (CPU/GPU/MPS)
    if have("torch") and have("transformers"):
        return "transformers", spec

    raise RuntimeError(
        f"Cannot serve '{model_id}'. No compatible backend found. "
        f"Install one of: ollama, llama-cpp-python, vllm, or torch+transformers.")


# --------------------------------------------------------------------------- #
# Backend discovery -- scan the system so the USER can choose (not just auto)
# --------------------------------------------------------------------------- #
def _torch_devices() -> list[str]:
    """Compute devices torch can actually reach on this machine."""
    if not have("torch"):
        return []
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    devices.append("cpu")
    return devices


def scan_backends(model_id: str | None = None) -> list[dict]:
    """Probe the system and report every serving backend + whether it's usable.

    Returns an ordered list (fastest → most compatible) of descriptors:
        {name, available, device, note}
    `available` means installed AND runnable here; `note` explains why not, or
    what it buys you. If `model_id` is given, the format-aware `recommended`
    flag marks the one `select_serving_backend` would auto-pick.
    """
    spec = ModelSpec(model_id) if model_id else None
    devices = _torch_devices()
    accel = next((d for d in devices if d != "cpu"), "cpu")

    def _mlx_ok() -> bool:
        try:
            from . import backends_mlx
            return backends_mlx.is_available()
        except Exception:
            return False

    def _ollama_ok() -> bool:
        if not have("ollama"):
            return False
        try:
            import ollama
            ollama.list()
            return True
        except Exception:
            return False

    def _cuda() -> bool:
        return any(d.startswith("cuda") for d in devices)

    rows = [
        {"name": "ollama", "available": _ollama_ok(),
         "device": "metal/gpu",
         "note": "local model manager; needs the ollama daemon running"
                 if not _ollama_ok() else "daemon reachable"},
        {"name": "mlx", "available": _mlx_ok(),
         "device": "apple-silicon",
         "note": "Apple-Silicon-native, fastest M-series inference"
                 if _mlx_ok() else "requires Apple Silicon + `pip install mlx-lm`"},
        {"name": "llama_cpp", "available": have("llama_cpp"),
         "device": "cpu/metal",
         "note": "GGUF models; Metal on macOS"
                 if have("llama_cpp") else "`pip install llama-cpp-python`"},
        {"name": "vllm", "available": have("vllm") and _cuda(),
         "device": accel,
         "note": "continuous batching, best on CUDA"
                 if have("vllm") and _cuda()
                 else ("installed but no CUDA GPU found" if have("vllm")
                       else "`pip install vllm` (needs a CUDA GPU)")},
        {"name": "transformers", "available": have("torch") and have("transformers"),
         "device": accel,
         "note": f"universal fallback on {accel}"
                 if have("torch") and have("transformers")
                 else "`pip install torch transformers`"},
    ]

    if spec is not None:
        try:
            auto, _ = select_serving_backend(model_id)
        except Exception:
            auto = None
        for r in rows:
            r["recommended"] = (r["name"] == auto)
    return rows


# --------------------------------------------------------------------------- #
# Backend loaders
# --------------------------------------------------------------------------- #
def load_for_serving_transformers(model_id: str, device: str):
    """Load via transformers for serving."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    ).to(device).eval()
    model.config.use_cache = True
    return model, tokenizer


def load_for_serving_llama_cpp(model_path: str):
    """Load a GGUF model via llama.cpp."""
    from llama_cpp import Llama
    log(f"Loading GGUF with llama.cpp: {model_path}")
    model = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)
    # Return a simple wrapper that mimics our inference interface.
    return _LlamaCppWrapper(model), None  # no tokenizer needed


class _LlamaCppWrapper:
    """Wraps llama.cpp's Llama so it mimics transformers' generate() interface."""

    def __init__(self, llama_model):
        self.model = llama_model

    def generate(self, input_ids, max_new_tokens, do_sample, temperature, pad_token_id):
        # Use the GGUF model's OWN tokenizer (llama.cpp), never a foreign one --
        # a mismatched vocab silently corrupts every prompt and completion.
        import torch
        ids = input_ids[0].tolist()
        prompt = self.model.detokenize(ids).decode("utf-8", errors="ignore")
        out = self.model(prompt, max_tokens=max_new_tokens, temperature=temperature,
                         echo=False, top_p=0.95)["choices"][0]["text"]
        # Re-encode with the same model tokenizer; return prompt+continuation so
        # the shape matches transformers' generate() contract.
        out_ids = self.model.tokenize(out.encode("utf-8"), add_bos=False)
        return torch.tensor([ids + out_ids])


# --------------------------------------------------------------------------- #
# Ollama serving proxy -- if ollama is installed, use its API directly
# --------------------------------------------------------------------------- #
def get_ollama_model_name(model_id: str) -> str | None:
    """If model_id is 'ollama://<name>', validate it exists and return the name."""
    if not model_id.startswith("ollama://"):
        return None
    model_name = model_id.replace("ollama://", "")
    if not have("ollama"):
        return None
    try:
        import ollama
        models = [m["name"].split(":")[0] for m in ollama.list().get("models", [])]
        if model_name in models or f"{model_name}:latest" in models:
            return model_name
    except Exception:
        pass
    return None
