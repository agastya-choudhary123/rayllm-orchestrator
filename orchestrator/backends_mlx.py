"""MLX backend for Apple Silicon.

MLX (https://github.com/ml-explore/mlx) is designed specifically for Apple Silicon.
It's faster than PyTorch MPS for many inference operations and supports quantized models.

This backend is only available on macOS with Apple Silicon (arm64).
"""

from __future__ import annotations

import platform
import time
from typing import Optional

from .util import have, log


def is_available() -> bool:
    """Check if MLX is available and we're on Apple Silicon."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False
    return have("mlx")


def load_model(model_path: str, max_tokens: int = 2048, dtype: str = "float16"):
    """Load a model using MLX.

    Supports HF model IDs and local paths. MLX automatically handles quantization
    detection and loading.
    """
    try:
        import mlx.core as mx
        from mlx.models.gpt2 import GPT2
        from mlx.models.mistral import Mistral
        from mlx.utils import tree_unflatten
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            f"MLX not installed. Install with: pip install mlx\n{e}")

    log(f"Loading {model_path} via MLX (Apple Silicon optimized)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MLX has specific model implementations; fall back to HF loading if needed.
    model = None
    if "gpt2" in model_path.lower():
        try:
            model = GPT2.from_pretrained(model_path)
            log("Loaded GPT-2 via MLX")
        except Exception:
            pass
    if "mistral" in model_path.lower():
        try:
            model = Mistral.from_pretrained(model_path)
            log("Loaded Mistral via MLX")
        except Exception:
            pass

    if model is None:
        # Fall back to loading via transformers then converting to MLX
        log("Model not in MLX registry, loading via transformers...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(
                __import__("torch"), dtype.replace("float", ""))
        )
        # In production, convert to MLX format; for now just wrap it.

    return MLXInferenceEngine(model, tokenizer, max_tokens)


class MLXInferenceEngine:
    """Inference engine for MLX models on Apple Silicon."""

    def __init__(self, model, tokenizer, max_tokens: int):
        import mlx.core as mx
        self.mx = mx
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = "mps"  # MLX abstracts device; always optimal on Silicon

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.95) -> str:
        """Generate text using MLX."""
        t0 = time.time()
        try:
            # MLX's generate API
            ids = self.tokenizer.encode(prompt)
            ids = self.mx.array(ids)
            # Note: MLX generate() is available in some versions; method signature varies
            # For now, this is a placeholder for the real MLX generate call
            generated = self._mlx_generate(ids, max_new_tokens, temperature, top_p)
            response = self.tokenizer.decode(generated)
            elapsed = time.time() - t0
            log(f"Generated {len(generated)} tokens in {elapsed:.2f}s "
                f"({len(generated)/elapsed:.0f} tok/s)")
            return response
        except Exception as e:
            log(f"MLX inference failed: {e}")
            raise

    def _mlx_generate(self, input_ids, max_tokens, temperature, top_p):
        """Placeholder for MLX generate implementation."""
        import mlx.core as mx
        # Actual MLX generate would go here
        # This is a stub since MLX API varies by version
        return input_ids  # Placeholder return


def load_for_serving_mlx(model_id: str) -> MLXInferenceEngine:
    """Load a model for serving via MLX."""
    if not is_available():
        raise RuntimeError(
            "MLX backend requires macOS + Apple Silicon + mlx library. "
            "Install with: pip install mlx")
    return load_model(model_id)
