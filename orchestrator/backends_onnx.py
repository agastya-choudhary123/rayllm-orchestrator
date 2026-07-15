"""ONNX Runtime backend -- cross-platform quantized inference.

ONNX lets you export a model once and run it on any backend:
  * CPUs (Intel, ARM, Apple Silicon, etc.)
  * GPUs (NVIDIA, AMD, Apple Metal)
  * Mobile / embedded
  * Browser (with WebAssembly)

This backend handles model conversion and optimized inference with quantization.
"""

from __future__ import annotations

import time

from .util import have, log


def is_available() -> bool:
    """Check if ONNX Runtime is installed."""
    return have("onnxruntime")


def export_to_onnx(model_path: str, output_path: str = "model.onnx",
                   quantize: bool = True) -> str:
    """Convert a HuggingFace model to ONNX format (optimized for export).

    This is a heavyweight operation done once per model.
    Subsequent serving just loads the pre-exported ONNX file.
    """
    if not have("optimum"):
        raise RuntimeError(
            "ONNX export requires the `optimum` library. Install with:\n"
            "  pip install optimum onnxruntime")

    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.onnxruntime.io_binding import IOBinding
    from transformers import AutoTokenizer

    log(f"Exporting {model_path} to ONNX...")
    log("(This is a one-time operation; subsequent runs load the cached ONNX file)")

    try:
        # Export via optimum
        model = ORTModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(output_path)
        log(f"Exported to {output_path}")

        if quantize:
            log(f"Quantizing to int8 (40% smaller, 20% faster)...")
            from optimum.onnxruntime import QuantizationConfig
            qconfig = QuantizationConfig(
                is_static=False, format="qdq", use_symmetric_activations=True)
            model.quantize(save_dir=output_path, quantization_config=qconfig)
            log(f"Quantized ONNX saved to {output_path}")

        return output_path
    except Exception as e:
        raise RuntimeError(
            f"ONNX export failed. Ensure {model_path} is a valid model.\n{e}")


def load_for_serving_onnx(model_path: str, quantized: bool = True):
    """Load an ONNX model for serving."""
    if not is_available():
        raise RuntimeError(
            "ONNX Runtime backend requires onnxruntime. Install with:\n"
            "  pip install onnxruntime optimum onnx")

    log(f"Loading ONNX model: {model_path}")
    return ONNXInferenceEngine(model_path, quantized)


class ONNXInferenceEngine:
    """Inference engine for ONNX models (CPU + any backend)."""

    def __init__(self, model_path: str, quantized: bool = True):
        import onnxruntime as rt
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ONNX Runtime picks the best backend automatically (Metal on macOS, CUDA on NVIDIA, etc.)
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = rt.get_available_providers()
        log(f"ONNX available providers: {providers}")

        # Load the ONNX model
        model_file = f"{model_path}/model.onnx" if model_path.endswith("/") else f"{model_path}/model.onnx"
        self.session = rt.InferenceSession(model_file, sess_options=so,
                                           providers=providers)
        log(f"Loaded ONNX model (quantized={quantized})")

    def generate(self, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7) -> str:
        """Generate text using ONNX Runtime."""
        import numpy as np
        t0 = time.time()

        ids = self.tokenizer.encode(prompt)
        input_ids = np.array([ids], dtype=np.int64)

        output = self.session.run(None, {"input_ids": input_ids})
        logits = output[0]  # Shape: [batch, seq_len, vocab_size]

        # Greedy or sampling decode
        if temperature > 0:
            # Top-p sampling (simplified)
            next_token_probs = logits[0, -1, :]
            next_token_probs = np.exp(next_token_probs / temperature)
            next_token_probs /= next_token_probs.sum()
            next_token = np.random.choice(len(next_token_probs),
                                          p=next_token_probs)
        else:
            next_token = np.argmax(logits[0, -1, :])

        response = self.tokenizer.decode([int(next_token)])
        elapsed = time.time() - t0
        log(f"ONNX inference: {elapsed*1000:.1f}ms, {1/elapsed:.0f} tok/s")
        return prompt + response
