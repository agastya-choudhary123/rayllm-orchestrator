"""Shared utilities: logging, capability detection, model presets.

Kept dependency-free so every other layer can import it safely.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import time

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  RayLLM-Orchestrator :: {title}\n{line}", flush=True)


# --------------------------------------------------------------------------- #
# Capability detection -- everything degrades gracefully
# --------------------------------------------------------------------------- #
def have(module: str) -> bool:
    """True if an importable module is installed (without importing it)."""
    return importlib.util.find_spec(module) is not None


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def gpu_count() -> int:
    """GPU count via torch if available, else 0. Never raises."""
    if not have("torch"):
        return 0
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def capabilities() -> dict:
    return {
        "torch": have("torch"),
        "ray": have("ray"),
        "vllm": have("vllm"),
        "transformers": have("transformers"),
        "prometheus": have("prometheus_client"),
        "streamlit": have("streamlit"),
        "gpus": gpu_count(),
        "numactl": have_cmd("numactl"),
        "taskset": have_cmd("taskset"),
    }


# --------------------------------------------------------------------------- #
# Model presets -- friendly aliases -> HF ids + approx parameter counts
# --------------------------------------------------------------------------- #
MODEL_PRESETS = {
    "phi-3":      {"hf": "microsoft/Phi-3-mini-4k-instruct", "params_b": 3.8},
    "llama-3-1b": {"hf": "meta-llama/Llama-3.2-1B",          "params_b": 1.2},
    "llama-3-3b": {"hf": "meta-llama/Llama-3.2-3B",          "params_b": 3.2},
    "qwen-1.5b":  {"hf": "Qwen/Qwen2.5-1.5B",                "params_b": 1.5},
    "gpt2":       {"hf": "gpt2",                             "params_b": 0.124},
}


def resolve_model(name: str) -> dict:
    """Map an alias or raw HF id/path to {hf, params_b}."""
    if name in MODEL_PRESETS:
        return dict(MODEL_PRESETS[name], alias=name)
    # Unknown id/path: guess params from the name, default 1.5B.
    guess = 1.5
    for tok, b in [("70b", 70), ("13b", 13), ("8b", 8), ("7b", 7),
                   ("3b", 3.2), ("1b", 1.2), ("0.5b", 0.5)]:
        if tok in name.lower():
            guess = b
            break
    return {"hf": name, "params_b": guess, "alias": os.path.basename(name.rstrip("/"))}
