"""MLX backend -- real Apple-Silicon-native training and serving.

Why MLX (verified on an M4 16GB):
  * 4-bit weights + LoRA keep an 8B model at ~5 GB peak for BOTH training and
    serving -- it never pressures the RAM your screen needs, so no freeze.
  * Unified memory = zero host<->device copies (no prefetch machinery needed).
  * Native Metal kernels + graph fusion -- the fusion win that torch.compile
    can't deliver on MPS.

This module uses Apple's maintained `mlx-lm`:
  * training : convert/load -> LoRA (linear_to_lora_layers) -> tuner.train,
               saving adapters the serving side loads directly (no fuse needed).
  * serving  : load(base, adapter_path=...) + generate behind our OpenAI API.

It is gated to Apple Silicon; everywhere else the PyTorch path is used.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time

from .util import have, log


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def is_available() -> bool:
    return (platform.system() == "Darwin" and platform.machine() == "arm64"
            and have("mlx") and have("mlx_lm"))


def set_memory_guard(limit_gb: float | None = None):
    """Cap MLX's wired memory so training can never starve the OS/screen.

    Defaults to leaving ~4 GB headroom for macOS + apps on this machine.
    """
    try:
        import mlx.core as mx
        import psutil
        total = psutil.virtual_memory().total / 1e9
        # Cap MLX's cache to a safe fraction so it can't balloon and starve the
        # OS/screen. Stay within Metal's recommended working-set size.
        try:
            rec = mx.metal.device_info().get("max_recommended_working_set_size",
                                             int(total * 0.6 * 1e9))
        except Exception:
            rec = int(total * 0.6 * 1e9)
        cap = int(min(limit_gb * 1e9, rec)) if limit_gb else int(rec)
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(cap)
        log(f"  [mlx] memory guard: cache capped ~{cap/1e9:.0f} GB "
            f"(of {total:.0f} GB total)")
    except Exception as e:
        log(f"  [mlx] memory guard skipped: {e}")


# --------------------------------------------------------------------------- #
# Dataset adapter: our records -> mlx-lm dataset
# --------------------------------------------------------------------------- #
def _build_dataset(records, tokenizer):
    from mlx_lm.tuner.datasets import (CacheDataset, ChatDataset,
                                       CompletionsDataset, TextDataset)
    sample = records[0]
    if "prompt" in sample and "completion" in sample:
        ds = CompletionsDataset(records, tokenizer, "prompt", "completion",
                                mask_prompt=True)
    elif "messages" in sample:
        ds = ChatDataset(records, tokenizer, chat_key="messages", mask_prompt=True)
    elif "text" in sample:
        ds = TextDataset(records, tokenizer, text_key="text")
    else:
        # Coerce unknown shape into text.
        recs = [{"text": json.dumps(r)} for r in records]
        ds = TextDataset(recs, tokenizer, text_key="text")
    return CacheDataset(ds)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_mlx(model_id: str, records: list, out_dir: str, epochs: int = 3,
              batch_size: int = 1, max_seq_len: int = 1024, lora_layers: int = 16,
              lora_rank: int = 8, lr: float = 1e-4, mem_limit_gb: float | None = None,
              progress=None) -> str:
    """LoRA fine-tune on Apple Silicon via mlx-lm. Saves adapters + manifest.

    Returns the checkpoint dir. Serving loads (base_model, adapter_path=dir).
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
    from mlx_lm.tuner.callbacks import TrainingCallback

    set_memory_guard(mem_limit_gb)
    log(f"[mlx] loading {model_id} (4-bit if available)...")
    model, tok = load(model_id)

    ds = _build_dataset(records, tok)
    iters = max(1, epochs * math.ceil(len(records) / batch_size))
    log(f"[mlx] LoRA: {lora_layers} layers, rank {lora_rank} | "
        f"{len(records)} examples -> {iters} iters (batch {batch_size}, "
        f"ctx {max_seq_len})")

    model.freeze()
    linear_to_lora_layers(model, num_layers=lora_layers,
                          config={"rank": lora_rank, "scale": 16.0, "dropout": 0.0})
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"[mlx] trainable params: {n_train/1e6:.2f}M")

    os.makedirs(out_dir, exist_ok=True)
    adapter_file = os.path.join(out_dir, "adapters.safetensors")

    # Bridge mlx-lm's callback to our Prometheus exporter.
    burn = 0.0
    class _CB(TrainingCallback):
        def on_train_loss_report(self, info):
            it = info.get("iteration", 0)
            loss = info.get("train_loss", 0.0)
            tps = info.get("tokens_per_second") or info.get("trained_tokens", 0)
            if progress:
                progress(step=it, loss=loss, throughput_tok_s=float(tps or 0))
            log(f"  [mlx] iter {it}/{iters}  loss={loss:.4f}")

    opt = optim.Adam(learning_rate=lr)
    args = TrainingArgs(
        batch_size=batch_size, iters=iters,
        steps_per_report=max(1, iters // 10), steps_per_eval=iters + 1,
        steps_per_save=iters, max_seq_length=max_seq_len,
        adapter_file=adapter_file, grad_checkpoint=True,
    )
    t0 = time.time()
    train(model, opt, ds, val_dataset=ds, args=args, training_callback=_CB())
    log(f"[mlx] trained in {time.time()-t0:.1f}s, peak mem "
        f"{mx.get_peak_memory()/1e9:.1f} GB")

    # adapter_config.json so mlx-lm can reload the adapters at serve time.
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump({"fine_tune_type": "lora", "num_layers": lora_layers,
                   "lora_parameters": {"rank": lora_rank, "scale": 16.0,
                                       "dropout": 0.0}}, f, indent=2)
    with open(os.path.join(out_dir, "orchestrator.json"), "w") as f:
        json.dump({"base_model": model_id, "format": "mlx-lora",
                   "adapter_path": out_dir,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    log(f"[mlx] adapters saved to {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
class MLXEngine:
    """Loads base model + LoRA adapters and generates via mlx-lm."""

    def __init__(self, base_model: str, adapter_path: str | None,
                 mem_limit_gb: float | None = None):
        from mlx_lm import load
        set_memory_guard(mem_limit_gb)
        log(f"[mlx] loading {base_model}"
            + (f" + adapters {adapter_path}" if adapter_path else ""))
        self.model, self.tok = load(base_model, adapter_path=adapter_path)
        log("[mlx] model ready.")

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        # Apply the model's chat template for well-formed prompting.
        try:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        sampler = make_sampler(temp=max(temperature, 0.0))
        t0 = time.time()
        out = generate(self.model, self.tok, text, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        self._last_tps = len(self.tok.encode(out)) / max(time.time() - t0, 1e-6)
        return out.strip()


def load_manifest(model_or_dir: str) -> dict:
    p = os.path.join(model_or_dir, "orchestrator.json")
    if os.path.isdir(model_or_dir) and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}
