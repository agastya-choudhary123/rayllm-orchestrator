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
              val_fraction: float = 0.1, seed: int = 0, progress=None) -> str:
    """LoRA fine-tune on Apple Silicon via mlx-lm. Saves adapters + manifest.

    Uses a real held-out validation split (never the training data) so the
    reported val loss is a genuine overfitting signal. Returns the checkpoint
    dir; serving loads (base_model, adapter_path=dir).
    """
    import random

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
    from mlx_lm.tuner.callbacks import TrainingCallback

    set_memory_guard(mem_limit_gb)
    log(f"[mlx] loading {model_id} (4-bit if available)...")
    model, tok = load(model_id)

    # Safety net: at a high ctx this drops nothing, but it guarantees no example
    # is fully truncated into an all-masked batch (which yields NaN loss and
    # poisons the run). See data.filter_for_training.
    from . import data as _data
    records, fstats = _data.filter_for_training(records, tok, max_seq_len)
    if fstats["dropped_prompt_too_long"] or fstats["dropped_over_length"]:
        log(f"[mlx] filtered: kept {fstats['kept']}/{fstats['total']} "
            f"(dropped {fstats['dropped_prompt_too_long']} prompt>ctx, "
            f"{fstats['dropped_over_length']} over-length) at ctx={max_seq_len}")

    # Real train/val split -- shuffle deterministically, hold out val_fraction.
    recs = list(records)
    random.Random(seed).shuffle(recs)
    n_val = max(1, int(len(recs) * val_fraction)) if len(recs) > 10 else max(1, len(recs) // 5)
    val_recs, train_recs = recs[:n_val], recs[n_val:]
    train_ds, val_ds = _build_dataset(train_recs, tok), _build_dataset(val_recs, tok)

    iters = max(1, epochs * math.ceil(len(train_recs) / batch_size))
    log(f"[mlx] split: {len(train_recs)} train / {len(val_recs)} val (held out)")
    log(f"[mlx] LoRA: {lora_layers} layers, rank {lora_rank} | "
        f"{iters} iters (batch {batch_size}, ctx {max_seq_len})")
    ds = train_ds  # trained on the training split only

    model.freeze()
    linear_to_lora_layers(model, num_layers=lora_layers,
                          config={"rank": lora_rank, "scale": 16.0, "dropout": 0.0})
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"[mlx] trainable params: {n_train/1e6:.2f}M")

    os.makedirs(out_dir, exist_ok=True)
    adapter_file = os.path.join(out_dir, "adapters.safetensors")

    # Bridge mlx-lm's callbacks to our exporter + capture train/val loss history,
    # AND implement early stopping: whenever held-out val loss improves, snapshot
    # the current adapters as the ones we keep. This guarantees the saved model is
    # the best-generalizing checkpoint, never an overfit later one.
    history = {"train": [], "val": []}
    best = {"val": float("inf"), "iter": -1}

    def _save_adapters(path):
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(path, weights)

    class _CB(TrainingCallback):
        def on_train_loss_report(self, info):
            it = info.get("iteration", 0)
            loss = info.get("train_loss", 0.0)
            tps = info.get("tokens_per_second") or info.get("trained_tokens", 0)
            history["train"].append((it, loss))
            if progress:
                progress(step=it, loss=loss, throughput_tok_s=float(tps or 0))
            log(f"  [mlx] iter {it}/{iters}  train_loss={loss:.4f}")

        def on_val_loss_report(self, info):
            it = info.get("iteration", 0)
            vloss = info.get("val_loss", 0.0)
            history["val"].append((it, vloss))
            tag = ""
            if vloss < best["val"]:
                best["val"], best["iter"] = vloss, it
                _save_adapters(adapter_file)   # keep the best-val adapter
                tag = "  <- best, saved"
            log(f"  [mlx] iter {it}/{iters}  VAL_loss={vloss:.4f}  (held-out){tag}")

    opt = optim.Adam(learning_rate=lr)
    # Evaluate on held-out val often enough that early stopping has resolution.
    eval_every = max(1, iters // 6)
    args = TrainingArgs(
        batch_size=batch_size, iters=iters,
        steps_per_report=max(1, iters // 10), steps_per_eval=eval_every,
        val_batches=max(1, min(len(val_recs), 20)),
        steps_per_save=iters + 1,          # we control saving (best-val only)
        max_seq_length=max_seq_len,
        adapter_file=adapter_file, grad_checkpoint=True,
    )
    t0 = time.time()
    train(model, opt, ds, val_dataset=val_ds, args=args, training_callback=_CB())
    if best["iter"] < 0:                    # val never ran -> save final as fallback
        _save_adapters(adapter_file)
    log(f"[mlx] trained in {time.time()-t0:.1f}s, peak mem "
        f"{mx.get_peak_memory()/1e9:.1f} GB")
    log(f"[mlx] early stopping: kept best-val adapter from iter {best['iter']} "
        f"(val_loss={best['val']:.3f})")

    # Overfitting check on the checkpoint we ACTUALLY KEEP (best-val), comparing
    # its val loss to the train loss at that same point -- not a later overfit one.
    if history["val"]:
        # Train loss nearest the best-val iteration.
        bt = min(history["train"], key=lambda x: abs(x[0] - best["iter"]),
                 default=(0, float("nan")))[1] if history["train"] else float("nan")
        bv = best["val"]
        gap = bv - bt
        verdict = ("healthy" if gap < 0.5 else
                   "watch" if gap < 1.0 else "OVERFIT RISK")
        log(f"[mlx] KEPT checkpoint (iter {best['iter']}): train_loss={bt:.3f}  "
            f"val_loss={bv:.3f}  gap={gap:+.3f} -> {verdict}")
        history["verdict"] = verdict
        history["final_train"] = bt
        history["final_val"] = bv
        history["best_iter"] = best["iter"]

    # adapter_config.json so mlx-lm can reload the adapters at serve time.
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump({"fine_tune_type": "lora", "num_layers": lora_layers,
                   "lora_parameters": {"rank": lora_rank, "scale": 16.0,
                                       "dropout": 0.0}}, f, indent=2)
    with open(os.path.join(out_dir, "orchestrator.json"), "w") as f:
        json.dump({"base_model": model_id, "format": "mlx-lora",
                   "adapter_path": out_dir,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "n_train": len(train_recs), "n_val": len(val_recs),
                   "epochs": epochs, "lora_layers": lora_layers,
                   "lora_rank": lora_rank,
                   "final_train_loss": history.get("final_train"),
                   "final_val_loss": history.get("final_val"),
                   "overfit_verdict": history.get("verdict"),
                   "loss_history": history}, f, indent=2)
    log(f"[mlx] adapters saved to {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
class MLXEngine:
    """Loads a model (+ optional LoRA adapters) and generates via mlx-lm.

    Supports token streaming and speculative decoding (a small `draft_model`
    proposes tokens the main model verifies in batches -- typically 1.5-2.5x
    faster generation with identical output).
    """

    def __init__(self, base_model: str, adapter_path: str | None = None,
                 draft_model: str | None = None, mem_limit_gb: float | None = None):
        from mlx_lm import load
        set_memory_guard(mem_limit_gb)
        log(f"[mlx] loading {base_model}"
            + (f" + adapters {adapter_path}" if adapter_path else ""))
        self.model, self.tok = load(base_model, adapter_path=adapter_path)
        self.draft = None
        if draft_model:
            log(f"[mlx] loading draft model for speculative decoding: {draft_model}")
            self.draft, _ = load(draft_model)
        log("[mlx] model ready.")

    def _render(self, prompt: str) -> str:
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            return prompt

    def stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
        """Yield (text_chunk, response) as tokens are generated.

        Uses speculative decoding automatically when a draft model is loaded.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=max(temperature, 0.0))
        for r in stream_generate(self.model, self.tok, self._render(prompt),
                                 max_tokens=max_tokens, draft_model=self.draft,
                                 sampler=sampler):
            yield r.text, r

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Non-streaming: collect the full response. Records last-run stats."""
        chunks, last = [], None
        for text, r in self.stream(prompt, max_tokens, temperature):
            chunks.append(text)
            last = r
        if last is not None:
            self._last_tps = last.generation_tps
            self._last_accepted = getattr(last, "from_draft", None)
        return "".join(chunks).strip()


def benchmark_serving(models: list[str], prompt: str = "Explain what a stop-loss "
                      "order is in two sentences.", max_tokens: int = 80,
                      draft_for: dict | None = None,
                      delete_after: bool = True) -> list[dict]:
    """Benchmark generation tok/s across model sizes on this hardware.

    Memory hygiene (as requested): loads exactly ONE model at a time, measures,
    then FREES it (del + mx.clear_cache) before the next -- the 8B is never in
    memory while the 0.5B is benchmarked. Models are downloaded only when needed
    and (optionally) their local cache is deleted afterwards to reclaim disk.
    """
    import gc

    import mlx.core as mx
    from mlx_lm import load

    results = []
    for model_id in models:
        log(f"\n[bench] === {model_id} ===")
        had_cache = _is_cached(model_id)
        try:
            eng = MLXEngine(model_id,
                            draft_model=(draft_for or {}).get(model_id))
            # Warmup (compile/first-token cost excluded).
            for _ in eng.stream(prompt, max_tokens=4):
                pass
            t0 = time.time()
            n, last = 0, None
            for _text, r in eng.stream(prompt, max_tokens=max_tokens):
                n += 1
                last = r
            dt = time.time() - t0
            peak = mx.get_peak_memory() / 1e9
            tps = last.generation_tps if last else n / max(dt, 1e-6)
            results.append({"model": model_id, "tok_s": tps,
                            "peak_gb": peak, "tokens": n,
                            "speculative": eng.draft is not None})
            log(f"[bench] {model_id}: {tps:.1f} tok/s | peak {peak:.1f} GB")
        except Exception as e:
            log(f"[bench] {model_id} failed: {e}")
            results.append({"model": model_id, "error": str(e)[:100]})
        finally:
            # Free from memory BEFORE touching the next model.
            try:
                del eng
            except Exception:
                pass
            gc.collect()
            mx.clear_cache()
            log(f"[bench] freed {model_id} from memory (active "
                f"{mx.get_active_memory()/1e9:.1f} GB)")
            # Reclaim disk: only delete models we downloaded for the benchmark.
            if delete_after and not had_cache:
                _delete_cache(model_id)
    return results


def _hf_cache_dir(model_id: str) -> str:
    import os
    name = "models--" + model_id.replace("/", "--")
    return os.path.expanduser(f"~/.cache/huggingface/hub/{name}")


def _is_cached(model_id: str) -> bool:
    import os
    return os.path.isdir(_hf_cache_dir(model_id))


def _delete_cache(model_id: str):
    import shutil
    d = _hf_cache_dir(model_id)
    try:
        shutil.rmtree(d)
        log(f"[bench] deleted downloaded cache for {model_id} (reclaimed disk)")
    except Exception as e:
        log(f"[bench] could not delete cache for {model_id}: {e}")


def load_manifest(model_or_dir: str) -> dict:
    p = os.path.join(model_or_dir, "orchestrator.json")
    if os.path.isdir(model_or_dir) and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}
