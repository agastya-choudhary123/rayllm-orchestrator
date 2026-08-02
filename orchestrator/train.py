"""Training layer -- real fine-tuning.

Single-device training loop (CPU or 1 GPU): tokenized DataLoader, forward/backward,
AdamW, gradient accumulation, real loss, periodic checkpoints (fault tolerance +
resume), and a final HuggingFace checkpoint that the serving layer loads directly.
Throughput is logged as training progresses.

Quantization: with --quant 4bit/8bit we load the base weights quantized
(bitsandbytes) and train LoRA adapters (QLoRA). At the end the adapters are
merged back into the base weights so the checkpoint is a plain HF model that any
serving backend can load with no adapter juggling.
"""

from __future__ import annotations

import json
import math
import os
import time

from . import data, models
from .util import gpu_count, have, log, resolve_model


def run(model: str, dataset: str, epochs: int, strategy: str, quant: str,
        out: str, num_workers: int = 0, lr: float = 2e-5, batch_size: int = 4,
        grad_accum: int = 4, max_len: int = 1024, use_lora: bool | None = None,
        resume: bool = True, opt=None, max_examples: int | None = None) -> str:
    m = resolve_model(model)
    from . import backends_mlx, fast
    if opt is None:
        opt = fast.OptConfig()

    # Apple-Silicon-native path first: MLX doesn't need torch/transformers and is
    # the fastest, most memory-safe trainer on a Mac. If MLX can't load this
    # model (e.g. the repo ships only PyTorch .bin weights, no safetensors), fall
    # back to the torch path instead of crashing the user's run.
    ckpt_dir_mlx = os.path.join(out, m["alias"])
    if fast.pick_engine() == "mlx" and backends_mlx.is_available() and _mlx_ok(m["hf"]):
        os.makedirs(ckpt_dir_mlx, exist_ok=True)
        try:
            return _train_mlx(m, dataset, epochs, ckpt_dir_mlx, opt, max_len,
                              max_examples, lr)
        except Exception as e:
            if not (have("torch") and have("transformers")):
                raise RuntimeError(
                    f"MLX training failed for {m['hf']} ({e}) and torch is not "
                    "installed to fall back to.\n"
                    "  Install torch to serve models MLX can't load:\n"
                    "  pip install torch transformers datasets peft accelerate"
                ) from e
            log(f"[mlx] can't train {m['hf']} ({type(e).__name__}: {e}); "
                "falling back to the PyTorch path.")

    if not (have("torch") and have("transformers")):
        raise RuntimeError(
            "Training requires torch + transformers (or MLX on Apple Silicon).\n"
            "  pip install torch transformers datasets peft accelerate")
    ckpt_dir = os.path.join(out, m["alias"])
    os.makedirs(ckpt_dir, exist_ok=True)
    workers = num_workers or max(1, gpu_count())
    # QLoRA is the default when quantizing; else follow the fast-path LoRA flag.
    if use_lora is None:
        use_lora = quant in ("4bit", "8bit") or opt.lora

    cfg = dict(hf=m["hf"], dataset=dataset, epochs=epochs, quant=quant,
               lr=lr, batch_size=batch_size, grad_accum=grad_accum,
               max_len=max_len, use_lora=use_lora, out=ckpt_dir,
               params_b=m["params_b"], resume=resume, opt=opt,
               max_examples=max_examples)

    log(f"Strategy={strategy} workers={workers} quant={quant} "
        f"lora={use_lora} gpus={gpu_count()}")

    _train_single(cfg, ckpt_dir)

    _write_manifest(ckpt_dir, m, quant, strategy)
    return ckpt_dir


def _mlx_ok(model_id: str) -> bool:
    """True if we should use the MLX trainer for this model.

    Prefer already-4-bit MLX repos (memory-safe on 16 GB). Small models can be
    converted on the fly; large raw HF models are left to the torch path to
    avoid a full-precision conversion that could exhaust unified memory.
    """
    mid = model_id.lower()
    if "mlx" in mid:
        return True
    return resolve_model(model_id)["params_b"] <= 4


def _train_mlx(m, dataset, epochs, ckpt_dir, opt, max_len=1024, max_examples=None,
               lr: float = 2e-5):
    """Apple-Silicon LoRA fine-tune via mlx-lm (see backends_mlx.train_mlx)."""
    from . import backends_mlx, data
    records = data.load_records(dataset, max_examples=max_examples)
    log(f"[mlx] dataset: {len(records)} examples")

    backends_mlx.train_mlx(
        m["hf"], records, ckpt_dir, epochs=epochs,
        batch_size=1, max_seq_len=max_len, lr=lr)
    return ckpt_dir


# --------------------------------------------------------------------------- #
# Model construction (shared by single + Ray workers)
# --------------------------------------------------------------------------- #
def build_model_and_tokenizer(cfg: dict, device: str):
    import torch
    model, tok = models.load_for_training(cfg["hf"], quant=cfg["quant"], device=device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cfg["use_lora"]:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        # If model is quantized (4bit/8bit), prep it for LoRA.
        if cfg["quant"] in ("4bit", "8bit"):
            model = prepare_model_for_kbit_training(model)
        lcfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=_lora_targets(model),
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    # Move to device (quantized models are already on device).
    if cfg["quant"] not in ("4bit", "8bit"):
        model = model.to(device)
    model.config.use_cache = False
    return model, tok


def _lora_targets(model) -> list[str]:
    """Pick attention/proj linear names present in this architecture."""
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",  # llama/phi/qwen
                  "c_attn", "c_proj", "c_fc"}           # gpt2
    present = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            present.add(leaf)
    return sorted(present) or ["c_attn"]


# --------------------------------------------------------------------------- #
# Single-device real training loop
# --------------------------------------------------------------------------- #
def _train_single(cfg: dict, ckpt_dir: str):
    import torch
    from . import fast

    o = cfg["opt"]
    device = fast.pick_device()
    log(f"Device: {device}  | fast-path: {o.summary()}")

    model, tok = build_model_and_tokenizer(cfg, device)
    # Apply compile / grad-checkpoint / flash to the model.
    model = fast.optimize_model(model, o, device)

    records = data.load_records(cfg["dataset"], max_examples=cfg.get("max_examples"))
    n_tokens = data.count_tokens(records, tok)

    # Adaptive data strategy: analyze the length distribution and pick pack /
    # bucket / pad to minimize wasted padding-FLOPs. Generalizes to any dataset.
    loader, dinfo = data.build_adaptive_dataloader(
        records, tok, batch_size=cfg["batch_size"], max_len=cfg["max_len"],
        strategy=o.data_strategy)
    log(f"Dataset: {len(records)} examples, {n_tokens:,} tokens")
    log(f"Data strategy: {dinfo['strategy'].upper()} -- {dinfo.get('reason','')}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    total_steps = math.ceil(len(loader) / cfg["grad_accum"]) * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(total_steps, 1))


    start_epoch = _maybe_resume(model, opt, ckpt_dir) if cfg["resume"] else 0
    model.train()
    gstep = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        running, seen_tokens, t0 = 0.0, 0, time.time()
        opt.zero_grad()
        # Async prefetch overlaps H2D copy with compute (GPU never stalls).
        for i, batch in enumerate(fast.Prefetcher(loader, device, o.prefetch)):
            with fast.autocast_ctx(o, device):   # bf16 mixed precision
                out = model(**batch)
                loss = out.loss / cfg["grad_accum"]
            loss.backward()
            running += out.loss.item()
            seen_tokens += int(batch["attention_mask"].sum().item())

            if (i + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                dt = time.time() - t0
                tput = seen_tokens / dt if dt > 0 else 0
                if gstep % 5 == 0:
                    log(f"  epoch {epoch} step {gstep}/{total_steps} "
                        f"loss={running/(i+1):.4f} lr={sched.get_last_lr()[0]:.2e} "
                        f"{tput:.0f} tok/s")
        _save_epoch(model, opt, tok, ckpt_dir, epoch)

    _save_final(model, tok, ckpt_dir, cfg)


# --------------------------------------------------------------------------- #
# Ray Train + FSDP / DeepSpeed (multi-GPU)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Checkpointing / resume / final save
# --------------------------------------------------------------------------- #
def _save_epoch(model, opt, tok, ckpt_dir, epoch):
    import torch
    cp = os.path.join(ckpt_dir, "checkpoints", f"epoch-{epoch}")
    os.makedirs(cp, exist_ok=True)
    torch.save({"epoch": epoch,
                "model": _state_dict(model),
                "optim": opt.state_dict()},
               os.path.join(cp, "training_state.pt"))
    tok.save_pretrained(cp)
    log(f"  checkpoint saved: {cp}")


def _maybe_resume(model, opt, ckpt_dir) -> int:
    import torch
    root = os.path.join(ckpt_dir, "checkpoints")
    if not os.path.isdir(root):
        return 0
    epochs = sorted(int(d.split("-")[1]) for d in os.listdir(root)
                    if d.startswith("epoch-"))
    if not epochs:
        return 0
    last = epochs[-1]
    state = torch.load(
        os.path.join(root, f"epoch-{last}", "training_state.pt"),
        map_location="cpu", weights_only=False)
    _load_state_dict(model, state["model"])
    opt.load_state_dict(state["optim"])
    log(f"Resumed from epoch {last}; continuing at epoch {last + 1}.")
    return last + 1


def _unwrap(model):
    """Strip torch.compile (_orig_mod) and DDP/FSDP (.module) wrappers."""
    m = model
    m = getattr(m, "_orig_mod", m)          # torch.compile
    m = getattr(m, "module", m)             # DDP / FSDP
    m = getattr(m, "_orig_mod", m)          # compile-inside-wrapper
    return m


def _state_dict(model):
    return _unwrap(model).state_dict()


def _load_state_dict(model, sd):
    _unwrap(model).load_state_dict(sd, strict=False)


def _save_final(model, tok, ckpt_dir, cfg):
    """Write a plain HF checkpoint the serving layer can load directly.
    LoRA adapters are merged into the base weights first."""
    m = _unwrap(model)
    if cfg["use_lora"] and hasattr(m, "merge_and_unload"):
        log("Merging LoRA adapters into base weights...")
        m = m.merge_and_unload()
    m.config.use_cache = True
    m.save_pretrained(ckpt_dir, safe_serialization=True)
    tok.save_pretrained(ckpt_dir)
    log(f"Final model saved to {ckpt_dir}")


def _write_manifest(ckpt_dir, m, quant, strategy):
    manifest = {
        "base_model": m["hf"], "alias": m["alias"], "quant": quant,
        "strategy": strategy, "format": "hf-causal-lm",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(ckpt_dir, "orchestrator.json"), "w") as f:
        json.dump(manifest, f, indent=2)
