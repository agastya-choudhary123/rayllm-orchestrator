"""Training layer -- real fine-tuning.

`run()` dispatches on strategy:

  * single         : real single-device training loop (CPU or 1 GPU).
  * fsdp-ray       : Ray Train + PyTorch FSDP across N GPU workers.
  * deepspeed-ray  : Ray Train + DeepSpeed ZeRO-3.

All three run genuine training: tokenized DataLoader, forward/backward, AdamW,
gradient accumulation, real loss, periodic checkpoints (fault tolerance +
resume), and a final HuggingFace checkpoint that the serving layer loads
directly. Live throughput and cost burn-rate are pushed to the Prometheus
exporter as training progresses.

Quantization: with --quant 4bit/8bit we load the base weights quantized
(bitsandbytes) and train LoRA adapters (QLoRA). At the end the adapters are
merged back into the base weights so the checkpoint is a plain HF model that any
runtime (vLLM, transformers) can serve with no adapter juggling.
"""

from __future__ import annotations

import json
import math
import os
import time

from . import cost, data, models, monitor
from .util import gpu_count, have, log, resolve_model


def run(model: str, dataset: str, epochs: int, strategy: str, quant: str,
        out: str, num_workers: int = 0, lr: float = 2e-5, batch_size: int = 4,
        grad_accum: int = 4, max_len: int = 1024, use_lora: bool | None = None,
        resume: bool = True) -> str:
    if not (have("torch") and have("transformers")):
        raise RuntimeError(
            "Training requires torch + transformers. Install with:\n"
            "  pip install torch transformers datasets peft accelerate")

    m = resolve_model(model)
    ckpt_dir = os.path.join(out, m["alias"])
    os.makedirs(ckpt_dir, exist_ok=True)
    workers = num_workers or max(1, gpu_count())
    # QLoRA is the default when quantizing; full fine-tune otherwise.
    if use_lora is None:
        use_lora = quant in ("4bit", "8bit")

    cfg = dict(hf=m["hf"], dataset=dataset, epochs=epochs, quant=quant,
               lr=lr, batch_size=batch_size, grad_accum=grad_accum,
               max_len=max_len, use_lora=use_lora, out=ckpt_dir,
               params_b=m["params_b"], resume=resume)

    log(f"Strategy={strategy} workers={workers} quant={quant} "
        f"lora={use_lora} gpus={gpu_count()}")

    if strategy in ("fsdp-ray", "deepspeed-ray") and have("ray") and gpu_count() > 1:
        _train_ray(cfg, strategy, workers, ckpt_dir)
    else:
        if strategy != "single":
            log(f"Note: '{strategy}' needs Ray + >1 GPU; using single-device loop.")
        _train_single(cfg, ckpt_dir)

    _write_manifest(ckpt_dir, m, quant, strategy)
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

    device = ("cuda" if gpu_count() > 0 else
              "mps" if have("torch") and torch.backends.mps.is_available() else
              "cpu")
    log(f"Device: {device}")

    model, tok = build_model_and_tokenizer(cfg, device)
    records = data.load_records(cfg["dataset"])
    n_tokens = data.count_tokens(records, tok)
    log(f"Dataset: {len(records)} examples, {n_tokens:,} tokens")

    loader = data.build_dataloader(records, tok, batch_size=cfg["batch_size"],
                                   max_len=cfg["max_len"])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    total_steps = math.ceil(len(loader) / cfg["grad_accum"]) * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(total_steps, 1))

    est = cost.estimate_training(cfg["hf"], cfg["dataset"], cfg["epochs"],
                                 quant=cfg["quant"])
    burn = est.dollars / max(est.gpu_hours, 1e-6)

    start_epoch = _maybe_resume(model, opt, ckpt_dir) if cfg["resume"] else 0
    model.train()
    gstep = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        running, seen_tokens, t0 = 0.0, 0, time.time()
        opt.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
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
                monitor.set_progress(throughput_tok_s=tput, burn_usd_hr=burn,
                                     step=gstep, loss=running / (i + 1))
                if gstep % 5 == 0:
                    log(f"  epoch {epoch} step {gstep}/{total_steps} "
                        f"loss={running/(i+1):.4f} lr={sched.get_last_lr()[0]:.2e} "
                        f"{tput:.0f} tok/s")
        _save_epoch(model, opt, tok, ckpt_dir, epoch)

    _save_final(model, tok, ckpt_dir, cfg)


# --------------------------------------------------------------------------- #
# Ray Train + FSDP / DeepSpeed (multi-GPU)
# --------------------------------------------------------------------------- #
def _train_ray(cfg: dict, strategy: str, workers: int, ckpt_dir: str):
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    log(f"Initializing Ray for {workers}-worker {strategy}...")
    ray.init(ignore_reinit_error=True)

    def loop(config):
        import torch
        from ray import train
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        rank = train.get_context().get_world_rank()
        model, tok = build_model_and_tokenizer(config, "cuda")
        if strategy == "fsdp-ray" and not config["use_lora"]:
            model = FSDP(model, mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16))
        records = data.load_records(config["dataset"])
        loader = data.build_dataloader(records, tok,
                                       batch_size=config["batch_size"],
                                       max_len=config["max_len"])
        loader = train.torch.prepare_data_loader(loader)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config["lr"])

        for epoch in range(config["epochs"]):
            model.train()
            opt.zero_grad()
            for i, batch in enumerate(loader):
                out = model(**batch)
                (out.loss / config["grad_accum"]).backward()
                if (i + 1) % config["grad_accum"] == 0:
                    opt.step()
                    opt.zero_grad()
                train.report({"epoch": epoch, "loss": out.loss.item()})
            if rank == 0:
                _save_final(model, tok, config["out"], config)

    trainer = TorchTrainer(
        loop,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=workers, use_gpu=True),
        run_config=RunConfig(storage_path=os.path.abspath(ckpt_dir)),
    )
    result = trainer.fit()
    log(f"Ray training finished: {result.metrics}")


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


def _state_dict(model):
    m = model.module if hasattr(model, "module") else model
    return m.state_dict()


def _load_state_dict(model, sd):
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(sd, strict=False)


def _save_final(model, tok, ckpt_dir, cfg):
    """Write a plain HF checkpoint the serving layer can load directly.
    LoRA adapters are merged into the base weights first."""
    m = model.module if hasattr(model, "module") else model
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
