#!/usr/bin/env python3
"""
RayLLM-Orchestrator CLI
=======================
One command to train a model, one command to serve it.

    rayllm train --model phi-3 --dataset my-data --epochs 3
    rayllm serve --model ./checkpoints/phi-3 --port 8000

Training handles: checkpointing, quantization (LoRA), and resume from checkpoints.
Serving scans your hardware and lets YOU choose the backend (MLX, transformers, ...)
-- pass --backend, or pick interactively. Run `rayllm backends` to see what's
available on this machine.
"""

import argparse
import sys

from . import serve, train
from .util import banner, log

BACKEND_CHOICES = ["ollama", "mlx", "llama_cpp", "transformers"]


# --------------------------------------------------------------------------- #
# Section 1. Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rayllm",
        description="Train and serve small foundation models with one command.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- run (the one command) ------------------------------------------
    r = sub.add_parser("run",
                       help="Fine-tune AND serve in one shot. The main command.")
    r.add_argument("--model", default="mlx-community/Qwen3-8B-4bit",
                   help="Model id/alias/path. Default: a 4-bit 8B that trains + "
                        "serves in ~5 GB on Apple Silicon.")
    r.add_argument("--data", "--dataset", dest="data", required=True,
                   help="Local .jsonl or HF dataset id.")
    r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--ctx", type=int, default=4096,
                   help="Context window (max sequence length). Higher fits long "
                        "examples without truncation; 8192 is the practical max "
                        "for a 16 GB Mac.")
    r.add_argument("--max-examples", type=int, default=None,
                   help="Cap dataset size (keeps laptop training bounded).")
    r.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    r.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate. Higher learns faster but can degrade a "
                        "model that already does the task well; lower is safer.")
    r.add_argument("--out", default="./checkpoints")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--no-fast", action="store_true",
                   help="Disable the fast-path optimization stack.")
    r.add_argument("--no-serve", action="store_true",
                   help="Train only; skip serving.")
    r.add_argument("--backend", default=None, choices=BACKEND_CHOICES,
                   help="Serving backend. If omitted, you'll pick from the "
                        "backends available on this machine.")

    # ---- train -----------------------------------------------------------
    t = sub.add_parser("train", help="Fine-tune / train a model.")
    t.add_argument("--model", required=True,
                   help="Model id or local path (e.g. phi-3, meta-llama/Llama-3.2-1B).")
    t.add_argument("--dataset", required=True,
                   help="HF dataset id or local jsonl path.")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--strategy", default="single",
                   choices=["single"],
                   help="Training backend (single-device).")
    t.add_argument("--quant", default="none",
                   choices=["none", "8bit", "4bit"],
                   help="Load base weights quantized to save VRAM (QLoRA-style).")
    t.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate. Higher learns faster but can degrade a "
                        "model that already does the task well; lower is safer.")
    t.add_argument("--out", default="./checkpoints",
                   help="Where to write checkpoints.")
    t.add_argument("--num-workers", type=int, default=0,
                   help="Ray training workers. 0 = auto-detect GPUs.")

    # ---- serve -----------------------------------------------------------
    s = sub.add_parser("serve", help="Serve a checkpoint with auto-selected backend.")
    s.add_argument("--model", required=True, help="Checkpoint path or model id.")
    s.add_argument("--quant", default="none", choices=["none", "8bit", "4bit", "awq", "gptq"])
    s.add_argument("--continuous-batching", action="store_true", default=True,
                   help="Enable micro-batching.")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--tensor-parallel", type=int, default=1,
                   help="GPUs to shard across.")
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--draft-model", default=None,
                   help="Small model for speculative decoding (MLX). Faster "
                        "generation, identical output. e.g. a 0.5B-1B.")
    s.add_argument("--backend", default=None, choices=BACKEND_CHOICES,
                   help="Serving backend. If omitted, you'll pick from the "
                        "backends available on this machine. Run `rayllm "
                        "backends` to see them first.")

    # ---- backends --------------------------------------------------------
    sub.add_parser("backends",
                   help="Scan this machine and list available serving backends.")

    return p


# --------------------------------------------------------------------------- #
# Backend selection helpers
# --------------------------------------------------------------------------- #
def _print_backends(rows: list[dict]) -> None:
    log("Available serving backends on this machine:")
    for i, r in enumerate(rows, 1):
        mark = "✓" if r["available"] else "✗"
        rec = "  <- recommended for this model" if r.get("recommended") else ""
        log(f"  {i}. [{mark}] {r['name']:14s} ({r['device']:14s}) "
            f"{r['note']}{rec}")


class BackendSelectionError(RuntimeError):
    """Raised when no serving backend could be selected."""


def _resolve_backend(requested: str | None, model: str) -> str:
    """Return the backend to serve with. Selection is always the user's.

    If `requested` is given (via --backend) it's validated against what's
    actually available here. Otherwise the scanned backends are listed and the
    user picks one interactively. There is no auto-selection.
    """
    from . import models

    rows = models.scan_backends(model)
    by_name = {r["name"]: r for r in rows}

    if requested:
        pick = by_name.get(requested)
        if pick and not pick["available"]:
            raise BackendSelectionError(
                f"Backend '{requested}' isn't available here: {pick['note']}")
        return requested

    _print_backends(rows)
    usable = [r for r in rows if r["available"]]
    if not usable:
        raise BackendSelectionError(
            "No serving backend is available on this machine. Install one "
            "(see notes above) and try again.")

    # RECOMMENDED: the format-aware pick from scan_backends, but only if it's
    # actually usable here; otherwise fall back to the fastest available one
    # (scan order is fastest -> most compatible).
    rec = next((r["name"] for r in usable if r.get("recommended")), None)
    if rec is None:
        rec = usable[0]["name"]

    if not sys.stdin.isatty():
        raise BackendSelectionError(
            "No --backend given and this isn't an interactive terminal. "
            f"Pass --backend (recommended: {rec}); available: "
            + ", ".join(r["name"] for r in usable))

    log(f"→ RECOMMENDED: {rec}  (press Enter to accept)")
    while True:
        try:
            raw = input(f"Choose a backend by number [Enter = {rec}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise BackendSelectionError("No backend selected.")
        if not raw:
            return rec
        try:
            pick = rows[int(raw) - 1]
        except (ValueError, IndexError):
            log("Not a valid choice; enter a listed number or Enter for the "
                "recommended one.")
            continue
        if not pick["available"]:
            log(f"'{pick['name']}' isn't available here ({pick['note']}). "
                "Pick an available one.")
            continue
        return pick["name"]


# --------------------------------------------------------------------------- #
# Section 2. Command handlers
# --------------------------------------------------------------------------- #
def cmd_run(a: argparse.Namespace) -> int:
    """The one command: fine-tune, then serve. Optimized by default."""
    from . import fast
    banner("RUN  (fine-tune + serve)")

    device, engine = fast.pick_device(), fast.pick_engine()
    cfg = fast.OptConfig() if not a.no_fast else fast.OptConfig(
        data_strategy="pad", bf16=False, grad_checkpoint=False,
        lora=False, prefetch=False, flash_attn=False)
    log(f"Hardware: device={device}  engine={engine}")
    log(f"Fast-path: {cfg.summary()}")

    ckpt = train.run(
        model=a.model, dataset=a.data, epochs=a.epochs,
        strategy="single", quant=a.quant, out=a.out, opt=cfg,
        max_len=a.ctx, max_examples=a.max_examples, lr=a.lr,
    )
    log(f"✓ Fine-tune complete: {ckpt}")

    if a.no_serve:
        log(f"Serve later with:  rayllm serve --model {ckpt}")
        return 0

    override = _resolve_backend(a.backend, ckpt)
    log(f"→ Serving on http://localhost:{a.port} (Ctrl-C to stop)")
    return serve.run(model=ckpt, quant=a.quant, port=a.port, host="0.0.0.0",
                     tensor_parallel=1, max_model_len=2048,
                     continuous_batching=True, backend_override=override)


def cmd_train(a: argparse.Namespace) -> int:
    banner("TRAIN")
    ckpt = train.run(
        model=a.model, dataset=a.dataset, epochs=a.epochs,
        strategy=a.strategy, quant=a.quant, out=a.out,
        num_workers=a.num_workers, lr=a.lr,
    )
    log(f"Training complete. Checkpoint: {ckpt}")
    log(f"Serve it with:  rayllm serve --model {ckpt}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    banner("SERVE")
    override = _resolve_backend(a.backend, a.model)
    return serve.run(
        model=a.model, quant=a.quant, port=a.port, host=a.host,
        tensor_parallel=a.tensor_parallel, max_model_len=a.max_model_len,
        continuous_batching=a.continuous_batching,
        draft_model=a.draft_model, backend_override=override,
    )


def cmd_backends(a: argparse.Namespace) -> int:
    from . import models
    banner("BACKENDS")
    _print_backends(models.scan_backends())
    return 0


# --------------------------------------------------------------------------- #
# Section 3. Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "run": cmd_run,
        "train": cmd_train,
        "serve": cmd_serve,
        "backends": cmd_backends,
    }[args.command]
    try:
        return handler(args)
    except BackendSelectionError as e:
        log(f"Backend selection failed: {e}")
        return 2
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
