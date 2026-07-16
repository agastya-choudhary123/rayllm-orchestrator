#!/usr/bin/env python3
"""
RayLLM-Orchestrator CLI
=======================
One command to train a model, one command to serve it.

    rayllm train --model phi-3 --dataset my-data --epochs 3
    rayllm serve --model ./checkpoints/phi-3 --port 8000

Training handles: checkpointing, quantization (LoRA), multi-GPU scaling via Ray+FSDP.
Serving auto-selects the best backend for your hardware (MLX, vLLM, transformers).
"""

import argparse
import sys

from . import serve, train
from .util import banner, log


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
    r.add_argument("--out", default="./checkpoints")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--no-fast", action="store_true",
                   help="Disable the fast-path optimization stack.")
    r.add_argument("--no-serve", action="store_true",
                   help="Train only; skip serving.")

    # ---- train -----------------------------------------------------------
    t = sub.add_parser("train", help="Fine-tune / train a model.")
    t.add_argument("--model", required=True,
                   help="Model id or local path (e.g. phi-3, meta-llama/Llama-3.2-1B).")
    t.add_argument("--dataset", required=True,
                   help="HF dataset id or local jsonl path.")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--strategy", default="fsdp-ray",
                   choices=["single", "fsdp-ray", "deepspeed-ray"],
                   help="Distributed training backend.")
    t.add_argument("--quant", default="none",
                   choices=["none", "8bit", "4bit"],
                   help="Load base weights quantized to save VRAM (QLoRA-style).")
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

    return p


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
        max_len=a.ctx, max_examples=a.max_examples,
    )
    log(f"✓ Fine-tune complete: {ckpt}")

    if a.no_serve:
        log(f"Serve later with:  rayllm serve --model {ckpt}")
        return 0

    log(f"→ Serving on http://localhost:{a.port} (Ctrl-C to stop)")
    return serve.run(model=ckpt, quant=a.quant, port=a.port, host="0.0.0.0",
                     tensor_parallel=1, max_model_len=2048,
                     continuous_batching=True)


def cmd_train(a: argparse.Namespace) -> int:
    banner("TRAIN")
    ckpt = train.run(
        model=a.model, dataset=a.dataset, epochs=a.epochs,
        strategy=a.strategy, quant=a.quant, out=a.out,
        num_workers=a.num_workers,
    )
    log(f"Training complete. Checkpoint: {ckpt}")
    log(f"Serve it with:  rayllm serve --model {ckpt}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    banner("SERVE")
    return serve.run(
        model=a.model, quant=a.quant, port=a.port, host=a.host,
        tensor_parallel=a.tensor_parallel, max_model_len=a.max_model_len,
        continuous_batching=a.continuous_batching,
        draft_model=a.draft_model,
    )


# --------------------------------------------------------------------------- #
# Section 3. Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "run": cmd_run,
        "train": cmd_train,
        "serve": cmd_serve,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
