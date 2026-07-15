#!/usr/bin/env python3
"""
RayLLM-Orchestrator
===================
One command to train a small foundation model, one command to serve it.

Everything in between -- checkpointing, sharding, quantization, loading the
result into a fast inference server, cost accounting and observability -- is
handled for you.

    python orchestrator.py train --model phi-3 --dataset my-data --epochs 3 \
        --strategy fsdp-ray --quant 4bit

    python orchestrator.py serve --model ./checkpoints/phi-3 --quant 4bit \
        --continuous-batching --port 8000

This file is deliberately a thin, readable CLI. The real work lives in small,
single-responsibility modules under ./orchestrator so each layer (train, serve,
monitor, kernel, networking, cost) can be read and tested on its own.

Design goal: this must *run* on a laptop with nothing installed and *scale* on a
4-GPU node. Heavy deps (ray, vllm, torch) are imported lazily and degrade
gracefully to a simulation mode so the control plane is always demoable.
"""

import argparse
import sys

from orchestrator import cost, kernel, monitor, networking, serve, train
from orchestrator.util import banner, log


# --------------------------------------------------------------------------- #
# Section 1. Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchestrator.py",
        description="Train and serve small foundation models with one command.",
    )
    sub = p.add_subparsers(dest="command", required=True)

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
    t.add_argument("--kernel-profile", default="default",
                   choices=["default", "low-latency"],
                   help="Apply CPU pinning / RT scheduling for the trainer.")
    t.add_argument("--dry-run", action="store_true",
                   help="Plan + cost estimate only, do not launch.")

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
    s.add_argument("--kernel-profile", default="default",
                   choices=["default", "low-latency"])
    s.add_argument("--backend", default=None,
                   choices=["ollama", "mlx", "llama_cpp", "vllm", "transformers", "webgpu"],
                   help="Override auto-selected backend.")
    s.add_argument("--webgpu", action="store_true",
                   help="Export for browser-based WebGPU inference (no server).")

    # ---- cost ------------------------------------------------------------
    c = sub.add_parser("cost", help="Estimate training/serving cost, no run.")
    c.add_argument("--model", required=True)
    c.add_argument("--dataset", default=None)
    c.add_argument("--epochs", type=int, default=1)
    c.add_argument("--gpu", default="A100-80GB")
    c.add_argument("--spot", action="store_true", help="Use spot pricing.")
    c.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])

    # ---- monitor ---------------------------------------------------------
    m = sub.add_parser("monitor", help="Launch the observability dashboard.")
    m.add_argument("--port", type=int, default=8501)

    # ---- benchmarks ------------------------------------------------------
    b = sub.add_parser("rdma-demo", help="Demo fast GPU-to-GPU / shared-mem transfer.")
    b.add_argument("--size-mb", type=int, default=256)

    return p


# --------------------------------------------------------------------------- #
# Section 2. Command handlers
# --------------------------------------------------------------------------- #
def cmd_train(a: argparse.Namespace) -> int:
    banner("TRAIN")
    # 1) Cost first -- never launch a run blind.
    est = cost.estimate_training(a.model, a.dataset, a.epochs, quant=a.quant)
    cost.print_estimate(est)
    if a.dry_run:
        log("Dry run: plan validated, nothing launched.")
        return 0

    # 2) Optional real-time / pinned CPU profile for the driver process.
    kernel.apply_profile(a.kernel_profile)

    # 3) Start the metrics exporter so the dashboard lights up live.
    monitor.start_exporter()

    # 4) Hand off to the training layer (Ray + FSDP/DeepSpeed or local).
    ckpt = train.run(
        model=a.model, dataset=a.dataset, epochs=a.epochs,
        strategy=a.strategy, quant=a.quant, out=a.out,
        num_workers=a.num_workers,
    )
    log(f"Training complete. Checkpoint: {ckpt}")
    log(f"Serve it with:  python orchestrator.py serve --model {ckpt} --quant 4bit")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    banner("SERVE")
    if a.webgpu:
        log("WebGPU mode: exporting model for browser inference...")
        from orchestrator import backends_webgpu
        return backends_webgpu.serve_webgpu(a.model)
    kernel.apply_profile(a.kernel_profile)
    monitor.start_exporter()
    return serve.run(
        model=a.model, quant=a.quant, port=a.port, host=a.host,
        tensor_parallel=a.tensor_parallel, max_model_len=a.max_model_len,
        continuous_batching=a.continuous_batching, backend_override=a.backend,
    )


def cmd_cost(a: argparse.Namespace) -> int:
    banner("COST ESTIMATE")
    est = cost.estimate_training(
        a.model, a.dataset, a.epochs, gpu=a.gpu, spot=a.spot, quant=a.quant)
    cost.print_estimate(est)
    return 0


def cmd_monitor(a: argparse.Namespace) -> int:
    banner("MONITOR")
    return monitor.launch_dashboard(port=a.port)


def cmd_rdma(a: argparse.Namespace) -> int:
    banner("RDMA / FAST-TRANSFER DEMO")
    networking.transfer_demo(size_mb=a.size_mb)
    return 0


# --------------------------------------------------------------------------- #
# Section 3. Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "train": cmd_train,
        "serve": cmd_serve,
        "cost": cmd_cost,
        "monitor": cmd_monitor,
        "rdma-demo": cmd_rdma,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
