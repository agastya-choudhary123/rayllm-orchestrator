"""Cost estimator.

Turns "how much will this run cost?" into a number *before* you burn GPUs.

Method:
  * Training FLOPs  ~= 6 * N_params * N_tokens   (Kaplan/Chinchilla rule of thumb;
    6 = 2 for forward + 4 for backward, per parameter per token).
  * Wall-clock       = FLOPs / (num_gpus * peak_flops * MFU).
  * Dollar cost      = gpu_hours * $/gpu-hour (on-demand or spot).
  * Quantization     = memory + a modest throughput multiplier (4bit lets you
                       use fewer/smaller GPUs, which is where the real savings
                       come from).

These are estimates, not billing. The point is to kill "black-box cost
surprises" -- you see the order of magnitude and the levers before you commit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .util import log, resolve_model

# Peak bf16/fp16 TFLOPs (dense, realistic) and rough cloud pricing ($/gpu-hr).
GPU_SPECS = {
    #                 tflops   on-demand   spot
    "A100-80GB":   {"tflops": 312, "od": 3.67, "spot": 1.10},
    "A100-40GB":   {"tflops": 312, "od": 2.90, "spot": 0.90},
    "H100-80GB":   {"tflops": 990, "od": 5.50, "spot": 2.20},
    "L4-24GB":     {"tflops": 121, "od": 0.80, "spot": 0.30},
    "T4-16GB":     {"tflops":  65, "od": 0.53, "spot": 0.18},
    "RTX4090":     {"tflops": 165, "od": 0.44, "spot": 0.34},
}

# Model Flops Utilization -- fraction of peak you actually sustain in practice.
MFU = 0.40
# 4bit/8bit base weights reduce memory and give a modest compute win.
QUANT_SPEEDUP = {"none": 1.0, "8bit": 1.15, "4bit": 1.30}
QUANT_MEM_FACTOR = {"none": 1.0, "8bit": 0.5, "4bit": 0.28}


@dataclass
class Estimate:
    model: str
    params_b: float
    tokens: int
    gpu: str
    num_gpus: int
    quant: str
    petaflops: float
    gpu_hours: float
    dollars: float
    dollars_no_quant: float
    vram_gb_per_gpu: float
    spot: bool


def _tokens_for(dataset: str | None, epochs: int) -> int:
    """Real token count when the dataset is a local jsonl we can scan; otherwise
    a realistic default so the estimator is always usable (e.g. for a bare
    `cost` query with no dataset)."""
    import os
    if dataset and os.path.isfile(dataset):
        try:
            from . import data
            recs = data.load_records(dataset)
            # Approximate tokens without loading a tokenizer: ~1 token / 4 chars.
            chars = sum(len(str(r.get("prompt", "")) + str(r.get("completion", ""))
                            + str(r.get("text", ""))) for r in recs)
            return max(1, chars // 4) * max(1, epochs)
        except Exception:
            pass
    per_epoch = 50_000_000  # ~50M tokens: a realistic small fine-tune set.
    return per_epoch * max(1, epochs)


def estimate_training(model: str, dataset: str | None, epochs: int,
                      gpu: str = "A100-80GB", spot: bool = False,
                      quant: str = "none", num_gpus: int | None = None) -> Estimate:
    m = resolve_model(model)
    params_b = m["params_b"]
    spec = GPU_SPECS.get(gpu, GPU_SPECS["A100-80GB"])
    tokens = _tokens_for(dataset, epochs)

    # VRAM: params + optimizer(Adam ~ 3x in fp32 states) + activations, adj by quant.
    vram_full = params_b * (2 + 12) * QUANT_MEM_FACTOR[quant] + 4  # GB, rough
    if num_gpus is None:
        num_gpus = max(1, int(vram_full // spec_mem(gpu)) + (1 if vram_full % spec_mem(gpu) else 0))

    flops = 6 * (params_b * 1e9) * tokens
    petaflops = flops / 1e15
    eff_tflops = spec["tflops"] * MFU * QUANT_SPEEDUP[quant]
    gpu_seconds = flops / (num_gpus * eff_tflops * 1e12)
    gpu_hours = gpu_seconds / 3600 * num_gpus  # billed per-GPU

    rate = spec["spot"] if spot else spec["od"]
    dollars = gpu_hours * rate

    # What it would have cost with no quantization (same GPU budget).
    gpu_hours_nq = (flops / (num_gpus * spec["tflops"] * MFU * 1e12)) / 3600 * num_gpus
    dollars_no_quant = gpu_hours_nq * rate

    return Estimate(
        model=m["hf"], params_b=params_b, tokens=tokens, gpu=gpu,
        num_gpus=num_gpus, quant=quant, petaflops=petaflops,
        gpu_hours=gpu_hours, dollars=dollars, dollars_no_quant=dollars_no_quant,
        vram_gb_per_gpu=vram_full / num_gpus, spot=spot,
    )


def spec_mem(gpu: str) -> float:
    """Parse the GB figure baked into the GPU name (e.g. A100-80GB -> 80)."""
    for part in gpu.replace("-", " ").split():
        if part.upper().endswith("GB"):
            try:
                return float(part[:-2])
            except ValueError:
                pass
    return 24.0


def print_estimate(e: Estimate) -> None:
    saved = e.dollars_no_quant - e.dollars
    pct = 100 * saved / e.dollars_no_quant if e.dollars_no_quant else 0
    log(f"Model            : {e.model}  (~{e.params_b:.1f}B params)")
    log(f"Tokens           : {e.tokens/1e6:.0f}M")
    log(f"Compute          : {e.petaflops:.1f} PFLOPs")
    log(f"Hardware         : {e.num_gpus} x {e.gpu}  (~{e.vram_gb_per_gpu:.0f} GB/GPU)")
    log(f"Quantization     : {e.quant}")
    log(f"GPU-hours        : {e.gpu_hours:.2f}")
    log(f"Pricing          : {'SPOT' if e.spot else 'on-demand'}")
    log(f"Estimated cost   : ${e.dollars:,.2f}")
    if e.quant != "none":
        log(f"Quant savings    : ${saved:,.2f}  ({pct:.0f}% vs fp16)")
