"""RayLLM-Orchestrator package.

Layers:
    util        -- logging, banners, capability detection
    data        -- dataset loading + tokenization (packing / bucketing)
    models      -- universal model loader + backend selection
    fast        -- fast-path optimizations (bf16, prefetch, packing, LoRA)
    train       -- training with Ray + FSDP and graceful local fallback
    serve       -- serving with auto-selected backend + continuous batching
    backends_mlx-- Apple Silicon native training + serving
    cli         -- command-line interface (the `rayllm` console script)
"""

__version__ = "0.1.0"
