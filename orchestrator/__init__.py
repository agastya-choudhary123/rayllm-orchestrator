"""RayLLM-Orchestrator package.

Layers:
    util        -- logging, banners, capability detection
    cost        -- FLOPs + cloud pricing + quantization savings estimator
    kernel      -- CPU pinning, cgroups, real-time scheduling (low-latency profile)
    networking  -- fast GPU-to-GPU / shared-memory transfer demo
    monitor     -- Prometheus exporter + Streamlit dashboard
    train       -- Ray + FSDP/DeepSpeed training with graceful local fallback
    serve       -- vLLM serving with continuous batching + fallback
"""

__version__ = "0.1.0"
