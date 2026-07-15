"""Observability layer: Prometheus exporter + Streamlit dashboard.

Two pieces:
  * start_exporter()   -- runs an in-process Prometheus endpoint on :9090/metrics
                          and a background thread that samples GPU/CPU/mem and
                          updates gauges. Called automatically by train/serve.
  * launch_dashboard() -- runs the Streamlit app (dashboard/app.py) which reads
                          those metrics and shows GPU util, memory, throughput
                          and live cost burn-rate.

Metrics exported (all prefixed rayllm_):
  rayllm_gpu_util_percent{gpu="0"}
  rayllm_gpu_mem_used_gb{gpu="0"}
  rayllm_cpu_util_percent
  rayllm_throughput_tokens_per_s
  rayllm_cost_burn_usd_per_hour
  rayllm_step_total
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from .util import gpu_count, have, log

_STARTED = False
_METRICS = {}          # gauge registry, populated if prometheus_client present
_STATE = {"throughput": 0.0, "burn": 0.0, "step": 0, "loss": 0.0}


def set_progress(throughput_tok_s: float = None, burn_usd_hr: float = None,
                 step: int = None, loss: float = None) -> None:
    """Called by the train/serve loops to push live numbers into the exporter."""
    if throughput_tok_s is not None:
        _STATE["throughput"] = throughput_tok_s
    if burn_usd_hr is not None:
        _STATE["burn"] = burn_usd_hr
    if step is not None:
        _STATE["step"] = step
    if loss is not None:
        _STATE["loss"] = loss


def start_exporter(port: int = 9090) -> None:
    global _STARTED
    if _STARTED:
        return
    if not have("prometheus_client"):
        log("prometheus_client not installed -- metrics disabled (pip install prometheus_client).")
        return

    from prometheus_client import Gauge, start_http_server

    _METRICS["gpu_util"] = Gauge("rayllm_gpu_util_percent", "GPU utilization", ["gpu"])
    _METRICS["gpu_mem"] = Gauge("rayllm_gpu_mem_used_gb", "GPU memory used (GB)", ["gpu"])
    _METRICS["cpu"] = Gauge("rayllm_cpu_util_percent", "CPU utilization")
    _METRICS["tput"] = Gauge("rayllm_throughput_tokens_per_s", "Throughput tok/s")
    _METRICS["burn"] = Gauge("rayllm_cost_burn_usd_per_hour", "Cost burn rate $/hr")
    _METRICS["step"] = Gauge("rayllm_step_total", "Training/serving steps")
    _METRICS["loss"] = Gauge("rayllm_train_loss", "Training loss")

    try:
        start_http_server(port)
        _STARTED = True
        log(f"Prometheus exporter live at http://localhost:{port}/metrics")
        threading.Thread(target=_sample_loop, daemon=True).start()
    except OSError as e:
        log(f"Could not bind exporter on :{port} ({e}).")


def _sample_loop() -> None:
    while True:
        _sample_once()
        time.sleep(2)


def _sample_once() -> None:
    # GPU stats via pynvml if available, else torch, else nothing.
    for gpu, util, mem in _gpu_stats():
        _METRICS["gpu_util"].labels(gpu=str(gpu)).set(util)
        _METRICS["gpu_mem"].labels(gpu=str(gpu)).set(mem)
    _METRICS["cpu"].set(_cpu_util())
    _METRICS["tput"].set(_STATE["throughput"])
    _METRICS["burn"].set(_STATE["burn"])
    _METRICS["step"].set(_STATE["step"])
    _METRICS["loss"].set(_STATE["loss"])


def _gpu_stats():
    if have("pynvml"):
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
                yield i, util, mem
            return
        except Exception:
            pass
    if have("torch") and gpu_count() > 0:
        import torch
        for i in range(gpu_count()):
            yield i, 0.0, torch.cuda.memory_allocated(i) / 1e9


def _cpu_util() -> float:
    if have("psutil"):
        import psutil
        return psutil.cpu_percent()
    return 0.0


def launch_dashboard(port: int = 8501) -> int:
    if not have("streamlit"):
        log("streamlit not installed. Run: pip install streamlit")
        return 1
    import os
    app = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "app.py")
    log(f"Launching dashboard at http://localhost:{port}")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                            "--server.port", str(port)])
