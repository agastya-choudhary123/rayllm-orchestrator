"""Streamlit observability dashboard.

Scrapes the Prometheus exporter (http://localhost:9090/metrics) that train/serve
start automatically, and shows GPU utilization, memory, throughput and live cost
burn-rate. Auto-refreshes every 2s.

Run:  python orchestrator.py monitor      (or)   streamlit run dashboard/app.py
"""

import re
import time
from urllib.request import urlopen

import streamlit as st

METRICS_URL = "http://localhost:9090/metrics"

st.set_page_config(page_title="RayLLM-Orchestrator", layout="wide")
st.title("RayLLM-Orchestrator — Live Observability")


def scrape():
    out = {}
    try:
        text = urlopen(METRICS_URL, timeout=1).read().decode()
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"(\w+)(\{[^}]*\})?\s+([-\d.eE+]+)", line)
        if m:
            out.setdefault(m.group(1), []).append((m.group(2) or "", float(m.group(3))))
    return out


def first(metrics, name, default=0.0):
    vals = metrics.get(name) if metrics else None
    return vals[0][1] if vals else default


placeholder = st.empty()

while True:
    metrics = scrape()
    with placeholder.container():
        if metrics is None:
            st.warning("No exporter found on :9090. Start a train/serve run first.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Throughput", f"{first(metrics,'rayllm_throughput_tokens_per_s')/1e3:.1f}k tok/s")
            c2.metric("Cost burn", f"${first(metrics,'rayllm_cost_burn_usd_per_hour'):.2f}/hr")
            c3.metric("Step", int(first(metrics, "rayllm_step_total")))
            c4.metric("Loss", f"{first(metrics,'rayllm_train_loss'):.3f}")
            c5.metric("CPU", f"{first(metrics,'rayllm_cpu_util_percent'):.0f}%")

            st.subheader("GPUs")
            gutil = metrics.get("rayllm_gpu_util_percent", [])
            gmem = metrics.get("rayllm_gpu_mem_used_gb", [])
            if gutil:
                for i, (lbl, v) in enumerate(gutil):
                    mem = gmem[i][1] if i < len(gmem) else 0.0
                    st.write(f"GPU {i}: util {v:.0f}%  |  mem {mem:.1f} GB")
                    st.progress(min(v / 100, 1.0))
            else:
                st.info("No GPUs reported (running on CPU / simulation).")
        st.caption(f"Updated {time.strftime('%H:%M:%S')} — refreshes every 2s")
    time.sleep(2)
