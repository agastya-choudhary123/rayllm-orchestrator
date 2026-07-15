# RayLLM-Orchestrator container.
# Base is CUDA runtime so the same image runs training and vLLM serving on a
# GPU node. On a CPU-only host it still runs the control plane (sim/stub mode).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip numactl util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install the light control plane by default; heavy stack is opt-in at build
# time with:  docker build --build-arg FULL=1 .
ARG FULL=0
RUN pip3 install --upgrade pip && \
    if [ "$FULL" = "1" ]; then pip3 install -r requirements.txt; \
    else pip3 install prometheus_client streamlit psutil; fi

COPY . .

EXPOSE 8000 8501 9090
# Default: print help. Override the command to train/serve/monitor.
ENTRYPOINT ["python3", "orchestrator.py"]
CMD ["--help"]
