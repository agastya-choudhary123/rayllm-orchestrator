# RayLLM-Orchestrator container (GPU).
# CUDA runtime base so the same image trains and serves (vLLM) on a GPU node.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip numactl util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

EXPOSE 8000
# Default: print help. Override the command to train/serve.
ENTRYPOINT ["python3", "orchestrator.py"]
CMD ["--help"]
