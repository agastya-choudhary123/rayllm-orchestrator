"""Serving layer -- smart backend selection for any model format.

Detects model format (local/HF/GGUF/Ollama) and installed libraries, then picks
the best serving backend:

  * Ollama    : local model management + efficient serving (any quant)
  * llama.cpp : GGUF on CPU (memory efficient)
  * vLLM      : GPU + continuous batching (fastest)
  * transformers : fallback, CPU/GPU, any HF model

All backends expose the same OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid

from . import models
from .util import gpu_count, have, log


def _clamp_ctx(model: str, requested: int) -> int:
    """Clamp requested context length to the model's real max positions."""
    cfg_path = os.path.join(model, "config.json") if os.path.isdir(model) else None
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                c = json.load(f)
            real = c.get("max_position_embeddings") or c.get("n_positions")
            if real and requested > real:
                log(f"Clamping max_model_len {requested} -> {real} (model limit)")
                return int(real)
        except Exception:
            pass
    return requested


def _load_manifest(model: str) -> dict:
    path = os.path.join(model, "orchestrator.json")
    if os.path.isdir(model) and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"base_model": model, "quant": "none"}


def run(model: str, quant: str, port: int, host: str, tensor_parallel: int,
        max_model_len: int, continuous_batching: bool, backend_override: str = None,
        draft_model: str = None) -> int:
    manifest = _load_manifest(model)
    log(f"Serving {model} | manifest: base={manifest.get('base_model')} "
        f"quant={manifest.get('quant','none')}")

    # MLX-LoRA checkpoints (from the Apple-Silicon trainer) carry a manifest that
    # tells us the base model + adapter path -- serve them via MLX directly.
    if backend_override in (None, "mlx"):
        from . import backends_mlx
        mlx_manifest = backends_mlx.load_manifest(model)
        if backend_override == "mlx" or mlx_manifest.get("format") == "mlx-lora" \
                or ("mlx" in model.lower() and backends_mlx.is_available()):
            if backends_mlx.is_available():
                return _serve_mlx(model, port, host, mlx_manifest, draft_model)

    if backend_override:
        backend, spec = backend_override, models.ModelSpec(model)
        log(f"Backend override: {backend}")
    else:
        backend, spec = models.select_serving_backend(model)
        log(f"Selected backend: {backend}")

    if backend == "ollama":
        return _serve_ollama(model, port, host)
    elif backend == "mlx":
        return _serve_mlx(model, port, host)
    elif backend == "llama_cpp":
        return _serve_llama_cpp(model, port, host)
    elif backend == "vllm":
        return _serve_vllm(model, quant, port, host, tensor_parallel, max_model_len)
    elif backend == "transformers":
        return _serve_transformers(model, quant, port, host, max_model_len,
                                   continuous_batching)
    raise RuntimeError(f"Unknown backend: {backend}")


# --------------------------------------------------------------------------- #
# Ollama backend
# --------------------------------------------------------------------------- #
def _serve_ollama(model: str, port: int, host: str) -> int:
    """Proxy to Ollama's local HTTP API -- no need to reinvent serving."""
    import ollama
    import urllib.request
    import json as _json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model_name = model.replace("ollama://", "")
    log(f"Using Ollama backend for model: {model_name}")
    log(f"(Ollama handles model caching + quantization automatically)")

    class OllamaProxy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json_resp(self, code, obj):
            body = _json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json_resp(200, {"status": "ok", "backend": "ollama",
                                             "model": model_name})
            if self.path == "/v1/models":
                return self._json_resp(200, {"object": "list",
                                             "data": [{"id": model_name, "object": "model"}]})
            self._json_resp(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = _json.loads(self.rfile.read(n) or "{}")

            if self.path == "/v1/chat/completions":
                msgs = req.get("messages", [])
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                max_tokens = int(req.get("max_tokens", 128))
                temp = float(req.get("temperature", 0.7))
                resp = ollama.generate(model_name, prompt, stream=False,
                                       options={"temperature": temp,
                                               "num_predict": max_tokens})
                text = resp.get("response", "").strip()
                return self._json_resp(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"completion_tokens": resp.get("eval_count", 0)}
                })
            if self.path == "/v1/completions":
                prompt = req.get("prompt", "")
                max_tokens = int(req.get("max_tokens", 128))
                temp = float(req.get("temperature", 0.7))
                resp = ollama.generate(model_name, prompt, stream=False,
                                       options={"temperature": temp,
                                               "num_predict": max_tokens})
                text = resp.get("response", "").strip()
                return self._json_resp(200, {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion",
                    "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]
                })
            self._json_resp(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), OllamaProxy)
    log(f"Ollama proxy listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# MLX backend (Apple Silicon native)
# --------------------------------------------------------------------------- #
def _serve_mlx(model_path: str, port: int, host: str, manifest: dict = None,
               draft_model: str = None) -> int:
    """Serve via MLX (Apple Silicon native). Handles both a plain MLX model id
    and an MLX-LoRA checkpoint (base model + adapters) produced by our trainer.
    Supports streaming (SSE) and speculative decoding (draft_model)."""
    from . import backends_mlx
    manifest = manifest or {}
    if manifest.get("format") == "mlx-lora":
        base = manifest["base_model"]
        adapters = manifest.get("adapter_path", model_path)
    else:
        base, adapters = model_path, None
    engine = backends_mlx.MLXEngine(base, adapters, draft_model=draft_model)
    disp_model = manifest.get("base_model", model_path)

    # Single-threaded on purpose: MLX streams are thread-local, so generation
    # must run on the server's main thread. A local single-model server serves
    # requests sequentially anyway (the GPU serializes them).
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class MLXServer(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json_resp(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json_resp(200, {"status": "ok", "backend": "mlx",
                                             "device": "apple_silicon"})
            if self.path == "/v1/models":
                return self._json_resp(200, {"object": "list",
                                             "data": [{"id": model_path, "object": "model"}]})
            self._json_resp(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")

            if self.path == "/v1/chat/completions":
                msgs = req.get("messages", [])
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            elif self.path == "/v1/completions":
                prompt = req.get("prompt", "")
            else:
                return self._json_resp(404, {"error": "not found"})

            max_tokens = int(req.get("max_tokens", 128))
            temp = float(req.get("temperature", 0.7))
            is_chat = self.path == "/v1/chat/completions"

            # Streaming (OpenAI SSE): tokens are sent as they generate, so the
            # response feels instant even on a big model.
            if req.get("stream"):
                return self._stream(prompt, max_tokens, temp, is_chat)

            text = engine.generate(prompt, max_tokens, temp)
            if is_chat:
                return self._json_resp(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion", "created": int(time.time()),
                    "model": model_path,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"completion_tokens": len(text.split())}
                })
            return self._json_resp(200, {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion",
                "created": int(time.time()), "model": model_path,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]
            })

        def _stream(self, prompt, max_tokens, temp, is_chat):
            cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            obj = "chat.completion.chunk" if is_chat else "text_completion"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta, finish=None):
                if is_chat:
                    choice = {"index": 0, "delta": ({} if finish else
                              {"role": "assistant", "content": delta}),
                              "finish_reason": finish}
                else:
                    choice = {"index": 0, "text": delta, "finish_reason": finish}
                payload = {"id": cid, "object": obj, "created": int(time.time()),
                           "model": model_path, "choices": [choice]}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            try:
                for chunk, _r in engine.stream(prompt, max_tokens, temp):
                    if chunk:
                        send(chunk)
                send("", finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected mid-stream

    srv = HTTPServer((host, port), MLXServer)
    log(f"MLX server on http://{host}:{port} (Apple Silicon native)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# llama.cpp (GGUF) backend
# --------------------------------------------------------------------------- #
def _serve_llama_cpp(model_path: str, port: int, host: str) -> int:
    """Serve a GGUF via llama.cpp."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from llama_cpp import Llama

    log(f"Loading GGUF with llama.cpp: {model_path}")
    llama = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)
    log(f"Model loaded. CPU+GPU inference ready.")

    class LlamaCppServer(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json_resp(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json_resp(200, {"status": "ok", "backend": "llama.cpp"})
            if self.path == "/v1/models":
                return self._json_resp(200, {"object": "list",
                                             "data": [{"id": model_path, "object": "model"}]})
            self._json_resp(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")

            if self.path == "/v1/chat/completions":
                msgs = req.get("messages", [])
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            elif self.path == "/v1/completions":
                prompt = req.get("prompt", "")
            else:
                return self._json_resp(404, {"error": "not found"})

            max_tokens = int(req.get("max_tokens", 128))
            temp = float(req.get("temperature", 0.7))
            t0 = time.time()
            out = llama(prompt, max_tokens=max_tokens, temperature=temp,
                       echo=False, top_p=0.95)
            text = out["choices"][0]["text"].strip()
            toks = out.get("usage", {}).get("completion_tokens", 0)
            dt = time.time() - t0

            if self.path == "/v1/chat/completions":
                return self._json_resp(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion", "created": int(time.time()),
                    "model": model_path,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"completion_tokens": toks}
                })
            return self._json_resp(200, {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion",
                "created": int(time.time()), "model": model_path,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]
            })

    srv = ThreadingHTTPServer((host, port), LlamaCppServer)
    log(f"llama.cpp server on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


# --------------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------------- #
def _serve_vllm(model, quant, port, host, tp, max_model_len) -> int:
    import subprocess
    import sys
    # Clamp requested context to the model's real max positions so vLLM never
    # errors out (e.g. gpt2 is 1024). Reads config.json if present.
    max_model_len = _clamp_ctx(model, max_model_len)
    qflag = []
    if quant == "4bit":
        qflag = ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
    elif quant == "awq":
        qflag = ["--quantization", "awq"]
    elif quant == "gptq":
        qflag = ["--quantization", "gptq"]
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--host", host, "--port", str(port),
           "--tensor-parallel-size", str(tp),
           "--max-model-len", str(max_model_len), *qflag]
    log("Exec: " + " ".join(cmd))
    return subprocess.call(cmd)


# --------------------------------------------------------------------------- #
# transformers backend with real dynamic micro-batching
# --------------------------------------------------------------------------- #
class _BatchEngine:
    def __init__(self, model_path, device, batching=True, window=0.03, max_batch=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        log(f"Loading model on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype).to(device).eval()
        self.batching = batching
        self.window = window
        self.max_batch = max_batch
        self.q: queue.Queue = queue.Queue()
        self._served = 0
        threading.Thread(target=self._worker, daemon=True).start()
        log("Model ready.")

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        ev = threading.Event()
        slot = {"prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "event": ev, "out": None}
        self.q.put(slot)
        ev.wait()
        return slot["out"]

    def _worker(self):
        while True:
            first = self.q.get()
            batch = [first]
            deadline = time.time() + (self.window if self.batching else 0)
            while self.batching and len(batch) < self.max_batch:
                timeout = max(0, deadline - time.time())
                if timeout <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=timeout))
                except queue.Empty:
                    break
            self._run_batch(batch)

    def _run_batch(self, batch):
        torch = self.torch
        prompts = [b["prompt"] for b in batch]
        max_new = max(b["max_tokens"] for b in batch)
        temp = batch[0]["temperature"]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new,
                do_sample=temp > 0, temperature=max(temp, 1e-5),
                pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        new_tokens = int((gen != self.tok.pad_token_id).sum().item())
        dt = time.time() - t0
        self._served += len(batch)
        for b, text in zip(batch, texts):
            b["out"] = text.strip()
            b["event"].set()


def _serve_transformers(model, quant, port, host, max_model_len, batching) -> int:
    import torch
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    engine = _BatchEngine(model, device, batching=batching)

    def render_chat(messages):
        if len(messages) == 1 and messages[0].get("role") == "user":
            return f"### Instruction:\n{messages[0]['content']}\n\n### Response:\n"
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts) + "\nassistant:"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok", "backend": "transformers",
                                        "device": device})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list",
                                        "data": [{"id": model, "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")
            max_tokens = int(req.get("max_tokens", 128))
            temp = float(req.get("temperature", 0.7))
            if self.path == "/v1/chat/completions":
                prompt = render_chat(req.get("messages", []))
                text = engine.generate(prompt, max_tokens, temp)
                return self._json(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion", "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"completion_tokens": len(text.split())}
                })
            if self.path == "/v1/completions":
                prompt = req.get("prompt", "")
                text = engine.generate(prompt, max_tokens, temp)
                return self._json(200, {
                    "id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]
                })
            self._json(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), Handler)
    log(f"OpenAI-compatible server on http://{host}:{port} "
        f"(batching={'on' if batching else 'off'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0
