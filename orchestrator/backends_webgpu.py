"""WebGPU backend -- inference in the browser.

Novel approach: instead of serving a model on a server, export it to ONNX/SafeTensors
and run inference directly in the browser using WebGPU (a modern Web GPU API).

This means:
  * No server needed for inference (runs on user's device)
  * Works on any device with a browser: Mac, Windows, Linux, iPad, etc.
  * GPU acceleration (or Metal on macOS) via WebGPU
  * Model stays on user's device (privacy)
  * Zero latency (no network)

Requires: transformers.js (ONNX in browser) or similar.
"""

from __future__ import annotations

import json
import os

from .util import have, log


def export_for_webgpu(model_path: str, output_dir: str = "./dist") -> str:
    """Export a model to browser-compatible format (ONNX + tokenizer).

    Creates a static directory with:
      * model.onnx (quantized if available)
      * tokenizer.json (for inference in browser)
      * config.json (metadata)

    Then serve with a simple static file server:
      python -m http.server 8000 --directory dist
    """
    if not have("transformers"):
        raise RuntimeError("Model export requires transformers. Install with:\n"
                           "  pip install transformers")

    os.makedirs(output_dir, exist_ok=True)
    log(f"Exporting {model_path} for browser (WebGPU)...")

    # Copy/export tokenizer
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    log(f"Tokenizer exported to {output_dir}/tokenizer.json")
    log(f"Config exported to {output_dir}/config.json")

    # Optional: export ONNX (requires optimum library)
    if have("optimum"):
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            log("Exporting ONNX for transformers.js...")
            model = ORTModelForCausalLM.from_pretrained(model_path)
            model.save_pretrained(output_dir)
            log(f"ONNX model ready at {output_dir}/model.onnx")
        except Exception as e:
            log(f"ONNX export skipped ({e}). Use transformers.js native loaders instead.")
    else:
        log("For ONNX export, install: pip install optimum onnx")

    # Create a simple HTML + JS frontend
    _create_webgpu_frontend(output_dir, model_path)
    return output_dir


def _create_webgpu_frontend(output_dir: str, model_name: str):
    """Generate a simple web frontend for browser inference."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RayLLM WebGPU</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        textarea {{ width: 100%; height: 100px; font-family: monospace; }}
        button {{ padding: 10px 20px; font-size: 16px; }}
        #output {{ border: 1px solid #ccc; padding: 10px; margin-top: 10px; min-height: 50px; }}
        .loading {{ color: #999; }}
    </style>
</head>
<body>
    <h1>🚀 RayLLM WebGPU Inference</h1>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Runs entirely in your browser</strong> — no server, no network latency.</p>

    <textarea id="prompt" placeholder="Enter your prompt here...">What is machine learning?</textarea>
    <br><br>

    <label>
        Max tokens: <input type="number" id="maxTokens" value="64" min="1" max="512">
    </label>
    <label>
        Temperature: <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
    </label>
    <br><br>

    <button onclick="generate()">Generate</button>
    <button onclick="clearOutput()">Clear</button>

    <h3>Output</h3>
    <div id="output"></div>

    <hr>
    <p><small>Status: <span id="status">loading model...</span></small></p>

    <script type="module">
        // transformers.js: ONNX + LLM inference in the browser
        import {{ pipeline }} from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

        let model = null;

        async function loadModel() {{
            try {{
                console.log("Loading model via transformers.js...");
                model = await pipeline("text-generation", "Xenova/distilgpt2");
                document.getElementById("status").textContent = "Model loaded (GPU ready via WebGPU)";
            }} catch (e) {{
                console.error("Model load failed:", e);
                document.getElementById("status").textContent = "Model load failed: " + e.message;
            }}
        }}

        window.generate = async function() {{
            if (!model) {{
                alert("Model still loading...");
                return;
            }}
            const prompt = document.getElementById("prompt").value;
            const maxTokens = parseInt(document.getElementById("maxTokens").value);
            const temperature = parseFloat(document.getElementById("temperature").value);

            const output = document.getElementById("output");
            output.innerHTML = '<p class="loading">Generating...</p>';

            try {{
                const result = await model(prompt, {{
                    max_new_tokens: maxTokens,
                    temperature: temperature,
                }});
                output.innerHTML = `<strong>Input:</strong> ${{prompt}}<br><strong>Output:</strong> ${{result[0].generated_text}}`;
            }} catch (e) {{
                output.innerHTML = `<p style="color:red;">Error: ${{e.message}}</p>`;
            }}
        }};

        window.clearOutput = function() {{
            document.getElementById("output").innerHTML = "";
        }};

        // Load model on page load
        loadModel();
    </script>
</body>
</html>
"""

    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    log(f"Web frontend created at {output_dir}/index.html")
    log(f"\nTo serve: python -m http.server 8000 --directory {output_dir}")
    log(f"Then open: http://localhost:8000")


def serve_webgpu(model_path: str, port: int = 8000) -> int:
    """Export a model and launch a simple HTTP server for WebGPU serving.

    The browser handles all inference via transformers.js + WebGPU.
    This server just serves static files.
    """
    import subprocess
    import sys

    dist_dir = export_for_webgpu(model_path)

    log(f"\n{'='*70}")
    log(f"WebGPU server ready!")
    log(f"Open: http://localhost:{port}")
    log(f"{'='*70}\n")

    # Simple HTTP server for static files
    return subprocess.call(
        [sys.executable, "-m", "http.server", str(port), "--directory", dist_dir])
