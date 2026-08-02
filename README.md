# rayllm-orchestrator

Fine-tune a small LLM on your own data and serve it behind an OpenAI-compatible
API. Runs on a laptop, no GPU required.

```bash
pip install rayllm-orchestrator

rayllm train --model gpt2 --dataset my-data.jsonl --epochs 3
rayllm serve --model ./checkpoints/gpt2
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Training uses LoRA/QLoRA and writes a checkpoint after every epoch. On Apple
Silicon it runs through MLX, everywhere else through PyTorch. Serving works with
whichever backend you have installed: MLX, transformers, llama.cpp, or Ollama.

## Install

The base install pulls torch, transformers, datasets, peft, and accelerate.
Serving backends are separate, so you only download the one you need.

| Backend | Install | Runs on |
|---|---|---|
| transformers | included | CPU, Apple MPS, CUDA |
| MLX | `pip install 'rayllm-orchestrator[mlx]'` | Apple Silicon |
| llama.cpp | `pip install 'rayllm-orchestrator[llama-cpp]'` | CPU and Metal, GGUF models |
| Ollama | [Ollama](https://ollama.com) plus `pip install 'rayllm-orchestrator[ollama]'` | the local Ollama daemon |

From source:

```bash
git clone https://github.com/agastya-choudhary123/rayllm-orchestrator
cd rayllm-orchestrator
pip install -e .
```

## Preparing data

JSONL, one object per line. Any of these shapes work:

```json
{"text": "A plain training example."}
{"prompt": "Classify this review.", "completion": "positive"}
{"messages": [{"role": "user", "content": "..."}]}
```

Field names are matched loosely, so `instruction`/`output` and `question`/`answer`
work too. You can also pass a HuggingFace dataset id instead of a file:

```bash
rayllm train --model gpt2 --dataset wikitext --epochs 1
```

## Commands

**`train`** downloads the model, tokenizes, trains, and prints the checkpoint path.

```
rayllm train --model <id-or-path> --dataset <id-or-path.jsonl>
             [--epochs N] [--quant none|8bit|4bit] [--lr LR] [--out DIR]
```

Defaults are `--epochs 1`, `--quant none`, `--lr 2e-5`, `--out ./checkpoints`.
With `--quant`, base weights load in 4- or 8-bit and only LoRA adapters train.
The PyTorch path then merges those adapters back into a plain HuggingFace
checkpoint, so serving needs no special handling.

**`serve`** starts an OpenAI-compatible API on port 8000.

```
rayllm serve --model <checkpoint-or-id>
             [--backend ollama|mlx|llama_cpp|transformers]
             [--port 8000] [--host 0.0.0.0] [--max-model-len 4096]
             [--draft-model <id>]
```

Without `--backend` you get an interactive prompt listing what's available.

**`run`** does both in one shot.

```
rayllm run --model <id-or-path> --data <id-or-path.jsonl>
           [--epochs N] [--port 8000] [--backend ...] [--no-serve]
```

**`backends`** reports which backends this machine can actually use.

```
$ rayllm backends
1. [✗] ollama         needs the ollama daemon running
2. [✓] mlx            Apple-Silicon-native            <- recommended for this model
3. [✗] llama_cpp      `pip install llama-cpp-python`
4. [✓] transformers   universal fallback
```

Availability is probed live, so a backend is only marked usable if it's both
installed and runnable here.

## Speculative decoding

On the MLX backend, a small draft model can propose tokens for a larger model to
verify. Output is identical, generation is faster.

```bash
rayllm serve --model <large-mlx-model> --draft-model <small-mlx-model> --backend mlx
```

## Does fine-tuning help?

Only when the base model is bad at the task. Measured across the model-size ladder
with 500 training examples, 1 epoch, and 200 held-out test examples:

| Model | Emotion (6 labels) | Banking77 (77 intents) |
|---|:---:|:---:|
| 0.135B | 20.5% → **67.0%** | 2.5% → **27.0%** |
| 0.5B | 31.0% → **70.5%** | 2.5% → **57.5%** |
| 1.5B | 34.0% → **65.5%** | 29.0% → **64.5%** |
| 3B | 47.5% → **72.0%** | 49.0% → **70.5%** |
| 8B | 0% → **62.0%** | — |

Majority baselines are 37% for emotion and 3% for banking77, so every base model
except the 3B on banking77 starts at or below the score you'd get by guessing the
most common label.

The size of the gain tracks how bad the base model was, not how big it is. On
emotion the smallest model gains the most (+46.5 at 0.135B) and the strongest
base gains the least (+24.5 at 3B). Banking77 shows the same shape: +55.0 for the
0.5B starting from 2.5%, down to +21.5 for the 3B already at 49%. Run the
base-model eval first, so you know which case you're in.

The 8B base scored 0%, which means no response parsed to exactly one label.
That's a formatting failure rather than wrong answers, and it's what fine-tuning
fixes first.

Two things learned the hard way, both from an earlier single-task sweep on the
0.5B (measured at 74.5% on emotion under those settings, documented in full in
[examples/eval/](examples/eval/)):

**Learning rate matters most.** At `--lr 1e-4`, the old default, that run scored
51.0% on emotion instead of 74.5%, and 75.0% on SST-2 instead of 91.5%. The
SST-2 case is the alarming one: fine-tuning made a model 16.5 points worse at a
task it already handled. The default is now `2e-5`. If accuracy drops after
fine-tuning, lower `--lr` before changing anything else.

**Val loss is a training diagnostic, not a quality signal.** On SST-2 it ranked
the two runs backwards: the 75.0% run reported `val_loss=0.041`, better than the
91.5% run's `0.044`, and labelled both "healthy". Use an eval script with ground
truth labels.

## Performance

Serving throughput on one Apple Silicon Mac, MLX backend, 4-bit weights,
120-token outputs:

| Model | Throughput | Memory |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct-4bit | ~276 tok/s | 0.33 GB |
| Qwen3-0.6B-4bit | ~240 tok/s | 0.44 GB |
| Qwen3-8B-4bit | ~22 tok/s | 4.72 GB |

CPU serving runs closer to 20 tok/s. Your numbers will differ with model,
quantization, prompt length, and hardware, so benchmark your own setup.

## Troubleshooting

**`No module named 'torch'`**
Install the core stack: `pip install rayllm-orchestrator`.

**Out of memory while training**
Add `--quant 4bit` to load base weights quantized and train only adapters.

**"No serving backend is available"**
Run `rayllm backends` to see what's missing, then install one of them or pass
`--backend transformers`.

**Server doesn't respond**
Check it came up: `curl http://localhost:8000/v1/models`.

## Project layout

```
orchestrator/
├── cli.py           the rayllm command: train / serve / run / backends
├── train.py         training loop, MLX and torch dispatch
├── serve.py         OpenAI-compatible API, backend dispatch
├── models.py        model loading, backend detection
├── data.py          tokenization, packing and bucketing
├── fast.py          bf16, prefetch, packing, LoRA options
├── backends_mlx.py  Apple Silicon train and serve
└── util.py          logging, capability detection

examples/eval/       accuracy eval, base model vs fine-tuned
orchestrator.py      shim so `python orchestrator.py ...` still works
```

## License

MIT. Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
