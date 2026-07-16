# Quick Start Guide

Get up and running in 5 minutes.

## 1. Install

```bash
pip install rayllm-orchestrator
```

## 2. Prepare Data

Create a file `my-data.jsonl`:
```json
{"text": "The future of AI is here."}
{"text": "Machine learning makes predictions."}
{"text": "Neural networks learn patterns."}
```

(At least 10-100 examples for meaningful training)

## 3. Train

```bash
rayllm train --model gpt2 --dataset my-data.jsonl --epochs 1
```

**Output:**
```
✓ Training complete. Checkpoint: ./checkpoints/gpt2
Serve it with: rayllm serve --model ./checkpoints/gpt2
```

## 4. Serve

In a new terminal:
```bash
rayllm serve --model ./checkpoints/gpt2 --port 8000
```

## 5. Use It

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a sentence about AI"}
    ],
    "max_tokens": 50
  }'
```

**Response:**
```json
{
  "choices": [
    {
      "message": {
        "content": "The future of AI is bright and full of possibilities."
      }
    }
  ]
}
```

---

## Next Steps

### Use a Different Model
```bash
# Llama 3.2 (1B, fast)
rayllm train --model meta-llama/Llama-3.2-1B \
  --dataset my-data.jsonl --epochs 1

# Llama 3.2 (8B, better quality)
rayllm train --model meta-llama/Llama-3.2-8B \
  --dataset my-data.jsonl --epochs 1 --quant 4bit
```

### Save Memory (Quantization)
```bash
rayllm train --model meta-llama/Llama-3.2-8B \
  --dataset my-data.jsonl --quant 4bit
```

Uses 4-bit quantization (trains only 1-2% of parameters). 60% less VRAM.

### Use More Data
```bash
# Train for more epochs
rayllm train --model gpt2 --dataset my-data.jsonl --epochs 5

# Or use a HuggingFace dataset
rayllm train --model gpt2 --dataset wikitext --epochs 1
```

### Multi-GPU Training
```bash
rayllm train --model meta-llama/Llama-3.2-8B \
  --dataset my-data.jsonl --strategy fsdp-ray --num-workers 4
```

Uses 4 GPUs with distributed training.

---

## Troubleshooting

**"No module named torch"**
```bash
pip install torch transformers
```

**Training is slow**
- Use quantization: `--quant 4bit`
- Use a smaller model: `--model gpt2`

**Out of memory**
```bash
rayllm train --model gpt2 --dataset my-data.jsonl --quant 4bit
```

**Can't serve locally**
```bash
# Make sure training finished (check ./checkpoints/gpt2 exists)
ls ./checkpoints/gpt2/
# Should show: config.json, pytorch_model.bin, etc.
```

---

## Full Command Reference

See `rayllm --help` for all options.

```bash
rayllm train --help
rayllm serve --help
```

---

## What's Next?

- Read the [README](README.md) for full documentation
- Check [BACKENDS.md](BACKENDS.md) for hardware-specific tips
- Review [DATASETS.md](DATASETS.md) for data format details
