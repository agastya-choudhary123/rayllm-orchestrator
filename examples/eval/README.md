# Downstream task eval

Does fine-tuning actually make the model better? Training loss won't tell you —
it can fall while task accuracy drops. These two scripts measure accuracy on
held-out examples the model never saw.

```bash
# 1. Build train/test splits (test comes from the dataset's official held-out split)
python examples/eval/prepare_data.py emotion --out-dir /tmp/eval

# 2. Score the base model, before any fine-tuning
python examples/eval/evaluate.py /tmp/eval/emotion_test.jsonl \
  mlx-community/Qwen2.5-0.5B-Instruct-4bit

# 3. Fine-tune on the training half
rayllm train --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --dataset /tmp/eval/emotion_train.jsonl --epochs 1 --out ./checkpoints

# 4. Score the fine-tuned model on the same test set
python examples/eval/evaluate.py /tmp/eval/emotion_test.jsonl \
  mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter ./checkpoints/Qwen2.5-0.5B-Instruct-4bit
```

Tasks: `emotion` (6 labels, `dair-ai/emotion`) and `sst2` (2 labels, SST-2).
Requires MLX (Apple Silicon), matching the trainer's MLX path.

## Method

- Test examples come from each dataset's official held-out split, so they are
  never trained on — and they are separate from the validation split the trainer
  carves out of the training data for early stopping.
- Train and test share one prompt template, so fine-tuning teaches exactly the
  format the eval scores.
- Greedy decoding (temperature 0), so runs are reproducible.
- Exact match. A response counts as correct only if exactly one label appears in
  it; zero or several means the model didn't answer, scored wrong.
- Every result is printed next to the majority-class baseline — the score you'd
  get by always guessing the most common label. A result below that line is
  worse than not thinking at all.

## Results on this repo

`Qwen2.5-0.5B-Instruct-4bit`, 500 training examples, 1 epoch, 200 test examples,
Apple Silicon (17 GB). Training took 53s at 0.6 GB peak; eval takes ~22s.

| Task | Majority baseline | Base model | Fine-tuned |
|------|------------------:|-----------:|-----------:|
| emotion (6 labels) | 37.0% | 31.0% | **74.5%** |
| sst2 (2 labels) | 54.0% | 91.5% | 91.5% |

Emotion is where fine-tuning earns its keep: the base model scores *below* the
majority baseline and dumps 112 of 200 answers into "sadness", plus 10 responses
that name no label at all. After fine-tuning, predictions track the true label
distribution (joy 79 / sadness 75 / anger 23 vs. true 74 / 54 / 27) and every
response is parseable.

SST-2 shows the honest limit: the base model is already at 91.5%, so there is no
headroom and fine-tuning can only hold the line. **Fine-tuning helps when the
model is bad at the task, not automatically.** Run the base-model eval first —
if the score is already high, fine-tuning is unlikely to be worth it.

## Learning rate matters more than anything else here

Same data, same everything, only `--lr` changed:

| Task | `--lr 1e-4` | `--lr 2e-5` (default) |
|------|------------:|----------------------:|
| emotion | 51.0% | **74.5%** |
| sst2 | 75.0% | **91.5%** |

At `1e-4` fine-tuning actively destroyed 16.5 points on SST-2 — worse than not
fine-tuning at all. If your eval score drops after fine-tuning, lower `--lr`
first.

## The trainer's "healthy" verdict is not a quality score

The `val_loss=... -> healthy` line is a training diagnostic, not an accuracy
measure. On SST-2 it ranked the two runs backwards: the 75.0% run reported
`val_loss=0.041`, *better* than the 91.5% run's `0.044`, and both were labelled
"healthy". Use these scripts to judge quality.
