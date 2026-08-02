# Downstream task eval

Training loss won't tell you whether fine-tuning worked. It can fall while task
accuracy drops. These two scripts measure accuracy on held-out examples the model
never saw.

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

Three tasks are available: `emotion` (6 labels, `dair-ai/emotion`), `sst2`
(2 labels, SST-2), and `banking77` (77 fine-grained intents, `mteb/banking77`).
Split sizes default to 500 train and 200 test, adjustable with `--n-train` and
`--n-test`. Requires MLX, matching the trainer's Apple Silicon path.

## Method

Test examples come from each dataset's official held-out split, so they're never
trained on. They're also separate from the validation split the trainer carves
out of the training data for early stopping.

Train and test share one prompt template, so fine-tuning teaches exactly the
format the eval scores.

Decoding is greedy (temperature 0), so runs reproduce.

Scoring is exact match. A response counts as correct only if exactly one label
appears in it. Zero or several means the model didn't answer, and scores wrong.

Every result prints next to the majority-class baseline, the score you'd get by
always guessing the most common label. A result below that line is worse than not
thinking at all.

## Results

This is the single-task sweep on one model, kept because it's the run behind the
learning-rate and val-loss findings below. The full model-size ladder across
emotion and banking77 is in the [top-level README](../../README.md); the 0.5B
scores 70.5% on emotion there, under that run's settings rather than these.

`Qwen2.5-0.5B-Instruct-4bit`, 500 training examples, 1 epoch, 200 test examples,
Apple Silicon with 17 GB. Training took 53s at 0.6 GB peak. Eval takes about 22s.

| Task | Majority baseline | Base model | Fine-tuned |
|------|------------------:|-----------:|-----------:|
| emotion (6 labels) | 37.0% | 31.0% | **74.5%** |
| sst2 (2 labels) | 54.0% | 91.5% | 91.5% |

Emotion is where fine-tuning earns its keep. The base model scores below the
majority baseline, dumps 112 of 200 answers into "sadness", and returns 10
responses naming no label at all. Afterward, predictions track the true label
distribution (joy 79 / sadness 75 / anger 23 against a true 74 / 54 / 27) and
every response parses.

SST-2 shows the limit. The base model is already at 91.5%, so there's no headroom
and fine-tuning can only hold the line. Fine-tuning helps when the model is bad at
the task, not automatically. Run the base-model eval first: if the score is
already high, fine-tuning probably isn't worth it.

## Learning rate matters more than anything else

Same data, same settings, only `--lr` changed:

| Task | `--lr 1e-4` | `--lr 2e-5` (default) |
|------|------------:|----------------------:|
| emotion | 51.0% | **74.5%** |
| sst2 | 75.0% | **91.5%** |

At `1e-4`, fine-tuning destroyed 16.5 points on SST-2, leaving the model worse
than if it hadn't been trained at all. If your eval score drops after
fine-tuning, lower `--lr` before trying anything else.

## The trainer's "healthy" verdict is not a quality score

The `val_loss=... -> healthy` line is a training diagnostic, not an accuracy
measure. On SST-2 it ranked the two runs backwards: the 75.0% run reported
`val_loss=0.041`, better than the 91.5% run's `0.044`, and it labelled both
"healthy". Use these scripts to judge quality.
