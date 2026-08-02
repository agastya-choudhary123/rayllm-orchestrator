"""Build train/test splits for a downstream classification eval.

    python examples/eval/prepare_data.py emotion --out-dir /tmp/eval

Writes <task>_train.jsonl (prompt/completion, for `rayllm train`) and
<task>_test.jsonl (prompt/gold, for evaluate.py). Train and test share one
prompt template so fine-tuning teaches exactly the format the eval scores.

Test examples come from the dataset's official held-out split, so they are never
seen during training -- and they are separate from the validation split the
trainer carves out of the training data for its own early stopping.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

TASKS = {
    # 77 fine-grained banking intents. Labels are domain-specific, so even a
    # large model cannot guess the taxonomy -- there is headroom at every size.
    "banking77": {
        "hf_id": "mteb/banking77",
        "train_split": "train",
        "test_split": "test",
        "text_key": "text",
        # Label names live in a column rather than a fixed list.
        "label_text_key": "label_text",
        "labels": None,
        "template": ("Classify this banking customer message as one of the "
                     "following intents:\n{labels}\n\n"
                     "Message: {text}\n\n"
                     "Answer with one intent name exactly as written above."),
    },
    "emotion": {
        "hf_id": "dair-ai/emotion",
        "train_split": "train",
        "test_split": "test",
        "text_key": "text",
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "template": ("Classify the emotion expressed in this text as one of: "
                     "{labels}.\n\nText: {text}\n\nAnswer with one word."),
    },
    "sst2": {
        "hf_id": "stanfordnlp/sst2",
        "train_split": "train",
        # SST-2's test split has hidden labels; validation is the usable held-out set.
        "test_split": "validation",
        "text_key": "sentence",
        "labels": ["negative", "positive"],
        "template": ("Classify the sentiment of this movie review as positive "
                     "or negative.\n\nReview: {text}\n\nAnswer with one word."),
    },
}


def build(task: str, n_train: int, n_test: int, seed: int, out_dir: str) -> None:
    from datasets import load_dataset

    spec = TASKS[task]
    tmpl, key = spec["template"], spec["text_key"]

    train_raw = load_dataset(spec["hf_id"], split=spec["train_split"])
    test_raw = load_dataset(spec["hf_id"], split=spec["test_split"])

    # Two label conventions: an index into a fixed list, or a name in a column.
    text_key = spec.get("label_text_key")
    if text_key:
        labels = sorted(set(train_raw[text_key]))
        label_of = lambda row: row[text_key]          # noqa: E731
        label_list = "\n".join(f"- {l}" for l in labels)
    else:
        labels = spec["labels"]
        label_of = lambda row: labels[row["label"]]   # noqa: E731
        label_list = ", ".join(labels)

    rng = random.Random(seed)
    train_idx = rng.sample(range(len(train_raw)), min(n_train, len(train_raw)))
    test_idx = rng.sample(range(len(test_raw)), min(n_test, len(test_raw)))

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"{task}_train.jsonl")
    test_path = os.path.join(out_dir, f"{task}_test.jsonl")

    def prompt_for(row) -> str:
        return tmpl.format(labels=label_list, text=str(row[key]).strip())

    with open(train_path, "w") as f:
        for i in train_idx:
            row = train_raw[i]
            f.write(json.dumps({"prompt": prompt_for(row),
                                "completion": label_of(row)}) + "\n")

    with open(test_path, "w") as f:
        for i in test_idx:
            row = test_raw[i]
            f.write(json.dumps({"prompt": prompt_for(row),
                                "gold": label_of(row)}) + "\n")

    dist = Counter(label_of(test_raw[i]) for i in test_idx)
    majority = max(dist.values()) / len(test_idx)
    print(f"train: {len(train_idx)} examples -> {train_path}")
    print(f"test : {len(test_idx)} examples -> {test_path}")
    print(f"test distribution: {dict(dist)}")
    print(f"majority-class baseline: {majority:.1%}  "
          "(any honest result must beat this)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=".")
    a = p.parse_args()
    build(a.task, a.n_train, a.n_test, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
