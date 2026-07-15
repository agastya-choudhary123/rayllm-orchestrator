"""Data layer: real dataset loading, tokenization, and collation.

Supports:
  * local .jsonl  with {"prompt": ..., "completion": ...}  (instruction tuning)
  * local .jsonl  with {"text": ...}                        (plain LM)
  * a HuggingFace dataset id (uses the `datasets` library)

Produces a torch DataLoader of tokenized, padded batches with a causal-LM
`labels` field where prompt tokens are masked out (-100) so loss is computed
only on the completion -- the standard supervised fine-tuning setup.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(dataset: str, split: str = "train") -> list[dict]:
    """Return a list of {'prompt','completion'} or {'text'} records."""
    if os.path.isfile(dataset):
        return _read_jsonl(dataset)
    # Otherwise treat as a HuggingFace dataset id.
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    cols = ds.column_names
    out = []
    for r in ds:
        if "prompt" in cols and "completion" in cols:
            out.append({"prompt": r["prompt"], "completion": r["completion"]})
        elif "text" in cols:
            out.append({"text": r["text"]})
        elif "instruction" in cols and "output" in cols:
            out.append({"prompt": r["instruction"], "completion": r["output"]})
        else:  # fall back to the first text-ish column
            out.append({"text": str(r[cols[0]])})
    return out


def count_tokens(records: list[dict], tokenizer) -> int:
    """Exact token count over the rendered examples (used by the cost estimator)."""
    total = 0
    for r in records:
        text = _render(r)
        total += len(tokenizer(text, add_special_tokens=True)["input_ids"])
    return total


def _render(rec: dict) -> str:
    if "text" in rec:
        return rec["text"]
    return PROMPT_TEMPLATE.format(prompt=rec["prompt"]) + rec["completion"]


def build_dataloader(records: list[dict], tokenizer, batch_size: int = 4,
                     max_len: int = 1024, shuffle: bool = True):
    """Tokenize records and return a torch DataLoader with masked labels."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class SFTDataset(Dataset):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            rec = self.recs[i]
            if "text" in rec:
                ids = tokenizer(rec["text"], truncation=True,
                                max_length=max_len)["input_ids"]
                labels = list(ids)
            else:
                prompt = PROMPT_TEMPLATE.format(prompt=rec["prompt"])
                full = prompt + rec["completion"] + tokenizer.eos_token
                p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                ids = tokenizer(full, truncation=True, max_length=max_len,
                                add_special_tokens=True)["input_ids"]
                labels = list(ids)
                # Mask the prompt portion so loss is only on the response.
                for j in range(min(len(p_ids), len(labels))):
                    labels[j] = -100
            return {"input_ids": ids, "labels": labels}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return DataLoader(SFTDataset(records), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
