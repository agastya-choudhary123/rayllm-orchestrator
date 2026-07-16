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


def load_records(dataset: str, split: str = "train",
                 max_examples: int | None = None) -> list[dict]:
    """Return a list of {'prompt','completion'} or {'text'} records.

    Accepts a local .jsonl path or a HuggingFace dataset id. Understands the
    common instruction-tuning schemas: prompt/completion, instruction/output,
    instruction/response (+optional context, e.g. Databricks Dolly), plain text.
    `max_examples` caps how many rows are loaded (keeps training time/memory
    bounded on a laptop).
    """
    if os.path.isfile(dataset):
        rows = _read_jsonl(dataset)
        recs = [_normalize_record(r) for r in rows]
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        recs = [_normalize_record(dict(r)) for r in ds]
    recs = [r for r in recs if r]  # drop unusable rows
    if max_examples:
        recs = recs[:max_examples]
    return recs


def _normalize_record(r: dict) -> dict | None:
    """Map a raw row from any common schema into {prompt,completion} or {text}."""
    def g(*keys):
        for k in keys:
            if k in r and r[k] not in (None, ""):
                return str(r[k])
        return None

    prompt = g("prompt", "instruction", "question", "input")
    completion = g("completion", "response", "output", "answer")
    context = g("context", "input") if prompt else None
    # Avoid using the same field as both prompt and context.
    if context and context == prompt:
        context = None

    if prompt and completion:
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        return {"prompt": prompt, "completion": completion}
    text = g("text", "content")
    if text:
        return {"text": text}
    return None


def filter_for_training(records: list[dict], tokenizer, max_len: int,
                        min_completion_tokens: int = 4) -> tuple[list[dict], dict]:
    """Drop examples that would train badly, and report what/why.

    The critical case: if an example's *prompt* alone is >= max_len, truncation
    leaves zero completion tokens -> the loss is computed over an empty set ->
    NaN, which poisons the whole run. We also drop examples so long that fewer
    than `min_completion_tokens` of the response would survive truncation. This
    is what makes training on real, variable-length datasets robust.
    """
    def n_tokens(text: str) -> int:
        # Works for both HF tokenizers and MLX's TokenizerWrapper.
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
        return len(tokenizer(text)["input_ids"])

    kept, dropped_masked, dropped_long = [], 0, 0
    for r in records:
        if "text" in r:
            if n_tokens(r["text"]) < 2:
                dropped_long += 1
                continue
            kept.append(r)
            continue
        prompt = PROMPT_TEMPLATE.format(prompt=r["prompt"])
        p = n_tokens(prompt)
        c = n_tokens(r["completion"])
        if p >= max_len - min_completion_tokens:
            dropped_masked += 1        # prompt eats the whole window -> NaN risk
            continue
        if p + c > max_len:
            dropped_long += 1          # completion would be partly truncated
            continue
        kept.append(r)
    stats = {"kept": len(kept), "dropped_prompt_too_long": dropped_masked,
             "dropped_over_length": dropped_long, "total": len(records)}
    return kept, stats


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


def _tokenize_records(records: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenize all records once, returning input_ids + masked labels per example."""
    out = []
    for rec in records:
        if "text" in rec:
            ids = tokenizer(rec["text"], truncation=True, max_length=max_len)["input_ids"]
            labels = list(ids)
        else:
            prompt = PROMPT_TEMPLATE.format(prompt=rec["prompt"])
            full = prompt + rec["completion"] + tokenizer.eos_token
            p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            ids = tokenizer(full, truncation=True, max_length=max_len,
                            add_special_tokens=True)["input_ids"]
            labels = list(ids)
            for j in range(min(len(p_ids), len(labels))):
                labels[j] = -100
        out.append({"input_ids": ids, "labels": labels})
    return out


def build_packed_dataloader(records: list[dict], tokenizer, batch_size: int = 4,
                            max_len: int = 1024, shuffle: bool = True):
    """Sequence packing: concatenate short examples into dense `max_len` blocks.

    Why this matters for THIS project: we fine-tune on short instruction/response
    examples. With naive padding, a batch's compute is dominated by pad tokens —
    e.g. 30-token examples padded to 1024 waste ~97% of every FLOP. Packing
    stitches many examples end-to-end into full-length sequences so (almost)
    every token does useful work. Loss is still masked per-example (prompt tokens
    and cross-example boundaries are ignored), so gradients are identical in
    expectation — you just stop paying for padding.

    Result: for short-example datasets this is a 3-10x throughput win, and it's
    entirely project-specific (general LM training already uses full-length text).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    toks = _tokenize_records(records, tokenizer, max_len)

    # Greedily pack tokenized examples into blocks of exactly `max_len`.
    # We also record each block's per-example segment lengths so we can reset
    # position_ids per example (correctness: each example's positional encoding
    # restarts at 0 instead of continuing across the block boundary).
    packed_ids: list[list[int]] = []
    packed_labels: list[list[int]] = []
    packed_pos: list[list[int]] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_pos: list[int] = []
    for t in toks:
        ids, labels = t["input_ids"], t["labels"]
        if len(ids) >= max_len:
            packed_ids.append(ids[:max_len])
            packed_labels.append(labels[:max_len])
            packed_pos.append(list(range(max_len)))
            continue
        if len(cur_ids) + len(ids) > max_len:
            packed_ids.append(cur_ids)
            packed_labels.append(cur_labels)
            packed_pos.append(cur_pos)
            cur_ids, cur_labels, cur_pos = [], [], []
        cur_ids.extend(ids)
        cur_labels.extend(labels)
        cur_pos.extend(range(len(ids)))       # position_ids restart per example
    if cur_ids:
        packed_ids.append(cur_ids)
        packed_labels.append(cur_labels)
        packed_pos.append(cur_pos)

    class PackedDataset(Dataset):
        def __len__(self):
            return len(packed_ids)

        def __getitem__(self, i):
            return {"input_ids": packed_ids[i], "labels": packed_labels[i],
                    "position_ids": packed_pos[i]}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn, pos = [], [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
            pos.append(b["position_ids"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "position_ids": torch.tensor(pos, dtype=torch.long),
        }

    return DataLoader(PackedDataset(), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)


def packing_stats(records: list[dict], tokenizer, max_len: int = 1024) -> dict:
    """Report how much compute packing saves for this dataset."""
    toks = _tokenize_records(records, tokenizer, max_len)
    real = sum(len(t["input_ids"]) for t in toks)
    n = len(toks)
    # Naive padding cost: every example padded to the batch max (approx max_len).
    naive_padded = n * max_len
    # Packed cost: ceil(real / max_len) blocks of max_len.
    import math
    packed = math.ceil(real / max_len) * max_len
    return {
        "examples": n,
        "real_tokens": real,
        "naive_padded_tokens": naive_padded,
        "packed_tokens": packed,
        "waste_naive_pct": 100 * (1 - real / naive_padded) if naive_padded else 0,
        "waste_packed_pct": 100 * (1 - real / packed) if packed else 0,
        "speedup_vs_naive": naive_padded / packed if packed else 1,
    }


# --------------------------------------------------------------------------- #
# Adaptive data engine -- generalize the padding-waste win to ANY dataset
# --------------------------------------------------------------------------- #
def analyze_lengths(records: list[dict], tokenizer, max_len: int = 1024,
                    batch_size: int = 4) -> dict:
    """Measure the length distribution and compute wasted-FLOP fraction under
    each batching strategy, then recommend the one that minimizes waste.

    First principles: training compute is proportional to *tokens processed*,
    including padding you compute and discard. 'Useful fraction' = real tokens /
    tokens actually pushed through the model. We estimate it for three strategies
    and pick the best:

      * naive   : pad every example to the global max (worst).
      * bucket  : sort by length, pad each batch to its own max (safe, general).
      * pack    : concatenate examples into dense max_len blocks (best when short,
                  no-op when already full-length).

    Returns the recommendation plus the numbers behind it.
    """
    import math
    toks = _tokenize_records(records, tokenizer, max_len)
    lengths = sorted(len(t["input_ids"]) for t in toks)
    n = len(lengths)
    real = sum(lengths)
    if n == 0:
        return {"strategy": "bucket", "reason": "empty dataset"}

    # naive: n * global_max
    naive = n * max(lengths)
    # bucket: sort by length, sum of per-batch(max)*batch_size
    bucket = 0
    for i in range(0, n, batch_size):
        chunk = lengths[i:i + batch_size]
        bucket += max(chunk) * len(chunk)
    # pack: ceil(real / max_len) full blocks
    pack = math.ceil(real / max_len) * max_len

    def useful(x):
        return real / x if x else 1.0

    p50 = lengths[n // 2]
    p90 = lengths[min(n - 1, int(n * 0.9))]
    mx = max(lengths)

    # Decision: pack only when it beats bucketing by a clear margin (>15% more
    # useful compute) AND examples are short enough that many fit per block.
    # Otherwise bucket -- zero-caveat, semantically identical, still trims waste.
    pack_gain = useful(pack) / useful(bucket) if useful(bucket) else 1.0
    if pack_gain > 1.15 and p90 < max_len * 0.5:
        strategy = "pack"
        reason = (f"short high-variance examples (p50={p50}, p90={p90}, max_len={max_len}); "
                  f"packing gives {pack_gain:.1f}x more useful compute than bucketing")
    else:
        strategy = "bucket"
        reason = (f"examples already fill sequences (p50={p50}, p90={p90}); "
                  f"bucketing removes padding waste with no packing caveats")

    return {
        "strategy": strategy, "reason": reason,
        "examples": n, "p50_len": p50, "p90_len": p90, "max_len_seen": mx,
        "useful_naive": useful(naive), "useful_bucket": useful(bucket),
        "useful_pack": useful(pack),
        "pack_gain_over_bucket": pack_gain,
    }


def build_bucketed_dataloader(records: list[dict], tokenizer, batch_size: int = 4,
                              max_len: int = 1024, shuffle: bool = True):
    """Length-bucketed dynamic padding: sort by length so each batch pads only to
    its own longest member, not the global max. Fully general, semantically
    identical to naive padding -- it just stops computing on padding you'd
    discard. Batches are shuffled (order randomized) so training still sees
    varied batches; only *within-batch* lengths are similar.
    """
    import random
    import torch
    from torch.utils.data import DataLoader, Dataset

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    toks = _tokenize_records(records, tokenizer, max_len)
    # Sort by length, then chunk into batches, then shuffle the batch order.
    toks.sort(key=lambda t: len(t["input_ids"]))
    batches = [toks[i:i + batch_size] for i in range(0, len(toks), batch_size)]
    if shuffle:
        random.shuffle(batches)
    flat = [ex for b in batches for ex in b]

    class BucketDataset(Dataset):
        def __len__(self):
            return len(flat)

        def __getitem__(self, i):
            return flat[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            k = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * k)
            labels.append(b["labels"] + [-100] * k)
            attn.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # shuffle=False: we already shuffled batch order and must keep length-sorted
    # examples contiguous so each batch stays low-padding.
    return DataLoader(BucketDataset(), batch_size=batch_size, shuffle=False,
                      collate_fn=collate)


def build_adaptive_dataloader(records: list[dict], tokenizer, batch_size: int = 4,
                              max_len: int = 1024, shuffle: bool = True,
                              strategy: str = "auto"):
    """Pick the batching strategy that minimizes wasted padding-FLOPs for THIS
    dataset, then build it. `strategy` may be 'auto', 'pack', 'bucket', or 'pad'.
    Returns (dataloader, info)."""
    if strategy == "auto":
        info = analyze_lengths(records, tokenizer, max_len, batch_size)
        strategy = info["strategy"]
    else:
        info = {"strategy": strategy, "reason": "user-forced"}

    if strategy == "pack":
        return build_packed_dataloader(records, tokenizer, batch_size, max_len,
                                       shuffle), info
    if strategy == "bucket":
        return build_bucketed_dataloader(records, tokenizer, batch_size, max_len,
                                         shuffle), info
    return build_dataloader(records, tokenizer, batch_size, max_len, shuffle), info
