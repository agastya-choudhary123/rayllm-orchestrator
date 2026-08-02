"""Score a model on a held-out classification test set (exact match).

    # base model, no fine-tuning
    python examples/eval/evaluate.py /tmp/eval/emotion_test.jsonl <model-id>

    # same model + the adapters `rayllm train` produced
    python examples/eval/evaluate.py /tmp/eval/emotion_test.jsonl <model-id> \
        --adapter ./checkpoints/<name>

Greedy decoding (temperature 0) so runs are reproducible. The label set is
inferred from the gold answers. A response counts as correct only if exactly one
label appears in it -- zero or several means the model did not answer the
question, which is scored wrong rather than quietly excused.

Apple Silicon / MLX only, matching the trainer's MLX path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from orchestrator.backends_mlx import MLXEngine  # noqa: E402


def _norm(s: str) -> str:
    """Compare labels loosely enough that `card_arrival` and `card arrival`
    count as the same answer -- punctuation style is not what we're scoring."""
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def _match(out: str, labels: list[str]) -> str | None:
    """The single label the response names, or None.

    Some label sets nest ("card not working" inside "virtual card not working"),
    so a bare substring test would call every such answer ambiguous. Keep only
    maximal matches -- those not contained in another match -- and require
    exactly one to survive.
    """
    out_n = _norm(out)
    hits = [l for l in labels if _norm(l) in out_n]
    maximal = [a for a in hits
               if not any(a != b and _norm(a) in _norm(b) for b in hits)]
    return maximal[0] if len(maximal) == 1 else None


def evaluate(test_file: str, model: str, adapter: str | None,
             max_tokens: int, save: str | None) -> float:
    rows = [json.loads(line) for line in open(test_file)]
    if not rows:
        raise SystemExit(f"{test_file} is empty")
    labels = sorted({r["gold"] for r in rows})
    tag = "fine-tuned" if adapter else "base"

    engine = MLXEngine(model, adapter_path=adapter)

    correct = unparseable = 0
    preds = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        out = engine.generate(row["prompt"], max_tokens=max_tokens,
                              temperature=0.0)
        pred = _match(out, labels)
        if pred is None:
            unparseable += 1
        if pred == row["gold"]:
            correct += 1
        preds.append({"gold": row["gold"], "pred": pred, "raw": out[:80]})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  running accuracy={correct / i:.1%}",
                  flush=True)

    dt = time.time() - t0
    acc = correct / len(rows)
    majority = max(Counter(r["gold"] for r in rows).values()) / len(rows)

    print(f"\n=== {tag}: {model}" + (f" + {adapter}" if adapter else "") + " ===")
    print(f"accuracy     : {acc:.1%}  ({correct}/{len(rows)})")
    print(f"majority base: {majority:.1%}")
    print(f"unparseable  : {unparseable}/{len(rows)}")
    print(f"prediction   : {dict(Counter(p['pred'] for p in preds))}")
    print(f"eval time    : {dt:.1f}s  ({len(rows) / dt:.1f} examples/s)")

    if save:
        with open(save, "w") as f:
            json.dump({"tag": tag, "model": model, "adapter": adapter,
                       "accuracy": acc, "correct": correct, "n": len(rows),
                       "majority_baseline": majority,
                       "unparseable": unparseable, "seconds": dt,
                       "predictions": preds}, f, indent=2)
        print(f"predictions saved to {save}")
    return acc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("test_file", help="*_test.jsonl from prepare_data.py")
    p.add_argument("model", help="Base model id or path.")
    p.add_argument("--adapter", default=None,
                   help="Checkpoint dir from `rayllm train`. Omit to score the "
                        "base model.")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--save", default=None, help="Write per-example predictions here.")
    a = p.parse_args()
    evaluate(a.test_file, a.model, a.adapter, a.max_tokens, a.save)


if __name__ == "__main__":
    main()
