# Datasets — provenance, licensing, and quality

This project fine-tunes on **real, reputable, permissively-licensed** instruction
datasets. Nothing synthetic-from-a-black-box, nothing with unclear rights,
nothing with PII. You point `--dataset` at a local `.jsonl` or a HuggingFace
dataset id and the loader normalizes the common schemas automatically.

## Recommended datasets (all safe to use and cite)

| Dataset | HF id | Size | License | Why it's safe |
|---------|-------|------|---------|---------------|
| **Databricks Dolly 15k** | `databricks/databricks-dolly-15k` | 15,011 | **CC BY-SA 3.0** | Human-written by Databricks employees; commercially usable; no PII; widely cited. **Default choice.** |
| OpenAssistant oasst1 | `OpenAssistant/oasst1` | 88k msgs | Apache-2.0 | Community-generated, human-reviewed, permissive. |
| No Robots | `HuggingFaceH4/no_robots` | 10k | CC BY-NC 4.0 | Human-written, very high quality. *Non-commercial* — fine for research/portfolio, not for a commercial product. |

> **On Alpaca-style sets** (`tatsu-lab/alpaca`, etc.): these were generated with
> OpenAI models, so their license is *non-commercial* and their content can carry
> model-generated errors. We deliberately default to **Dolly** (human-written,
> CC BY-SA, commercially usable) to avoid both problems.

## How the loader handles schemas

`orchestrator/data.py::load_records` normalizes any of these into the training
format, so you don't have to reshape your data:

- `{"prompt", "completion"}`
- `{"instruction", "output"}`
- `{"instruction", "response", "context"}`  ← Dolly (context is folded in)
- `{"question", "answer"}`
- `{"text"}` (plain language modeling)

Unusable rows are dropped. `--max-examples N` caps how many rows load, so laptop
training stays time- and memory-bounded.

## How we prevent overfitting (and prove it)

Overfitting is the #1 thing people rightly question about a small fine-tune, so
the trainer is built to make it impossible to hide:

1. **Real held-out validation split.** The trainer shuffles deterministically
   and holds out a fraction (default 10%) that it **never trains on**. The
   reported `val_loss` is therefore a genuine generalization signal — not the
   training data scored against itself.
2. **Periodic val evaluation.** Val loss is measured several times during
   training, so a rising val loss (the signature of overfitting) is visible.
3. **A recorded verdict.** After training, `orchestrator.json` stores
   `final_train_loss`, `final_val_loss`, and an `overfit_verdict`
   (`healthy` / `watch` / `OVERFIT RISK`) based on the train↔val gap, plus the
   full `loss_history`. It's auditable, not a claim.
4. **Enough, diverse data.** Use hundreds-to-thousands of *diverse* examples
   (Dolly is broad general-knowledge), not a handful of repeated ones. A tiny
   repeated set will memorize — that's expected and why we don't ship that.

## Security / safety notes

- Datasets are fetched from HuggingFace over HTTPS into the local HF cache; no
  code from the dataset is executed (`trust_remote_code` is not enabled).
- No PII: the recommended sets are general-knowledge Q&A, not personal data.
- Everything runs locally — your data and the model never leave your machine.
- Licenses above are the upstream ones; keep attribution when you redistribute a
  fine-tuned model (CC BY-SA requires share-alike + attribution).
