# `scripts`

One subfolder per pipeline stage. Every script is runnable on its own; see
its module docstring for exactly what it reads and writes, and its `Config`
dataclass (or `--help`) for its arguments.

## `data/`

Convert the raw VUA corpus and build the Exposed/Held-out lexical splits.

| Script | Contents |
|---|---|
| `convert.py` | Raw VUA CSVs → JSONL, resolving each target to a lemma and character span |
| `create_splits.py` | Builds the Exposed/Held-out splits and the filtered training set |

## `training/`

| Script | Contents |
|---|---|
| `finetune.py` | Fine-tunes a `SequenceClassifier` on a training set |

## `inference/`

Turn a fine-tuned checkpoint into predictions or embeddings.

| Script | Contents |
|---|---|
| `predict.py` | Predicts on a test set; the Full Model and Context-only conditions differ by one flag (`--mask_target`) |
| `predict_word_only.py` | Fits and predicts with the Word-only probe (a logistic regression on static word embeddings) |
| `embed.py` | Extracts Contextualized Target Representations for a dataset |

## `evaluation/`

Turn predictions or embeddings into metrics.

| Script | Contents |
|---|---|
| `compute_metrics.py` | Precision/Recall/F1 from a predictions file |
| `knn_probe.py` | Neighborhood Purity and k-NN F1 from a pair of embeddings |

## Pipeline order

```
data/convert.py → data/create_splits.py → training/finetune.py
                                                  │
                            ┌─────────────────────┼─────────────────────┐
                            ▼                                           ▼
                inference/predict.py                          inference/embed.py
              inference/predict_word_only.py                          │
                            │                                         ▼
                            ▼                              evaluation/knn_probe.py
                evaluation/compute_metrics.py
```

Every experiment in the paper runs across 9 seeds; each script here handles
one seed at a time.
