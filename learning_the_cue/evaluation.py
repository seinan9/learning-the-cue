from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from learning_the_cue.utils import read_jsonl


def compute_classification_metrics(labels, preds) -> dict[str, float]:
    """Precision/recall/F1 for gold labels and predicted labels (both 1D int arrays)."""
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def for_trainer(metric_fn):
    """Adapt a (labels, preds) -> dict metric function for use as Trainer's compute_metrics."""

    def wrapped(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric_fn(labels, preds)

    return wrapped


def load_seed_metrics(parent_dir, filename, seeds) -> pd.DataFrame:
    """Collect one metrics file per seed into a single DataFrame.

    Expects the layout the pipeline scripts write: parent_dir/seed-<seed>/filename,
    each holding a single JSON record of metric name -> value (e.g. the output
    of compute_classification_metrics). Returns one row per seed, with a `seed`
    column alongside the metric columns.
    """
    records = []
    for seed in seeds:
        path = Path(parent_dir) / f"seed-{seed}" / filename
        metrics = read_jsonl(path)[0]
        records.append({"seed": seed, **metrics})

    return pd.DataFrame.from_records(records)


def mean_std(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean and standard deviation of every metric column (everything but seed)."""
    metrics = df.drop(columns="seed")
    return metrics.mean(), metrics.std()
