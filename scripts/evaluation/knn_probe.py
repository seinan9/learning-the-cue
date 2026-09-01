"""k-NN probe over Contextualized Target Representations: Neighborhood Purity
and k-NN F1, used for the geometric analysis in Section 4.3.

Input:  embeddings_dir from embed.py, holding <stem>.npy/.jsonl pairs for a
        reference set (train) and an evaluation set
Output: {eval_stem}-knn.jsonl (a single metrics record), consumed by
        every scripts/tables script via evaluation.load_seed_metrics
"""

# %%
# --- Imports
import logging
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from learning_the_cue.evaluation import compute_classification_metrics
from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.utils import read_jsonl, write_jsonl


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"

    # Directory written by embed.py, holding <stem>.npy and <stem>.jsonl pairs.
    embeddings_dir: str = "runs/embed/robertaf/seed-1"

    # The train embeddings form the reference space the eval set is projected
    # into. Both must come from the same condition: a masked eval set belongs
    # in a masked reference space, otherwise the neighbours still carry the
    # verb identity the condition is meant to remove.
    train_stem: str = "vua-verbs-train-filtered"
    eval_stem: str = "vua-verbs-test-exposed"

    k: int = 10


cfg = Config()
run = Run(cfg)

# %%
# --- Load embeddings
logging.info("Loading embeddings")
embeddings_dir = Path(cfg.embeddings_dir)


def load(stem):
    """Embedding matrix and labels, row-aligned by position (see embed.py)."""
    X = np.load(embeddings_dir / f"{stem}.npy")
    labels = [row["label"] for row in read_jsonl(embeddings_dir / f"{stem}.jsonl")]
    return X, np.array(labels)


X_train, y_train = load(cfg.train_stem)
X_eval, y_eval = load(cfg.eval_stem)

logging.info("Reference space %s: %s", cfg.train_stem, X_train.shape)
logging.info("Evaluation set %s: %s", cfg.eval_stem, X_eval.shape)

# %%
# --- Fit k-NN on the reference space
logging.info("Fitting k-NN (k=%d)", cfg.k)
knn = KNeighborsClassifier(n_neighbors=cfg.k, metric="cosine")
knn.fit(X_train, y_train)

# %%
# --- Neighborhood purity and k-NN F1
# k-NN F1 probes whether the label is recoverable without a learned decision
# boundary; purity asks how many of each instance's neighbours share its label,
# i.e. whether local structure is task-relevant at all.
neighbor_ids = knn.kneighbors(X_eval, return_distance=False)
purity = float((y_train[neighbor_ids] == y_eval[:, None]).mean())

preds = knn.predict(X_eval)
knn_f1 = compute_classification_metrics(y_eval, preds)["f1"]

metrics = {"purity": purity, "knn_f1": knn_f1}
logging.info(", ".join(f"{k.capitalize()}: {v:.3f}" for k, v in metrics.items()))

# %%
# --- Write
# Metrics-shaped output, so the seed aggregation used for every other table
# works here too -- it is generic over metric names.
logging.info("Writing metrics")
write_jsonl([metrics], run.dir / f"{cfg.eval_stem}-knn.jsonl")

# %%
# --- Done
run.done()
