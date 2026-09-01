"""Compute Precision/Recall/F1 from a predictions file.

Input:  predictions file from predict.py or predict_word_only.py (rows with
        label and pred columns)
Output: {predictions_stem}.jsonl (a single metrics record), consumed by
        every scripts/tables script via evaluation.load_seed_metrics
"""

# %%
# --- Imports
import logging
from pathlib import Path

from learning_the_cue.evaluation import compute_classification_metrics
from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.utils import read_jsonl, write_jsonl


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"
    predictions_path: str = "path/to/predictions.jsonl"


cfg = Config()
run = Run(cfg)

# %%
# --- Load data
logging.info("Loading data")
rows = read_jsonl(cfg.predictions_path)

# %%
# --- Compute metrics
logging.info("Computing metrics")
labels = [row["label"] for row in rows]
preds = [row["pred"] for row in rows]

metrics = compute_classification_metrics(labels, preds)
logging.info(", ".join([f"{k.capitalize()}: {v:.3f}" for k, v in metrics.items()]))

# %%
# --- Write
# Named after the predictions file, so the eval set (and any -masked or
# -word-only condition suffix) stays identifiable at every stage of the
# pipeline, matching how predict.py names its own output.
logging.info("Writing metrics")
write_jsonl([metrics], run.dir / Path(cfg.predictions_path).name)

# %%
# --- Done
run.done()
