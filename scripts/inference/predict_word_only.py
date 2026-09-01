"""Fit the Word-only probe and predict with it on a test set.

A logistic regression on static, out-of-context word embeddings looked up
from a fine-tuned checkpoint's embedding table.

Input:  checkpoint-best/ from finetune.py, train.jsonl (same one finetune.py
        used, so the probe sees the same lemmas the full model was exposed
        to), test.jsonl
Output: {test_stem}-word-only.jsonl (test rows + pred column), consumed by
        compute_metrics.py and the analysis scripts
"""

# %%
# --- Imports
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from learning_the_cue.modeling import SequenceClassifier
from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.utils import read_jsonl, timer, write_jsonl


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"

    checkpoint_path: str = "path/to/checkpoint_dir/"

    train_path: str = "path/to/train.jsonl"
    test_path: str = "path/to/test.jsonl"

    # The surface word form as it appears in the sentence (not the lemma),
    # tokenized in isolation and looked up in the static embedding table.
    target_field: str = "target_text"


cfg = Config()
run = Run(cfg)

# %%
# --- Setup
logging.info("Setting up")

model = SequenceClassifier.from_pretrained(cfg.checkpoint_path)
# get_input_embeddings() is generic across architectures (RoBERTa, ELECTRA,
# DeBERTa, ModernBERT, ...), unlike reaching for a specific attribute path.
word_embeddings = model.encoder.get_input_embeddings().weight.detach().numpy()
tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_path)


def word_embedding(word):
    """Static embedding for a word, out of context: tokenize in isolation and
    average subword embeddings if the word splits into multiple tokens."""
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    return word_embeddings[token_ids].mean(axis=0)


# %%
# --- Load data
logging.info("Loading data")
train_df = pd.DataFrame(read_jsonl(cfg.train_path))
test_df = pd.DataFrame(read_jsonl(cfg.test_path))

# %%
# --- Extract static word features
with timer("Extracting word embeddings"):
    X_train = np.stack(train_df[cfg.target_field].map(word_embedding))
    X_test = np.stack(test_df[cfg.target_field].map(word_embedding))

y_train = train_df["label"].to_numpy()

# %%
# --- Fit the probe
# class_weight="balanced" compensates for the label imbalance in the training
# set (unlike the evaluation sets, which are already balanced).
with timer("Fitting logistic regression"):
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=cfg.seed
    )
    clf.fit(X_train, y_train)

# %%
# --- Predict
logging.info("Predicting")
test_df["pred"] = clf.predict(X_test)

# %%
# --- Write
logging.info("Writing predictions")
# Named after the test file, like predict.py, so different eval sets don't
# overwrite each other; "-word-only" distinguishes this from the full model's
# predictions when both share an output directory.
stem = Path(cfg.test_path).stem
write_jsonl(test_df.to_dict(orient="records"), run.dir / f"{stem}-word-only.jsonl")

# %%
# --- Done
run.done()
