"""Convert raw VUA-style CSVs into the pipeline's JSONL format, resolving
each row's whitespace-indexed target to a Stanza-tokenized lemma and span.

Input:  --train_path/--test_path CSVs (sentence, target_index, label)
Output: {source}-{train,test}.jsonl, consumed by create_splits.py
"""

# %%
# --- Imports
import csv
import logging
import re

import stanza
from tqdm import tqdm

from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.utils import chunked, write_jsonl


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"
    seed: int = 42

    # Raw VUA-style CSVs: one row per instance, with sentence, target_index
    # (1-based, whitespace-token position), and label columns.
    train_path: str = "path/to/train.csv"
    test_path: str = "path/to/test.csv"
    source_name: str = "vua-verbs"

    batch_size: int = 32
    use_gpu: bool = True


cfg = Config()
run = Run(cfg)

# %%
# --- Setup
logging.info("Setting up")
pipeline = stanza.Pipeline(
    lang="en",
    processors="tokenize,lemma",
    tokenize_no_ssplit=True,  # sentences are already split; don't re-split them
    verbose=False,
    use_gpu=cfg.use_gpu,
)


# %%
# --- Load data
def load(data_path, split):
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{**row, "split": split} for row in reader]


logging.info("Loading data")
train_rows = load(cfg.train_path, "train")
test_rows = load(cfg.test_path, "test")


# %%
# --- Convert
def resolve_target(words, target_index, sentence):
    """Locate the target token via whitespace tokenization char spans,
    then find the Stanza token(s) overlapping that span."""
    ws_tokens = list(re.finditer(r"\S+", sentence))
    target_span = ws_tokens[target_index]
    char_start, char_end = target_span.start(), target_span.end()

    return [w for w in words if w.start_char < char_end and w.end_char > char_start]


def convert(rows):
    """Resolve each row's whitespace-indexed target to a Stanza word, adding
    its lemma and character span. Rows whose target overlaps zero or more
    than one Stanza word (tokenization mismatch) are skipped, not dropped
    silently -- they're returned separately so they can be inspected."""
    converted = []
    skipped = []

    for batch in tqdm(chunked(rows, cfg.batch_size)):
        docs = pipeline.bulk_process([row["sentence"] for row in batch])

        for sample, doc in zip(batch, docs):
            words = doc.sentences[0].words
            target_index = int(sample["target_index"]) - 1
            overlapping = resolve_target(words, target_index, sample["sentence"])

            if len(overlapping) != 1:
                skipped.append({**sample, "n_overlap": len(overlapping)})
                continue

            target_word = overlapping[0]
            converted.append({
                "source": cfg.source_name,
                "split": sample["split"],
                "id": sample["id"],
                "sentence": sample["sentence"],
                "target_text": target_word.text,
                "target_lemma": target_word.lemma,
                "target_start": target_word.start_char,
                "target_end": target_word.end_char,
                "label": int(sample["label"]),
            })
    return converted, skipped


logging.info("Converting data")
train_rows, train_skipped = convert(train_rows)
test_rows, test_skipped = convert(test_rows)

# %%
# --- Write
logging.info("Writing data")
write_jsonl(train_rows, run.dir / f"{cfg.source_name}-train.jsonl")
write_jsonl(test_rows, run.dir / f"{cfg.source_name}-test.jsonl")
write_jsonl(train_skipped, run.dir / f"{cfg.source_name}-train-skipped.jsonl")
write_jsonl(test_skipped, run.dir / f"{cfg.source_name}-test-skipped.jsonl")

# %%
# --- Done
run.done()
