"""Run a fine-tuned checkpoint over a test set and save its predictions.

Input:  checkpoint-best/ from finetune.py, test.jsonl from create_splits.py
Output: {test_stem}[-masked].jsonl (test rows + pred column), consumed by
        compute_metrics.py and every scripts/analysis script
"""

# %%
# --- Imports
import logging
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from learning_the_cue.modeling import SequenceClassifier
from learning_the_cue.preprocessing import (
    TargetMaskDataCollatorWithPadding,
    create_target_mask,
    get_mask_target_fn,
    get_tokenize_fn,
)
from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.utils import timer, write_jsonl


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir"

    device_id: str = "cuda:0"
    checkpoint_path: str = "path/to/checkpoint/"

    test_path: str = "path/to/test.jsonl"

    # Context-only condition: replace the target word's token(s) with the mask
    # token, removing verb identity while keeping the surrounding context.
    mask_target: bool = False

    max_length: int = 256
    batch_size: int = 64
    num_workers: int = 4


cfg = Config()
run = Run(cfg)

# %%
# --- Setup
logging.info("Setting up")

device = torch.device(cfg.device_id)
model = SequenceClassifier.from_pretrained(cfg.checkpoint_path)
model.to(device)
model.eval()

# A saved checkpoint bundles its tokenizer, so this loads the exact tokenizer
# the model was fine-tuned with rather than looking it up by encoder name.
tokenizer = cast(
    PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(cfg.checkpoint_path)
)
if cfg.mask_target and tokenizer.mask_token_id is None:
    raise ValueError(f"{tokenizer} has no mask token; cannot run with mask_target=True")

tokenize = get_tokenize_fn(
    tokenizer,
    source_field="sentence",
    return_offsets_mapping=True,
    max_length=cfg.max_length,
    truncation=True,
)

mask_target = get_mask_target_fn(tokenizer.mask_token_id)

data_collator = TargetMaskDataCollatorWithPadding(
    tokenizer,
    pad_to_multiple_of=8,
)

# %%
# --- Load data
logging.info("Loading data")
test_ds = load_dataset("json", data_files={"test": cfg.test_path})["test"]

# %%
# --- Preprocessing
logging.info("Preprocessing")

with timer("Tokenizing"):
    test_ds = test_ds.map(tokenize, batched=True)

with timer("Creating target mask"):
    test_ds = test_ds.map(create_target_mask, batched=True)

if cfg.mask_target:
    with timer("Masking target"):
        test_ds = test_ds.map(mask_target, batched=True)

# Sort by length to minimize padding -> faster inference.
# Add an index beforehand so the original order can be restored afterward.
test_ds = test_ds.add_column("orig_index", list(range(len(test_ds))))
test_ds = test_ds.map(
    lambda batch: {"length": [len(ids) for ids in batch["input_ids"]]}, batched=True
)
test_ds = test_ds.sort("length")

model_input_ds = test_ds.select_columns([
    "input_ids",
    "attention_mask",
    "target_mask",
    "label",
])

# %%
# --- Predict
test_dl = DataLoader(
    model_input_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=data_collator,
)

all_preds = []
with timer("Predicting"), torch.no_grad():
    for batch in tqdm(test_dl, leave=False):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(**batch).logits.argmax(-1)
        all_preds.extend(preds.cpu().tolist())

# %%
# --- Write
logging.info("Writing predictions")
test_ds = test_ds.add_column("pred", all_preds)
test_ds = test_ds.sort("orig_index")
test_ds = test_ds.remove_columns([
    "input_ids",
    "attention_mask",
    "target_mask",
    "offset_mapping",
    "length",
    "orig_index",
])
# Masked and unmasked runs can share an output dir, so the condition goes in
# the filename to keep them from overwriting each other.
stem = Path(cfg.test_path).stem
suffix = "-masked" if cfg.mask_target else ""
write_jsonl(test_ds.to_list(), run.dir / f"{stem}{suffix}.jsonl")

# %%
# --- Done
run.done()
