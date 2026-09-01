"""Extract Contextualized Target Representations for a dataset.

Input:  checkpoint-best/ from finetune.py, data.jsonl (any of train-filtered,
        test-exposed, test-heldout from create_splits.py)
Output: {data_stem}[-masked].npy (one pooled vector per row, in the original
        row order) plus a matching .jsonl of metadata, consumed by geometry.py
"""

# %%
# --- Imports
import logging
from pathlib import Path
from typing import cast

import numpy as np
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

    data_path: str = "path/to/data.jsonl"

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
data_ds = load_dataset("json", data_files={"data": cfg.data_path})["data"]

# %%
# --- Preprocessing
logging.info("Preprocessing")
with timer("Tokenizing"):
    data_ds = data_ds.map(tokenize, batched=True)

with timer("Creating target mask"):
    data_ds = data_ds.map(create_target_mask, batched=True)

if cfg.mask_target:
    with timer("Masking target"):
        data_ds = data_ds.map(mask_target, batched=True)

# Sort by length to minimize padding -> faster inference.
# Add an index beforehand so the original order can be restored afterward.
data_ds = data_ds.add_column("orig_index", list(range(len(data_ds))))
data_ds = data_ds.map(
    lambda batch: {"length": [len(ids) for ids in batch["input_ids"]]}, batched=True
)
data_ds = data_ds.sort("length")

model_input_ds = data_ds.select_columns([
    "input_ids",
    "attention_mask",
    "target_mask",
    "label",
])

# %%
# --- Embed
data_dl = DataLoader(
    model_input_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=data_collator,
)

embeddings = []
with timer("Embedding"), torch.no_grad():
    for batch in tqdm(data_dl, leave=False):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden_states = model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        # Reuse the model's own pooling so the representation is exactly what
        # the classification head sees.
        pooled = model.pool(hidden_states, batch["target_mask"])
        embeddings.append(pooled.cpu())

embeddings = torch.cat(embeddings).numpy().astype(np.float32)

# %%
# --- Write
logging.info("Writing embeddings")

# Restore the original order in both the matrix and the metadata, so the two
# stay row-aligned by position.
order = np.argsort(data_ds["orig_index"])
embeddings = embeddings[order]
data_ds = data_ds.sort("orig_index")

metadata = data_ds.remove_columns([
    "input_ids",
    "attention_mask",
    "target_mask",
    "offset_mapping",
    "length",
    "orig_index",
])

# Masked and unmasked runs can share an output dir, so the condition goes in
# the filename to keep them from overwriting each other, matching predict.py.
stem = Path(cfg.data_path).stem
suffix = "-masked" if cfg.mask_target else ""
np.save(run.dir / f"{stem}{suffix}.npy", embeddings)
write_jsonl(metadata.to_list(), run.dir / f"{stem}{suffix}.jsonl")

# %%
# --- Done
run.done()
