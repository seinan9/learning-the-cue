"""Fine-tune a SequenceClassifier on VUA-style metaphor detection data.

Input:  train.jsonl (sentence, target_start/end, label) from convert.py or
        create_splits.py (train, train-filtered)
Output: checkpoints/checkpoint-best/ (renamed from the best eval checkpoint),
        consumed by every script under scripts/inference and scripts/analysis
"""

# %%
# --- Imports
import logging
from pathlib import Path
from typing import Literal, cast

from datasets import ClassLabel, load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from learning_the_cue.evaluation import compute_classification_metrics, for_trainer
from learning_the_cue.modeling import SequenceClassifier, SequenceClassifierConfig
from learning_the_cue.preprocessing import (
    TargetMaskDataCollatorWithPadding,
    create_target_mask,
    get_tokenize_fn,
)
from learning_the_cue.run import Run, RunConfig, dataclass
from learning_the_cue.training import LoggingTrainer
from learning_the_cue.utils import timer


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"
    seed: int = 42

    # Model
    encoder_name: str = "roberta-base"
    num_labels: int = 2
    dropout: float = 0.1

    # Data
    train_path: str = "path/to/train.jsonl"
    eval_size: float = 0.05
    data_seed: int = 42
    max_length: int = 256

    # Optimization
    num_train_epochs: float = 10
    batch_size: int = 32
    learning_rate: float = 4e-5
    weight_decay: float = 0.03
    warmup_steps: float = 0.1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.1
    precision: Literal["no", "fp16", "bf16"] = "bf16"

    # Checkpointing / eval
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    metric_for_best_model: str = "f1"
    mark_best_checkpoint: bool = True

    # Misc
    logging_steps: int = 50
    num_workers: int = 4
    pin_memory: bool = True
    report_to: str = "none"


cfg = Config()
run = Run(cfg)

# %%
# --- Setup
logging.info("Setting up")

# Model and tokenizer
encoder = AutoModel.from_pretrained(cfg.encoder_name)
config = SequenceClassifierConfig(
    num_labels=cfg.num_labels,
    dropout=cfg.dropout,
    encoder_config=encoder.config.to_dict(),
)
model = SequenceClassifier(config, encoder)
tokenizer = cast(
    PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(cfg.encoder_name)
)

# Preprocessing
# Offsets mapping is required for creating the target mask
tokenize = get_tokenize_fn(
    tokenizer,
    source_field="sentence",
    return_offsets_mapping=True,
    max_length=cfg.max_length,
    truncation=True,
)

data_collator = TargetMaskDataCollatorWithPadding(
    tokenizer,
    pad_to_multiple_of=8,
)

# %%
# --- Load data
logging.info("Loading data")
train_ds = load_dataset("json", data_files=cfg.train_path)["train"]

# %%
# --- Preprocess data
logging.info("Preprocessing data")

with timer("Tokenizing"):
    train_ds = train_ds.map(tokenize, batched=True)

with timer("Creating target mask"):
    train_ds = train_ds.map(create_target_mask, batched=True)

logging.info("Splitting data")
train_ds = train_ds.cast_column("label", ClassLabel(cfg.num_labels))
ds_dict = train_ds.train_test_split(
    test_size=cfg.eval_size,
    seed=cfg.data_seed,
    stratify_by_column="label",
)
ds_dict = ds_dict.select_columns([
    "input_ids",
    "attention_mask",
    "target_mask",
    "label",
])

train_ds = ds_dict["train"]
eval_ds = ds_dict["test"]

# %%
# --- Training
training_args = TrainingArguments(
    output_dir=str(run.dir / "checkpoints"),
    seed=cfg.seed,
    data_seed=cfg.data_seed,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    learning_rate=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    warmup_steps=cfg.warmup_steps,
    lr_scheduler_type=cfg.lr_scheduler_type,
    max_grad_norm=cfg.max_grad_norm,
    bf16=cfg.precision == "bf16",
    fp16=cfg.precision == "fp16",
    eval_strategy=cfg.eval_strategy,
    save_strategy=cfg.save_strategy,
    load_best_model_at_end=True,
    metric_for_best_model=cfg.metric_for_best_model,
    greater_is_better=True,
    save_total_limit=cfg.save_total_limit,
    logging_steps=cfg.logging_steps,
    dataloader_num_workers=cfg.num_workers,
    dataloader_pin_memory=cfg.pin_memory,
    report_to=cfg.report_to,
    label_smoothing_factor=cfg.label_smoothing_factor,
)

trainer = LoggingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=for_trainer(compute_classification_metrics),
)

with timer("Training"):
    trainer.train()

# %%
# --- Rename best checkpoint
# Trainer records the winning checkpoint's path once training ends; renaming
# it (rather than re-saving) keeps trainer_state.json and all other checkpoint
# contents exactly as Trainer wrote them, and gives it a fixed, seed-independent
# name that downstream scripts can rely on.
if cfg.mark_best_checkpoint:
    logging.info("Renaming best checkpoint")
    best_dir = run.dir / "checkpoints" / "checkpoint-best"
    Path(trainer.state.best_model_checkpoint).rename(best_dir)

# %%
# --- Done
run.done()
