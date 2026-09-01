# `learning_the_cue`

Shared components used across every script in `scripts/`.

| File | Contents |
|---|---|
| `run.py` | `Run`/`RunConfig`: per-run config, seeding, output directory, logging |
| `modeling.py` | `SequenceClassifier`: encoder plus a linear head over the pooled target span |
| `preprocessing.py` | Tokenization, target-mask construction, target masking, data collator |
| `training.py` | `TrainerWithLogging`: HuggingFace `Trainer` with readable logging |
| `evaluation.py` | Classification metrics and per-seed metric aggregation |
| `utils.py` | Timing, JSONL I/O, batching |

## `Run` and `RunConfig`

Every script defines a `Config` dataclass (subclassing `RunConfig`) and starts
with:

```python
cfg = Config()
run = Run(cfg)
```

For example, `scripts/training/finetune.py` defines:

```python
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"
    seed: int = 42

    encoder_name: str = "roberta-base"
    train_path: str = "path/to/train.jsonl"
    learning_rate: float = 4e-5
    num_train_epochs: float = 10
    ...

cfg = Config()
run = Run(cfg)
```

`Run` seeds Python/NumPy/PyTorch, creates `cfg.output_dir`, and writes a log
file there containing the full resolved config and everything the script
logs. As a script, `cfg`'s fields become command line flags
(`--learning_rate 3e-5`); in a notebook or `# %%` cell, the dataclass
defaults are used as given. This is what lets every script run both ways
with no code change, and what makes every output directory
self-documenting: its log file records exactly which config produced it.
