import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser


@dataclass
class RunConfig:
    """Fields every run needs. Subclass this and add experiment parameters."""

    output_dir: str = "runs/default"
    seed: int = 42

    log_level: str = "INFO"

    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


class Run:
    """Seeds, creates the run directory, and starts logging.

    Interactively the config is used as given; as a script its values are the
    defaults for command line flags derived from the same dataclass. The config
    is updated in place, so the object passed in stays the source of truth.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._start_time = time.time()
        interactive = _is_interactive()

        if not interactive:
            _apply_cli(self.cfg)

        set_seed(self.cfg.seed)

        # Interactive runs share one scratch directory; script runs get their
        # own, named after the config so results don't overwrite each other.
        if interactive:
            self.dir = Path.cwd() / "interactive"
        else:
            self.dir = Path(self.cfg.output_dir)

        self.dir.mkdir(parents=True, exist_ok=True)

        setup_logger(cfg.log_level, self.dir / "log")

        # Interactive shells already have their own excepthook (e.g. IPython);
        # only override it for script runs so tracebacks land in the log file.
        if not interactive:
            _log_uncaught_exceptions()

        logging.info("Run directory: %s", self.dir)
        logging.info("Config: %s", self.cfg)

    def done(self) -> None:
        """Log the total elapsed time. Call at the end of the script."""
        logging.info("Completed in %.1fs", time.time() - self._start_time)


def _apply_cli(cfg: RunConfig) -> None:
    """Override the config in place with command line flags, if any were given."""
    parser = HfArgumentParser(type(cfg))
    parser.set_defaults(**asdict(cfg))
    (parsed,) = parser.parse_args_into_dataclasses()
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(parsed, f.name))


def _log_uncaught_exceptions() -> None:
    """Route uncaught exceptions through logging so they reach the log file."""
    default_excepthook = sys.excepthook

    def handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            default_excepthook(exc_type, exc_value, exc_tb)
            return
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = handler


def _is_interactive() -> bool:
    """True in a REPL or Jupyter/IPython kernel, False when run as a script."""
    return hasattr(sys, "ps1") or "ipykernel" in sys.modules


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all GPUs) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging to stdout, and also to `log_file` if given.

    Third-party libraries are capped at WARNING so their INFO/DEBUG logs don't
    drown out our own.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    for name in (
        "urllib3",
        "requests",
        "filelock",
        "huggingface_hub",
        "transformers",
        "datasets",
        "httpx",
        "tqdm",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
