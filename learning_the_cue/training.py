import logging

from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import Trainer
from transformers.trainer_callback import ProgressCallback


class _ProgressCallback(ProgressCallback):
    """Progress bar that shows epoch fraction and suppresses dict printing."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass  # the default callback dumps logs as a raw dict; _LoggingMixin.log handles it instead

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if self.training_bar is not None:
            epoch = state.epoch or 0.0
            self.training_bar.set_postfix_str(
                f"Epoch {epoch:.2f}/{state.num_train_epochs}"
            )


class _LoggingMixin:
    """Formatted logging for HuggingFace trainers.

    Mix in before `Trainer` (see `TrainerWithLogging`) to replace its raw dict
    logging with a single readable line per log event, without touching the
    rest of Trainer's behavior.
    """

    def log(self, logs: dict, start_time: float | None = None) -> None:
        super().log(logs, start_time)  # type: ignore[misc]
        with logging_redirect_tqdm():  # keep the progress bar intact while logging
            logging.info(", ".join(self._build_parts(logs)))

    def _build_parts(self, logs: dict) -> list[str]:
        """Assemble the log line's comma-separated parts for one log event."""
        parts = []

        if "loss" in logs:
            parts += self._format_train(logs)

        if "eval_loss" in logs:
            parts += self._format_eval(logs)

        if "train_runtime" in logs:
            parts += [
                f"Runtime: {float(logs['train_runtime']):.0f}s",
                f"Tok/s: {float(logs.get('train_tokens_per_second', 0)):.0f}",
            ]

        parts += [
            f"Step: {self.state.global_step}",  # type: ignore[misc]
            f"Epoch: {float(logs.get('epoch', self.state.epoch)):.3}",  # type: ignore[misc]
        ]

        return parts

    def _format_train(self, logs: dict) -> list[str]:
        return [
            f"Loss: {float(logs['loss']):.3f}",
            f"GradNorm: {float(logs.get('grad_norm', 0)):.3f}",
            f"LR: {float(logs.get('learning_rate', 0)):.3e}",
        ]

    def _format_eval(self, logs: dict) -> list[str]:
        """Eval loss plus every eval_* metric Trainer logged (e.g. eval_f1), auto-discovered."""
        skip = {
            "eval_loss",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        }
        parts = [f"Eval.Loss: {float(logs['eval_loss']):.3f}"]
        for key, value in logs.items():
            if key.startswith("eval_") and key not in skip:
                label = key[5:].replace("_", " ").title().replace(" ", "")
                parts.append(f"Eval.{label}: {float(value):.3f}")
        return parts


def _setup(trainer):
    """Swap in the epoch-aware progress bar in place of Trainer's default one."""
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(_ProgressCallback())


class LoggingTrainer(_LoggingMixin, Trainer):
    """Drop-in replacement for `Trainer` with readable, single-line logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _setup(self)
