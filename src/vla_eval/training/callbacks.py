"""Training callbacks: checkpointing and early stopping.

Kept intentionally simple (no external framework dependency) so the core
training loop in `trainer.py` remains easy to audit and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vla_eval.core.logging import get_logger

logger = get_logger(__name__)


class Callback(Protocol):
    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model) -> bool:
        """Return True to request early stopping."""
        ...


@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"  # "min" or "max"
    min_delta: float = 1e-4
    best: float = field(init=False, default=float("inf"))
    num_bad_epochs: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model) -> bool:
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            logger.info(
                "early_stopping_triggered",
                epoch=epoch,
                monitor=self.monitor,
                best=self.best,
                patience=self.patience,
            )
            return True
        return False


@dataclass
class ModelCheckpoint:
    """Save the best model (by a monitored metric) to disk each improving epoch."""

    checkpoint_dir: str
    monitor: str = "val_loss"
    mode: str = "min"
    best: float = field(init=False, default=float("inf"))

    def __post_init__(self) -> None:
        self.best = float("inf") if self.mode == "min" else float("-inf")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model) -> bool:
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            path = str(Path(self.checkpoint_dir) / "best_model.pt")
            model.save(path)
            logger.info("checkpoint_saved", epoch=epoch, path=path, **{self.monitor: value})
        return False
