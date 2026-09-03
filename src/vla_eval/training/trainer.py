"""Core supervised fine-tuning loop for VLA policies (behavior cloning).

Framework-light by design: a plain PyTorch loop rather than a heavyweight
trainer framework, to keep the codebase transparent and easy to extend
(e.g. adding action-chunking losses, diffusion losses, or auxiliary
objectives specific to a given backbone).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vla_eval.core.logging import get_logger
from vla_eval.data.preprocessing import VLAEpisodeDataset
from vla_eval.models.base import BaseVLAModel
from vla_eval.training.callbacks import Callback
from vla_eval.training.mlflow_utils import log_metrics, mlflow_run
from vla_eval.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for a training run (mirrors `configs/training/*.yaml`)."""

    run_name: str = "vla-training-run"
    experiment_name: str | None = None
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "./models/checkpoints"
    val_ratio: float = 0.1
    early_stopping_patience: int = 5
    log_every_n_steps: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.from_numpy(np.stack([b["image"] for b in batch])).float()
    states = torch.from_numpy(np.stack([b["state"] for b in batch])).float()
    actions = torch.from_numpy(np.stack([b["action"] for b in batch])).float()
    instructions = [b["instruction"] for b in batch]
    return {"image": images, "state": states, "action": actions, "instruction": instructions}


class Trainer:
    """Runs supervised (behavior-cloning) fine-tuning of a `BaseVLAModel`."""

    def __init__(
        self,
        model: BaseVLAModel,
        config: TrainingConfig,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        set_seed(config.seed)

    def fit(self, dataset: VLAEpisodeDataset) -> dict[str, Any]:
        """Train on `dataset`, splitting into train/val, and return run history."""
        cfg = self.config
        train_ds, val_ds = dataset.split(val_ratio=cfg.val_ratio, seed=cfg.seed)

        train_loader: DataLoader[dict[str, Any]] = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            drop_last=False,
        )
        val_loader: DataLoader[dict[str, Any]] = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        total_steps = max(1, cfg.num_epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            total_steps=total_steps,
            pct_start=min(0.3, cfg.warmup_steps / total_steps) if total_steps > 0 else 0.1,
        )
        loss_fn = torch.nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        params = {
            "model": self.model.name,
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.learning_rate,
            "dataset": dataset.name,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
        }

        with mlflow_run(cfg.run_name, experiment_name=cfg.experiment_name, params=params):
            global_step = 0
            for epoch in range(cfg.num_epochs):
                epoch_start = time.time()
                train_loss = self._train_epoch(
                    train_loader, optimizer, scheduler, loss_fn, epoch, global_step
                )
                global_step += len(train_loader)
                val_loss = self._validate_epoch(val_loader, loss_fn)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                epoch_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time_sec": time.time() - epoch_start,
                }
                log_metrics(epoch_metrics, step=epoch)
                logger.info("epoch_complete", epoch=epoch, **epoch_metrics)

                stop = False
                for callback in self.callbacks:
                    stop = callback.on_epoch_end(epoch, epoch_metrics, self.model) or stop
                if stop:
                    logger.info("training_stopped_early", epoch=epoch)
                    break

            final_path = str(Path(cfg.checkpoint_dir) / "final_model.pt")
            Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.model.save(final_path)

        return {"history": history, "final_checkpoint": final_path}

    def _train_epoch(
        self,
        loader: DataLoader[dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        epoch: int,
        step_offset: int,
    ) -> float:
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0
        net = getattr(self.model, "net", None)

        for i, batch in enumerate(loader):
            image = batch["image"].to(self.config.device)
            state = batch["state"].to(self.config.device)
            action = batch["action"].to(self.config.device)

            optimizer.zero_grad()
            if net is not None:
                pred = net(image, state)
            else:
                raise RuntimeError(
                    "Trainer currently supports models exposing a `.net` forward pass "
                    "(e.g. BaselineVLAModel). Extend Trainer for OpenVLA/LeRobot native losses."
                )
            loss = loss_fn(pred, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (step_offset + i) % self.config.log_every_n_steps == 0:
                logger.debug("train_step", epoch=epoch, step=step_offset + i, loss=loss.item())

        return total_loss / max(num_batches, 1)

    def _validate_epoch(
        self, loader: DataLoader[dict[str, Any]], loss_fn: torch.nn.Module
    ) -> float:
        self.model.eval_mode()
        total_loss = 0.0
        num_batches = 0
        net = getattr(self.model, "net", None)
        if net is None:
            raise RuntimeError(
                "Trainer currently supports models exposing a `.net` forward pass "
                "(e.g. BaselineVLAModel). Extend Trainer for OpenVLA/LeRobot native losses."
            )

        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(self.config.device)
                state = batch["state"].to(self.config.device)
                action = batch["action"].to(self.config.device)
                pred = net(image, state)
                loss = loss_fn(pred, action)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)
