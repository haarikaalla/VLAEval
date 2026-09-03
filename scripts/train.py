"""Hydra-driven training entry point for full experiment-config sweeps.

Usage:
    python scripts/train.py                                  # defaults
    python scripts/train.py model=openvla_lora training=lora_finetune dataset=pusht
    python scripts/train.py -m training.learning_rate=1e-4,5e-5  # multirun sweep
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from vla_eval.core.logging import configure_logging, get_logger
from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
from vla_eval.models.registry import create_model
from vla_eval.training.callbacks import EarlyStopping, ModelCheckpoint
from vla_eval.training.trainer import Trainer, TrainingConfig
from vla_eval.utils.device import resolve_device

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    logger.info("training_config", config=OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.training.device)

    if cfg.dataset.name == "synthetic":
        dataset = make_synthetic_dataset(
            num_samples=cfg.dataset.get("num_samples", 256),
            action_dim=cfg.dataset.get("action_dim", 7),
            state_dim=cfg.dataset.get("state_dim", 7),
        )
    else:
        dataset = preprocess_dataset(
            cfg.dataset.name,
            cache_dir=cfg.dataset.cache_dir,
            processed_dir=cfg.dataset.processed_dir,
            max_episodes=cfg.dataset.get("max_episodes"),
        )

    model_kwargs = {k: v for k, v in cfg.model.items() if k not in ("name", "type")}
    policy = create_model(cfg.model.type, device=device, **model_kwargs)

    training_config = TrainingConfig(
        run_name=cfg.training.run_name,
        experiment_name=cfg.training.experiment_name,
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        grad_clip_norm=cfg.training.grad_clip_norm,
        num_workers=cfg.training.num_workers,
        seed=cfg.seed,
        device=device,
        checkpoint_dir=cfg.training.checkpoint_dir,
        val_ratio=cfg.training.val_ratio,
        early_stopping_patience=cfg.training.early_stopping_patience,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=training_config.early_stopping_patience),
        ModelCheckpoint(checkpoint_dir=training_config.checkpoint_dir, monitor="val_loss"),
    ]

    trainer = Trainer(policy, training_config, callbacks=callbacks)
    result = trainer.fit(dataset)
    logger.info("training_finished", final_checkpoint=result["final_checkpoint"])


if __name__ == "__main__":
    main()
