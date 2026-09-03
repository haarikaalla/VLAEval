"""Integration-style unit test exercising the full training loop on synthetic data."""

from __future__ import annotations

from vla_eval.data.preprocessing import make_synthetic_dataset
from vla_eval.models.baseline import BaselineVLAModel
from vla_eval.training.callbacks import EarlyStopping, ModelCheckpoint
from vla_eval.training.trainer import Trainer, TrainingConfig


def test_trainer_fit_runs_end_to_end(tmp_path) -> None:
    dataset = make_synthetic_dataset(num_samples=32, action_dim=4, state_dim=4)
    model = BaselineVLAModel(state_dim=4, action_dim=4, device="cpu")

    config = TrainingConfig(
        run_name="unit-test-run",
        num_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_workers=0,
        val_ratio=0.25,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10),
        ModelCheckpoint(checkpoint_dir=config.checkpoint_dir, monitor="val_loss"),
    ]
    trainer = Trainer(model, config, callbacks=callbacks)
    result = trainer.fit(dataset)

    assert len(result["history"]["train_loss"]) == 2
    assert (tmp_path / "checkpoints" / "final_model.pt").exists()
