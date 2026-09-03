"""Unit tests for synthetic dataset generation and splitting."""

from __future__ import annotations

from vla_eval.data.preprocessing import make_synthetic_dataset


def test_make_synthetic_dataset_shapes() -> None:
    dataset = make_synthetic_dataset(num_samples=32, action_dim=5, state_dim=3)
    assert len(dataset) == 32
    sample = dataset[0]
    assert sample["action"].shape == (5,)
    assert sample["state"].shape == (3,)
    assert sample["image"].shape[0] == 3


def test_split_produces_disjoint_train_val() -> None:
    dataset = make_synthetic_dataset(num_samples=100)
    train, val = dataset.split(val_ratio=0.2, seed=1)
    assert len(train) + len(val) == 100
    assert len(val) == 20
