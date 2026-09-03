"""Unit tests for the baseline CNN policy and the model registry."""

from __future__ import annotations

import numpy as np

from vla_eval.models.base import Observation
from vla_eval.models.baseline import BaselineVLAModel
from vla_eval.models.registry import create_model, list_models


def test_baseline_model_predict_returns_correct_shape() -> None:
    model = BaselineVLAModel(state_dim=7, action_dim=7, device="cpu")
    obs = Observation(
        image=np.random.default_rng(0).random((3, 224, 224), dtype=np.float32),
        state=np.zeros(7, dtype=np.float32),
        instruction="pick up the cube",
    )
    action = model.predict(obs)
    assert action.values.shape == (7,)


def test_baseline_model_save_and_load_roundtrip(tmp_path) -> None:
    model = BaselineVLAModel(state_dim=7, action_dim=7, device="cpu")
    path = tmp_path / "model.pt"
    model.save(str(path))

    other = BaselineVLAModel(state_dim=7, action_dim=7, device="cpu")
    other.load(str(path))

    obs = Observation(
        image=np.zeros((3, 224, 224), dtype=np.float32), state=np.zeros(7, dtype=np.float32)
    )
    np.testing.assert_allclose(model.predict(obs).values, other.predict(obs).values)


def test_model_registry_lists_baseline() -> None:
    assert "baseline-cnn" in list_models()


def test_model_registry_create_baseline() -> None:
    model = create_model("baseline-cnn", state_dim=4, action_dim=4, device="cpu")
    assert model.action_dim == 4
