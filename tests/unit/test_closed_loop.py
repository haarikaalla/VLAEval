from __future__ import annotations

import numpy as np

from vla_eval.evaluation.benchmark import run_closed_loop_benchmark
from vla_eval.evaluation.synthetic_reach import SyntheticReachEnv
from vla_eval.models.base import Action, Observation


class OriginSeekingPolicy:
    name = "origin-seeking-policy"
    action_dim = 3

    def reset(self) -> None:
        return None

    def predict(self, observation: Observation) -> Action:
        assert observation.state is not None
        return Action(values=np.clip(-observation.state, -0.25, 0.25))


def test_synthetic_reach_closed_loop_reports_success() -> None:
    env = SyntheticReachEnv(action_dim=3, seed=7, max_steps=20)
    result = run_closed_loop_benchmark(OriginSeekingPolicy(), env, num_episodes=5, max_steps=20)

    assert result.dataset_name == "synthetic-reach-v1"
    assert result.success_rate == 1.0
    assert result.num_samples == 5
    assert result.composite_score > 60


def test_synthetic_reach_rejects_wrong_action_shape() -> None:
    env = SyntheticReachEnv(action_dim=3)
    env.reset()
    try:
        env.step(np.zeros(2, dtype=np.float32))
    except ValueError as exc:
        assert "Expected action shape" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("wrong-sized action should fail")
