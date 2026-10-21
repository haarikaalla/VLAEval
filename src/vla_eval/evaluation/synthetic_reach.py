"""Deterministic closed-loop environment for integration and smoke evaluation.

This environment is intentionally lightweight: it proves the rollout contract,
episode accounting, latency measurement and success-rate publication without a
GPU or external simulator. Real robot claims still require a validated simulator
such as LIBERO/ManiSkill or physical-hardware evaluation.
"""

from __future__ import annotations

import numpy as np

from vla_eval.models.base import Observation


class SyntheticReachEnv:
    """Move an action-dimensional point to the origin using bounded actions."""

    name = "synthetic-reach-v1"

    def __init__(
        self,
        action_dim: int = 7,
        *,
        seed: int = 42,
        tolerance: float = 0.08,
        action_limit: float = 0.25,
        max_steps: int = 100,
    ) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.tolerance = tolerance
        self.action_limit = action_limit
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._position = np.zeros(action_dim, dtype=np.float32)
        self._steps = 0

    def _observation(self) -> Observation:
        image = np.zeros((3, 64, 64), dtype=np.float32)
        magnitude = min(float(np.linalg.norm(self._position)) / np.sqrt(self.action_dim), 1.0)
        image[0, :, :] = magnitude
        image[1, :, :] = 1.0 - magnitude
        return Observation(
            image=image,
            state=self._position.copy(),
            instruction="move the controlled point to the target origin",
        )

    def reset(self) -> Observation:
        self._position = self._rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        self._steps = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, dict]:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, received {values.shape}")
        self._position += np.clip(values, -self.action_limit, self.action_limit)
        self._steps += 1
        distance = float(np.linalg.norm(self._position))
        success = distance <= self.tolerance
        truncated = self._steps >= self.max_steps
        return (
            self._observation(),
            -distance,
            success or truncated,
            {
                "success": success,
                "distance": distance,
                "truncated": truncated and not success,
            },
        )
