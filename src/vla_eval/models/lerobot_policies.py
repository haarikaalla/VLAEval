"""Wrapper exposing `lerobot` policies (ACT, Diffusion Policy, VQ-BeT, ...)
through the shared `VLAPolicy` interface used by this platform.

`lerobot` is an optional dependency (`pip install vla-eval[lerobot]`). This
module only imports it lazily, inside methods, so the rest of the codebase
(API, CLI, baseline models) works without it installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.core.exceptions import ModelLoadError
from vla_eval.core.logging import get_logger
from vla_eval.models.base import Action, BaseVLAModel, Observation

logger = get_logger(__name__)

SUPPORTED_POLICIES = ("act", "diffusion", "vqbet", "tdmpc")


class LeRobotPolicyModel(BaseVLAModel):
    """Adapter around a `lerobot` policy class (e.g. ACTPolicy, DiffusionPolicy)."""

    name = "lerobot-policy"

    def __init__(
        self,
        policy_type: str = "act",
        *,
        action_dim: int = 7,
        device: str = "cpu",
        pretrained_path: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(device=device)
        if policy_type not in SUPPORTED_POLICIES:
            raise ModelLoadError(
                f"Unsupported lerobot policy_type '{policy_type}'. "
                f"Supported: {SUPPORTED_POLICIES}"
            )
        self.policy_type = policy_type
        self.action_dim = action_dim
        self.pretrained_path = pretrained_path
        self.policy_kwargs = policy_kwargs or {}
        self._policy: Any = None

    def load(self, path: str | None = None) -> None:
        try:
            from lerobot.common.policies.factory import make_policy
        except ImportError as exc:
            raise ModelLoadError(
                "The optional 'lerobot' dependency is required for LeRobotPolicyModel. "
                "Install with `pip install vla-eval[lerobot]`."
            ) from exc

        checkpoint = path or self.pretrained_path
        logger.info(
            "lerobot_policy_load_start", policy_type=self.policy_type, checkpoint=checkpoint
        )
        self._policy = make_policy(
            self.policy_type,
            pretrained_policy_name_or_path=checkpoint,
            **self.policy_kwargs,
        )
        self._policy.to(self.device)
        logger.info("lerobot_policy_load_complete", policy_type=self.policy_type)

    def _ensure_loaded(self) -> None:
        if self._policy is None:
            self.load()

    def reset(self) -> None:
        self._ensure_loaded()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def predict(self, observation: Observation) -> Action:
        import torch

        self._ensure_loaded()
        self.eval_mode()

        batch = {
            "observation.image": torch.from_numpy(observation.image).unsqueeze(0).to(self.device),
            "observation.state": torch.from_numpy(
                observation.state
                if observation.state is not None
                else np.zeros(1, dtype=np.float32)
            )
            .unsqueeze(0)
            .to(self.device),
        }
        with torch.no_grad():
            action = self._policy.select_action(batch)
        return Action(values=action.squeeze(0).cpu().numpy())

    def train_mode(self) -> None:
        self._ensure_loaded()
        self._policy.train()

    def eval_mode(self) -> None:
        self._ensure_loaded()
        self._policy.eval()

    def parameters(self) -> Any:
        self._ensure_loaded()
        return self._policy.parameters()

    def save(self, path: str) -> None:
        self._ensure_loaded()
        self._policy.save_pretrained(path)
