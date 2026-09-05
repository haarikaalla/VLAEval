"""A lightweight CNN+MLP baseline policy.

This model has no dependency on large pretrained backbones and trains in
seconds on CPU. It is used for:
  - Fast unit/integration tests and CI pipelines (no network/GPU required).
  - Local development and pipeline smoke-testing before committing to a full
    OpenVLA fine-tuning run.
  - A reference "floor" baseline on the leaderboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from vla_eval.models.base import Action, BaseVLAModel, Observation


class _CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselinePolicyNet(nn.Module):
    """CNN image encoder + state MLP -> concatenated -> action head."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.image_encoder = _CNNEncoder(out_dim=hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(max(state_dim, 1), hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([img_feat, state_feat], dim=-1))


class BaselineVLAModel(BaseVLAModel):
    """Baseline VLA-style policy conforming to the `VLAPolicy` interface.

    Ignores the language instruction (no text encoder) -- purely a
    vision+state -> action regressor. Useful as a fast, dependency-light
    reference point for benchmarking pipeline correctness.
    """

    name = "baseline-cnn"

    def __init__(self, state_dim: int = 7, action_dim: int = 7, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.net = BaselinePolicyNet(state_dim, action_dim).to(device)

    def predict(self, observation: Observation) -> Action:
        self.eval_mode()
        with torch.no_grad():
            image = torch.from_numpy(observation.image).unsqueeze(0).float().to(self.device)
            state = (
                observation.state
                if observation.state is not None
                else np.zeros(self.state_dim, dtype=np.float32)
            )
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            out = self.net(image, state_t)
        return Action(values=out.squeeze(0).cpu().numpy())

    def train_mode(self) -> None:
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()

    def parameters(self) -> Any:
        return self.net.parameters()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        # `weights_only=True` avoids arbitrary code execution from untrusted
        # checkpoint files (CVE-class Pickle deserialization issue).
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
