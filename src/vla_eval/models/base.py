"""Abstract interface that all VLA models/policies must implement.

Defining a single, narrow interface (`VLAPolicy`) lets the training,
evaluation, and API layers remain agnostic to the specific backbone
(OpenVLA, a LeRobot policy such as ACT/Diffusion Policy, or a lightweight
baseline used for CI/local dev).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class Observation:
    """A single-timestep multimodal observation fed to a VLA policy."""

    image: np.ndarray  # CHW float32 in [0, 1]
    state: np.ndarray | None = None
    instruction: str = ""


@dataclass
class Action:
    """A predicted (or ground-truth) action vector."""

    values: np.ndarray


class VLAPolicy(Protocol):
    """Structural interface for a Vision-Language-Action policy."""

    name: str
    action_dim: int

    def predict(self, observation: Observation) -> Action:
        """Predict a single action given an observation."""
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g. action chunk buffers) between episodes."""
        ...


class BaseVLAModel(ABC):
    """Base class providing shared bookkeeping for concrete VLA implementations."""

    name: str = "base"
    action_dim: int = 7

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @abstractmethod
    def predict(self, observation: Observation) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        """No-op by default; override for stateful/chunked policies."""
        return None

    @abstractmethod
    def train_mode(self) -> None:
        """Switch underlying model to training mode."""
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch underlying model to evaluation mode."""
        raise NotImplementedError

    def parameters(self) -> Any:
        """Return trainable parameters (for optimizer construction)."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
