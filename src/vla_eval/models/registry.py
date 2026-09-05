"""Factory/registry for constructing `VLAPolicy` implementations by name.

This is the single place that maps a string identifier (as used in Hydra
configs, the CLI, and the API) to a concrete model class, keeping the rest
of the codebase decoupled from specific implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vla_eval.core.exceptions import ModelNotFoundError
from vla_eval.models.base import BaseVLAModel

_MODEL_FACTORIES: dict[str, Callable[..., BaseVLAModel]] = {}


def register_model(name: str, factory: Callable[..., BaseVLAModel]) -> None:
    _MODEL_FACTORIES[name] = factory


def create_model(name: str, **kwargs: Any) -> BaseVLAModel:
    """Instantiate a registered model by name.

    Raises:
        ModelNotFoundError: if `name` is not registered.
    """
    if name not in _MODEL_FACTORIES:
        available = ", ".join(sorted(_MODEL_FACTORIES))
        raise ModelNotFoundError(f"Unknown model '{name}'. Available models: {available}")
    return _MODEL_FACTORIES[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_MODEL_FACTORIES)


def _register_defaults() -> None:
    from vla_eval.models.baseline import BaselineVLAModel

    register_model("baseline-cnn", BaselineVLAModel)

    def _openvla_factory(**kwargs: Any) -> BaseVLAModel:
        from vla_eval.models.openvla import OpenVLAModel

        return OpenVLAModel(**kwargs)

    register_model("openvla", _openvla_factory)

    def _lerobot_factory(**kwargs: Any) -> BaseVLAModel:
        from vla_eval.models.lerobot_policies import LeRobotPolicyModel

        return LeRobotPolicyModel(**kwargs)

    register_model("lerobot-act", lambda **kw: _lerobot_factory(policy_type="act", **kw))
    register_model(
        "lerobot-diffusion", lambda **kw: _lerobot_factory(policy_type="diffusion", **kw)
    )


_register_defaults()
