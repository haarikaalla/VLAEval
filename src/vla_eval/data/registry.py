"""Registry of publicly available robotics datasets supported out-of-the-box.

Data provenance
---------------
All non-synthetic datasets are downloaded at runtime from the Hugging Face
Hub under the ``lerobot`` organization (https://huggingface.co/lerobot), the
standardized robot-learning dataset collection maintained by the Hugging Face
LeRobot project. They contain synchronized camera frames, proprioceptive
state, and action trajectories recorded on physical robots and in simulators.

``berkeley_autolab_ur5`` and ``bridge_orig`` are real-robot recordings
originally released as part of Open X-Embodiment
(https://robotics-transformer-x.github.io/) and re-published in LeRobot
format. No dataset content is vendored in this repository; downloads are
cached under ``./data/cache`` and can be pinned to a Hub revision.

Adding a new dataset only requires registering a `DatasetSpec` entry --
no code changes are needed elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DatasetSource(str, Enum):
    LEROBOT_HUB = "lerobot_hub"  # LeRobotDataset hosted on the HF Hub
    HF_DATASETS = "hf_datasets"  # Generic HF `datasets` repo


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a registered dataset."""

    name: str
    """Unique short identifier used throughout the CLI/API, e.g. 'pusht'."""

    hub_repo_id: str
    """Hugging Face Hub repo id, e.g. 'lerobot/pusht'."""

    source: DatasetSource
    description: str
    task_type: str
    """High-level task category, e.g. 'manipulation', 'navigation'."""
    action_dim: int
    """Dimensionality of the action space."""
    modalities: tuple[str, ...] = ("image", "state")
    license: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)


# Curated set of well-known, publicly available robot learning datasets.
# This list is intentionally small and high-quality; extend via
# `register_dataset` for organization-specific datasets.
_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    """Register (or override) a dataset spec in the global registry."""
    _REGISTRY[spec.name] = spec


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset spec by its registry name.

    Raises:
        DatasetNotFoundError: if `name` is not registered.
    """
    from vla_eval.core.exceptions import DatasetNotFoundError

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise DatasetNotFoundError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        ) from exc


def list_datasets() -> list[DatasetSpec]:
    """Return all registered dataset specs, sorted by name."""
    return [spec for _, spec in sorted(_REGISTRY.items())]


# --- Curated default registry -------------------------------------------------

register_dataset(
    DatasetSpec(
        name="pusht",
        hub_repo_id="lerobot/pusht",
        source=DatasetSource.LEROBOT_HUB,
        description="Push-T: pushing a T-shaped block to a target pose (simulation).",
        task_type="manipulation",
        action_dim=2,
        modalities=("image", "state"),
        license="mit",
        tags=("simulation", "single-task", "benchmark"),
    )
)

register_dataset(
    DatasetSpec(
        name="aloha_sim_insertion_human",
        hub_repo_id="lerobot/aloha_sim_insertion_human",
        source=DatasetSource.LEROBOT_HUB,
        description="ALOHA bimanual peg insertion, human-teleoperated demonstrations (simulation).",
        task_type="manipulation",
        action_dim=14,
        modalities=("image", "state"),
        license="mit",
        tags=("simulation", "bimanual", "benchmark"),
    )
)

register_dataset(
    DatasetSpec(
        name="aloha_sim_transfer_cube_human",
        hub_repo_id="lerobot/aloha_sim_transfer_cube_human",
        source=DatasetSource.LEROBOT_HUB,
        description="ALOHA bimanual cube transfer, human-teleoperated demonstrations (simulation).",
        task_type="manipulation",
        action_dim=14,
        modalities=("image", "state"),
        license="mit",
        tags=("simulation", "bimanual", "benchmark"),
    )
)

register_dataset(
    DatasetSpec(
        name="xarm_lift_medium",
        hub_repo_id="lerobot/xarm_lift_medium",
        source=DatasetSource.LEROBOT_HUB,
        description="xArm lifting task of medium difficulty (simulation).",
        task_type="manipulation",
        action_dim=4,
        modalities=("image", "state"),
        license="mit",
        tags=("simulation", "single-task"),
    )
)

register_dataset(
    DatasetSpec(
        name="berkeley_autolab_ur5",
        hub_repo_id="lerobot/berkeley_autolab_ur5",
        source=DatasetSource.LEROBOT_HUB,
        description="UR5 tabletop manipulation demonstrations from the Berkeley AUTOLAB (real robot).",
        task_type="manipulation",
        action_dim=7,
        modalities=("image", "state"),
        license="cc-by-4.0",
        tags=("real-robot", "open-x-embodiment"),
    )
)

register_dataset(
    DatasetSpec(
        name="bridge_orig",
        hub_repo_id="lerobot/bridge_orig",
        source=DatasetSource.LEROBOT_HUB,
        description="BridgeData V2: diverse real-robot manipulation across many kitchens/environments.",
        task_type="manipulation",
        action_dim=7,
        modalities=("image", "state"),
        license="cc-by-4.0",
        tags=("real-robot", "open-x-embodiment", "large-scale"),
    )
)
