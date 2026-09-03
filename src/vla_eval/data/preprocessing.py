"""Preprocessing pipeline: convert raw downloaded datasets into training-ready
PyTorch `Dataset` objects with train/val splits and cached normalization stats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from vla_eval.core.exceptions import DatasetDownloadError
from vla_eval.core.logging import get_logger
from vla_eval.data.registry import DatasetSpec, get_dataset_spec
from vla_eval.data.transforms import NormalizationStats, resize_image, to_chw_float

logger = get_logger(__name__)


@dataclass
class PreprocessedSample:
    image: np.ndarray  # CHW float32 in [0, 1]
    state: np.ndarray  # (state_dim,) float32
    action: np.ndarray  # (action_dim,) float32
    instruction: str


class VLAEpisodeDataset(Dataset[dict[str, Any]]):
    """A minimal, framework-agnostic PyTorch-style `Dataset` over VLA episodes.

    Wraps a `lerobot.LeRobotDataset` (if available) or a synthetic fallback so
    that the rest of the training/evaluation pipeline can operate uniformly
    regardless of whether the optional `lerobot` dependency is installed.
    """

    def __init__(
        self,
        name: str,
        samples: list[PreprocessedSample],
        action_stats: NormalizationStats,
        spec: DatasetSpec,
    ) -> None:
        self.name = name
        self.samples = samples
        self.action_stats = action_stats
        self.spec = spec

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "image": sample.image,
            "state": sample.state,
            "action": sample.action,
            "instruction": sample.instruction,
        }

    def split(
        self, val_ratio: float = 0.1, seed: int = 42
    ) -> tuple[VLAEpisodeDataset, VLAEpisodeDataset]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.samples))
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        train = VLAEpisodeDataset(
            self.name, [self.samples[i] for i in train_idx], self.action_stats, self.spec
        )
        val = VLAEpisodeDataset(
            self.name, [self.samples[i] for i in val_idx], self.action_stats, self.spec
        )
        return train, val


def preprocess_dataset(
    name: str,
    *,
    cache_dir: str | Path = "./data/cache",
    processed_dir: str | Path = "./data/processed",
    image_size: tuple[int, int] = (224, 224),
    max_episodes: int | None = None,
) -> VLAEpisodeDataset:
    """Preprocess a downloaded dataset into a `VLAEpisodeDataset`.

    Tries to load via `lerobot.LeRobotDataset` first; if unavailable, raises
    a clear error instructing the caller to install the optional dependency
    or use synthetic data for local development (see `make_synthetic_dataset`).
    """
    spec = get_dataset_spec(name)
    processed_path = Path(processed_dir) / name
    processed_path.mkdir(parents=True, exist_ok=True)

    try:
        from vla_eval.data.datasets import load_lerobot_dataset

        raw = load_lerobot_dataset(name, cache_dir=cache_dir)
    except ImportError as exc:
        raise DatasetDownloadError(
            f"Cannot preprocess '{name}': the optional 'lerobot' dependency is not "
            "installed. Install with `pip install vla-eval[lerobot]`, or use "
            "`make_synthetic_dataset` for local development/testing."
        ) from exc

    samples: list[PreprocessedSample] = []
    actions: list[np.ndarray] = []

    n = len(raw) if max_episodes is None else min(len(raw), max_episodes)
    for i in range(n):
        item = raw[i]
        image = np.asarray(item.get("observation.image", item.get("observation.images.top")))
        state = np.asarray(item.get("observation.state", np.zeros(1, dtype=np.float32)))
        action = np.asarray(item["action"], dtype=np.float32)
        instruction = item.get("language_instruction", spec.description)

        image = resize_image(image, image_size)
        image = to_chw_float(image)

        samples.append(
            PreprocessedSample(
                image=image, state=state.astype(np.float32), action=action, instruction=instruction
            )
        )
        actions.append(action)

    action_stats = NormalizationStats.from_samples(np.stack(actions, axis=0))
    _save_stats(processed_path / "action_stats.json", action_stats)

    logger.info("dataset_preprocess_complete", dataset=name, num_samples=len(samples))
    return VLAEpisodeDataset(name, samples, action_stats, spec)


def make_synthetic_dataset(
    name: str = "synthetic",
    num_samples: int = 256,
    image_size: tuple[int, int] = (224, 224),
    action_dim: int = 7,
    state_dim: int = 7,
    seed: int = 42,
) -> VLAEpisodeDataset:
    """Generate a synthetic dataset for local development, unit tests, and CI.

    This avoids requiring network access or the `lerobot` package for tests
    that only need to exercise the training/evaluation *pipeline*.
    """
    rng = np.random.default_rng(seed)
    samples = []
    actions = []
    for _ in range(num_samples):
        image = rng.random((3, *image_size), dtype=np.float32)
        state = rng.normal(size=state_dim).astype(np.float32)
        action = rng.normal(size=action_dim).astype(np.float32)
        samples.append(
            PreprocessedSample(
                image=image, state=state, action=action, instruction="pick up the object"
            )
        )
        actions.append(action)

    action_stats = NormalizationStats.from_samples(np.stack(actions, axis=0))
    spec = DatasetSpec(
        name=name,
        hub_repo_id="synthetic/local",
        source=get_dataset_spec("pusht").source,
        description="Synthetic dataset for local dev/testing",
        task_type="manipulation",
        action_dim=action_dim,
    )
    return VLAEpisodeDataset(name, samples, action_stats, spec)


def _save_stats(path: Path, stats: NormalizationStats) -> None:
    path.write_text(
        json.dumps({"mean": stats.mean.tolist(), "std": stats.std.tolist()}, indent=2),
        encoding="utf-8",
    )


def load_stats(path: str | Path) -> NormalizationStats:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationStats(mean=np.array(data["mean"]), std=np.array(data["std"]))
