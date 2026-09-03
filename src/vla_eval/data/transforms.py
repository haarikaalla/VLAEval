"""Image/state/action preprocessing transforms for VLA model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    """Per-dimension mean/std (or min/max) statistics for normalization."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.clip(self.std, a_min=1e-6, a_max=None)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> NormalizationStats:
        """Compute stats from an array of shape (N, D)."""
        return cls(mean=samples.mean(axis=0), std=samples.std(axis=0))


def resize_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize an HWC uint8 image to `size` using OpenCV (bilinear interpolation)."""
    import cv2

    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def to_chw_float(image: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 image in [0, 255] to CHW float32 in [0, 1]."""
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def normalize_action(action: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Normalize an action vector using precomputed dataset statistics."""
    return stats.normalize(action)


def unnormalize_action(action: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Invert `normalize_action`, mapping model outputs back to physical units."""
    return stats.unnormalize(action)
