"""Quantitative metrics for VLA policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionMetrics:
    mse: float
    mae: float
    max_error: float
    per_dim_mse: list[float]


def action_prediction_metrics(predictions: np.ndarray, targets: np.ndarray) -> ActionMetrics:
    """Compute offline (open-loop) action-prediction error metrics.

    Args:
        predictions: (N, action_dim) predicted actions.
        targets: (N, action_dim) ground-truth actions.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch between predictions {predictions.shape} and targets {targets.shape}"
        )
    error = predictions - targets
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    max_error = float(np.max(np.abs(error)))
    per_dim_mse = np.mean(error**2, axis=0).tolist()
    return ActionMetrics(mse=mse, mae=mae, max_error=max_error, per_dim_mse=per_dim_mse)


def success_rate(successes: list[bool]) -> float:
    """Fraction of episodes marked successful (closed-loop rollout evaluation)."""
    if not successes:
        return 0.0
    return sum(successes) / len(successes)


@dataclass
class LatencyMetrics:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_hz: float


def latency_metrics(latencies_sec: list[float]) -> LatencyMetrics:
    """Summarize per-inference latency samples (in seconds) into standard percentiles."""
    if not latencies_sec:
        return LatencyMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array(latencies_sec) * 1000.0  # to ms
    mean_ms = float(np.mean(arr))
    return LatencyMetrics(
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_hz=float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    )


def composite_score(
    action_metrics: ActionMetrics,
    latency: LatencyMetrics,
    success: float | None = None,
    *,
    weight_success: float = 0.6,
    weight_accuracy: float = 0.3,
    weight_speed: float = 0.1,
) -> float:
    """Combine sub-metrics into a single leaderboard score in [0, 100].

    Accuracy is transformed via `1 / (1 + mse)` to map lower-is-better MSE
    onto a higher-is-better [0, 1] scale. Speed is normalized against a
    10ms/inference reference latency. If `success` (closed-loop rollout
    success rate) is unavailable, its weight is redistributed to accuracy.
    """
    accuracy_score = 1.0 / (1.0 + action_metrics.mse)
    speed_score = min(1.0, 10.0 / max(latency.mean_ms, 1e-6))

    if success is None:
        weight_accuracy += weight_success
        weight_success = 0.0
        success = 0.0

    score = weight_success * success + weight_accuracy * accuracy_score + weight_speed * speed_score
    return round(score * 100, 2)
