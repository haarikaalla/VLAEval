"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.evaluation.metrics import (
    action_prediction_metrics,
    composite_score,
    latency_metrics,
    success_rate,
)


def test_action_prediction_metrics_perfect_prediction() -> None:
    targets = np.random.default_rng(0).normal(size=(10, 4))
    result = action_prediction_metrics(targets.copy(), targets)
    assert result.mse == pytest.approx(0.0, abs=1e-10)
    assert result.mae == pytest.approx(0.0, abs=1e-10)


def test_action_prediction_metrics_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        action_prediction_metrics(np.zeros((5, 3)), np.zeros((5, 4)))


def test_success_rate() -> None:
    assert success_rate([True, True, False, False]) == 0.5
    assert success_rate([]) == 0.0


def test_latency_metrics_basic() -> None:
    latencies = [0.01, 0.02, 0.03, 0.04, 0.05]
    metrics = latency_metrics(latencies)
    assert metrics.mean_ms == pytest.approx(30.0, rel=1e-3)
    assert metrics.throughput_hz > 0


def test_composite_score_bounds() -> None:
    from vla_eval.evaluation.metrics import ActionMetrics, LatencyMetrics

    action = ActionMetrics(mse=0.0, mae=0.0, max_error=0.0, per_dim_mse=[0.0])
    latency = LatencyMetrics(mean_ms=1.0, p50_ms=1.0, p95_ms=1.0, p99_ms=1.0, throughput_hz=1000.0)
    score = composite_score(action, latency, success=1.0)
    assert 0 <= score <= 100
