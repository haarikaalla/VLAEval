"""Unit tests for leaderboard ranking logic."""

from __future__ import annotations

from vla_eval.evaluation.benchmark import BenchmarkResult
from vla_eval.evaluation.leaderboard import build_leaderboard
from vla_eval.evaluation.metrics import ActionMetrics, LatencyMetrics


def _make_result(model_name: str, score: float, dataset_name: str = "pusht") -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        dataset_name=dataset_name,
        action_metrics=ActionMetrics(mse=0.1, mae=0.1, max_error=0.1, per_dim_mse=[0.1]),
        latency=LatencyMetrics(
            mean_ms=5.0, p50_ms=5.0, p95_ms=6.0, p99_ms=7.0, throughput_hz=200.0
        ),
        success_rate=None,
        composite_score=score,
        num_samples=100,
    )


def test_build_leaderboard_sorts_descending() -> None:
    results = [
        _make_result("model-a", 50.0),
        _make_result("model-b", 90.0),
        _make_result("model-c", 70.0),
    ]
    leaderboard = build_leaderboard(results)
    assert [e.model_name for e in leaderboard] == ["model-b", "model-c", "model-a"]
    assert [e.rank for e in leaderboard] == [1, 2, 3]


def test_build_leaderboard_filters_by_dataset() -> None:
    results = [
        _make_result("model-a", 50.0, dataset_name="pusht"),
        _make_result("model-b", 90.0, dataset_name="aloha_sim_insertion_human"),
    ]
    leaderboard = build_leaderboard(results, dataset_filter="pusht")
    assert len(leaderboard) == 1
    assert leaderboard[0].model_name == "model-a"
