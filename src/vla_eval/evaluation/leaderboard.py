"""Leaderboard aggregation logic (pure functions, framework-agnostic).

The FastAPI layer persists `LeaderboardEntry` rows to the database; this
module only implements the ranking/aggregation logic so it can be unit
tested without a database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from vla_eval.evaluation.benchmark import BenchmarkResult


@dataclass
class LeaderboardEntry:
    rank: int
    model_name: str
    dataset_name: str
    composite_score: float
    action_mse: float
    latency_p95_ms: float
    success_rate: float | None
    num_samples: int
    created_at: str


def result_to_entry(result: BenchmarkResult, *, rank: int = 0) -> LeaderboardEntry:
    return LeaderboardEntry(
        rank=rank,
        model_name=result.model_name,
        dataset_name=result.dataset_name,
        composite_score=result.composite_score,
        action_mse=result.action_metrics.mse,
        latency_p95_ms=result.latency.p95_ms,
        success_rate=result.success_rate,
        num_samples=result.num_samples,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_leaderboard(
    results: list[BenchmarkResult], *, dataset_filter: str | None = None
) -> list[LeaderboardEntry]:
    """Rank a set of benchmark results by composite score (descending).

    Args:
        results: Raw benchmark results, potentially across multiple datasets.
        dataset_filter: If provided, restrict the leaderboard to a single dataset.
    """
    filtered = [r for r in results if dataset_filter is None or r.dataset_name == dataset_filter]
    ranked = sorted(filtered, key=lambda r: r.composite_score, reverse=True)
    return [result_to_entry(r, rank=i + 1) for i, r in enumerate(ranked)]


def leaderboard_to_dicts(entries: list[LeaderboardEntry]) -> list[dict]:
    return [asdict(e) for e in entries]
