"""Public leaderboard endpoints (read-only, no auth required)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from vla_eval.api.dependencies import DbSession
from vla_eval.api.models_db import LeaderboardEntryDB
from vla_eval.api.schemas import LeaderboardEntryResponse

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=list[LeaderboardEntryResponse], summary="Get the ranked leaderboard")
def get_leaderboard(
    db: DbSession,
    dataset_name: str | None = Query(default=None, description="Filter by dataset name."),
    limit: int = Query(default=100, le=500),
) -> list[LeaderboardEntryResponse]:
    query = db.query(LeaderboardEntryDB)
    if dataset_name:
        query = query.filter(LeaderboardEntryDB.dataset_name == dataset_name)

    entries = query.order_by(LeaderboardEntryDB.composite_score.desc()).limit(limit).all()
    return [
        LeaderboardEntryResponse(
            rank=i + 1,
            model_name=e.model_name,
            dataset_name=e.dataset_name,
            composite_score=e.composite_score,
            action_mse=e.action_mse,
            latency_p95_ms=e.latency_p95_ms,
            success_rate=e.success_rate,
            num_samples=e.num_samples,
            created_at=e.created_at,
        )
        for i, e in enumerate(entries)
    ]
