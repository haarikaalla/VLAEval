"""Liveness/readiness endpoints (unauthenticated, used by orchestrators)."""

from __future__ import annotations

from fastapi import APIRouter

from vla_eval.__about__ import __version__
from vla_eval.api.schemas import HealthResponse
from vla_eval.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
def healthz() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok", app_name=settings.app_name, version=__version__, environment=settings.app_env
    )


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
def readyz() -> HealthResponse:
    """Checks that the database is reachable; returns 200 only if ready."""
    from sqlalchemy import text

    from vla_eval.api.db import get_engine

    settings = get_settings()
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return HealthResponse(
        status="ready",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )
