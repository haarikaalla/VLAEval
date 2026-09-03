"""SQLAlchemy ORM models for job tracking and the leaderboard."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from vla_eval.api.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobType(str, enum.Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    DATASET_DOWNLOAD = "dataset_download"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """A background unit of work (training run, evaluation run, dataset download)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_type: Mapped[JobType] = mapped_column(Enum(JobType), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, index=True
    )
    payload: Mapped[str] = mapped_column(Text, default="{}")  # JSON-encoded request params
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-encoded result
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class LeaderboardEntryDB(Base):
    """A persisted benchmark result contributing to the public leaderboard."""

    __tablename__ = "leaderboard_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    dataset_name: Mapped[str] = mapped_column(String(128), index=True)
    composite_score: Mapped[float] = mapped_column(Float)
    action_mse: Mapped[float] = mapped_column(Float)
    latency_p95_ms: Mapped[float] = mapped_column(Float)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_samples: Mapped[int] = mapped_column(Integer)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
