"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vla_eval.api.models_db import JobStatus, JobType

# --- Datasets -----------------------------------------------------------------


class DatasetInfo(BaseModel):
    name: str
    hub_repo_id: str
    description: str
    task_type: str
    action_dim: int
    modalities: list[str]
    license: str
    tags: list[str]


class DatasetDownloadRequest(BaseModel):
    name: str = Field(..., description="Registered dataset name, e.g. 'pusht'.")


# --- Models ---------------------------------------------------------------


class ModelInfo(BaseModel):
    name: str


# --- Jobs -----------------------------------------------------------------


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_type: JobType
    status: JobStatus
    result: dict[str, Any] | None = None
    error_message: str | None = None
    mlflow_run_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


# --- Training ---------------------------------------------------------------


class TrainingRequest(BaseModel):
    model_name: str = Field(default="baseline-cnn", description="Registered model identifier.")
    dataset_name: str = Field(
        default="synthetic", description="Registered dataset name or 'synthetic'."
    )
    num_epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=2048)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    device: str = Field(default="cpu", pattern="^(cpu|cuda|mps)$")
    run_name: str | None = None


# --- Evaluation ---------------------------------------------------------------


class EvaluationRequest(BaseModel):
    model_name: str = Field(default="baseline-cnn")
    dataset_name: str = Field(default="synthetic")
    checkpoint_path: str | None = None
    device: str = Field(default="cpu", pattern="^(cpu|cuda|mps)$")
    max_samples: int | None = Field(default=None, ge=1)


class BenchmarkResultResponse(BaseModel):
    model_name: str
    dataset_name: str
    composite_score: float
    action_mse: float
    action_mae: float
    latency_mean_ms: float
    latency_p95_ms: float
    throughput_hz: float
    success_rate: float | None
    num_samples: int


# --- Leaderboard ---------------------------------------------------------------


class LeaderboardEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int = 0
    model_name: str
    dataset_name: str
    composite_score: float
    action_mse: float
    latency_p95_ms: float
    success_rate: float | None
    num_samples: int
    created_at: datetime


# --- Health / misc ---------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
