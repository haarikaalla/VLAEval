"""Training job submission and status endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status

from vla_eval.api.dependencies import ApiKeyAuth, DbSession
from vla_eval.api.models_db import Job, JobType
from vla_eval.api.schemas import JobResponse, TrainingRequest
from vla_eval.api.services.job_manager import submit_training_job

router = APIRouter(prefix="/training", tags=["training"])


def _to_job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        result=json.loads(job.result) if job.result else None,
        error_message=job.error_message,
        mlflow_run_id=job.mlflow_run_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post(
    "/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a training job",
)
def create_training_job(
    request: TrainingRequest, db: DbSession, api_key: ApiKeyAuth
) -> JobResponse:
    job = submit_training_job(db, request)
    return _to_job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="Get training job status")
def get_training_job(job_id: str, db: DbSession) -> JobResponse:
    job = db.query(Job).filter(Job.id == job_id, Job.job_type == JobType.TRAINING).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found.")
    return _to_job_response(job)


@router.get("/jobs", response_model=list[JobResponse], summary="List training jobs")
def list_training_jobs(db: DbSession, limit: int = 50) -> list[JobResponse]:
    jobs = (
        db.query(Job)
        .filter(Job.job_type == JobType.TRAINING)
        .order_by(Job.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [_to_job_response(j) for j in jobs]
