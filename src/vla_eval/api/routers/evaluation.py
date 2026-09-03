"""Evaluation/benchmark job submission and status endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status

from vla_eval.api.dependencies import ApiKeyAuth, DbSession
from vla_eval.api.models_db import Job, JobType
from vla_eval.api.schemas import EvaluationRequest, JobResponse
from vla_eval.api.services.job_manager import submit_evaluation_job

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


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
    summary="Submit an evaluation/benchmark job",
)
def create_evaluation_job(
    request: EvaluationRequest, db: DbSession, api_key: ApiKeyAuth
) -> JobResponse:
    job = submit_evaluation_job(db, request)
    return _to_job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="Get evaluation job status")
def get_evaluation_job(job_id: str, db: DbSession) -> JobResponse:
    job = db.query(Job).filter(Job.id == job_id, Job.job_type == JobType.EVALUATION).first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation job not found."
        )
    return _to_job_response(job)


@router.get("/jobs", response_model=list[JobResponse], summary="List evaluation jobs")
def list_evaluation_jobs(db: DbSession, limit: int = 50) -> list[JobResponse]:
    jobs = (
        db.query(Job)
        .filter(Job.job_type == JobType.EVALUATION)
        .order_by(Job.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [_to_job_response(j) for j in jobs]
