"""Evaluation/benchmark job submission and status endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from vla_eval.api.dependencies import AppSettings, DbSession, WriteAuth
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
    request: EvaluationRequest, db: DbSession, principal: WriteAuth
) -> JobResponse:
    job = submit_evaluation_job(db, request, created_by=principal.subject)
    return _to_job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="Get evaluation job status")
def get_evaluation_job(job_id: str, db: DbSession, principal: WriteAuth) -> JobResponse:
    job = db.query(Job).filter(Job.id == job_id, Job.job_type == JobType.EVALUATION).first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation job not found."
        )
    return _to_job_response(job)


@router.get("/jobs/{job_id}/report", summary="Download a completed evaluation report")
def download_evaluation_report(
    job_id: str,
    db: DbSession,
    principal: WriteAuth,
    settings: AppSettings,
) -> FileResponse:
    job = db.query(Job).filter(Job.id == job_id, Job.job_type == JobType.EVALUATION).first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation job not found."
        )
    result = json.loads(job.result) if job.result else {}
    report_value = result.get("report_path")
    if not isinstance(report_value, str):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The evaluation report is not available yet.",
        )
    report_path = Path(report_value).resolve()
    report_root = Path(settings.report_root).resolve()
    if not report_path.is_relative_to(report_root) or not report_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report file not found.")
    return FileResponse(
        report_path,
        media_type="text/markdown",
        filename=f"vla-eval-{job_id[:8]}-report.md",
    )


@router.get("/jobs", response_model=list[JobResponse], summary="List evaluation jobs")
def list_evaluation_jobs(db: DbSession, principal: WriteAuth, limit: int = 50) -> list[JobResponse]:
    jobs = (
        db.query(Job)
        .filter(Job.job_type == JobType.EVALUATION)
        .order_by(Job.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [_to_job_response(j) for j in jobs]
