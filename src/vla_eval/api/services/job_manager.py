"""Background job orchestration for training and evaluation runs.

Jobs are persisted in the database (`Job` rows) so their status survives
API process restarts and can be observed by any number of worker processes.
Two execution modes are supported simultaneously:

1. **In-process** (default, good for local dev / small deployments): the API
   process itself submits jobs to a bounded `ThreadPoolExecutor`.
2. **Out-of-process worker** (`vla_eval.api.services.background_worker`):
   a separate container polls the `jobs` table for `PENDING` rows and
   executes them, allowing training/evaluation load to be scaled
   independently from the API tier.

Both paths call the same `_execute_training_job` / `_execute_evaluation_job`
functions, ensuring identical behavior regardless of where a job runs.
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from vla_eval.api.db import new_session
from vla_eval.api.models_db import Job, JobStatus, JobType, LeaderboardEntryDB
from vla_eval.api.schemas import EvaluationRequest, TrainingRequest
from vla_eval.core.logging import get_logger

logger = get_logger(__name__)

# Bounded pool: training/evaluation jobs are CPU/GPU heavy, so we cap
# in-process concurrency to avoid oversubscribing the host. Tune via env
# in production, or run dedicated worker containers instead.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vla-eval-job")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_job(db: Session, job_type: JobType, payload: dict) -> Job:
    job = Job(job_type=job_type, status=JobStatus.PENDING, payload=json.dumps(payload))
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def submit_training_job(db: Session, request: TrainingRequest) -> Job:
    job = create_job(db, JobType.TRAINING, request.model_dump())
    _executor.submit(_execute_training_job, job.id, request)
    return job


def submit_evaluation_job(db: Session, request: EvaluationRequest) -> Job:
    job = create_job(db, JobType.EVALUATION, request.model_dump())
    _executor.submit(_execute_evaluation_job, job.id, request)
    return job


def claim_pending_job(db: Session) -> Job | None:
    """Atomically claim the oldest PENDING job by transitioning it to RUNNING.

    Used by the standalone worker's polling loop. Uses a single UPDATE with a
    WHERE clause to minimize the race window between multiple workers.
    """
    query = db.query(Job).filter(Job.status == JobStatus.PENDING).order_by(Job.created_at.asc())
    # `SELECT ... FOR UPDATE SKIP LOCKED` allows multiple worker processes to
    # safely compete for jobs without duplicating work. SQLite (used for local
    # dev) doesn't support row locking, so we fall back to the plain query
    # there and rely on the conditional UPDATE below for safety.
    if db.bind is not None and db.bind.dialect.name != "sqlite":
        query = query.with_for_update(skip_locked=True)
    job = query.first()
    if job is None:
        return None

    result = db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status == JobStatus.PENDING)
        .values(status=JobStatus.RUNNING, started_at=_utcnow())
    )
    db.commit()
    if cast(CursorResult, result).rowcount == 0:
        return None  # another worker claimed it first
    db.refresh(job)
    return job


def _mark_running(db: Session, job_id: str) -> None:
    db.query(Job).filter(Job.id == job_id).update(
        {"status": JobStatus.RUNNING, "started_at": _utcnow()}
    )
    db.commit()


def _mark_succeeded(
    db: Session, job_id: str, result: dict, mlflow_run_id: str | None = None
) -> None:
    db.query(Job).filter(Job.id == job_id).update(
        {
            "status": JobStatus.SUCCEEDED,
            "result": json.dumps(result, default=str),
            "finished_at": _utcnow(),
            "mlflow_run_id": mlflow_run_id,
        }
    )
    db.commit()


def _mark_failed(db: Session, job_id: str, error: str) -> None:
    db.query(Job).filter(Job.id == job_id).update(
        {"status": JobStatus.FAILED, "error_message": error, "finished_at": _utcnow()}
    )
    db.commit()


def _execute_training_job(job_id: str, request: TrainingRequest) -> None:
    """Run a training job end-to-end; safe to call from any thread/process."""
    db = new_session()
    try:
        _mark_running(db, job_id)
        logger.info("training_job_started", job_id=job_id, model=request.model_name)

        from vla_eval.core.config import get_settings
        from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
        from vla_eval.models.registry import create_model
        from vla_eval.training.callbacks import Callback, EarlyStopping, ModelCheckpoint
        from vla_eval.training.trainer import Trainer, TrainingConfig
        from vla_eval.utils.device import resolve_device

        device = resolve_device(request.device)
        dataset = (
            make_synthetic_dataset()
            if request.dataset_name == "synthetic"
            else preprocess_dataset(request.dataset_name)
        )
        policy = create_model(request.model_name, device=device, action_dim=dataset.spec.action_dim)
        config = TrainingConfig(
            run_name=request.run_name or f"job-{job_id[:8]}",
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            device=device,
            checkpoint_dir=f"{get_settings().checkpoint_root}/{job_id}",
        )
        callbacks: list[Callback] = [
            EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience),
            ModelCheckpoint(checkpoint_dir=config.checkpoint_dir, monitor="val_loss"),
        ]
        trainer = Trainer(policy, config, callbacks=callbacks)
        result = trainer.fit(dataset)

        _mark_succeeded(db, job_id, result)
        logger.info("training_job_succeeded", job_id=job_id)
    except Exception as exc:  # noqa: BLE001 - report all failures to the job record
        logger.error(
            "training_job_failed", job_id=job_id, error=str(exc), traceback=traceback.format_exc()
        )
        _mark_failed(db, job_id, str(exc))
    finally:
        db.close()


def _execute_evaluation_job(job_id: str, request: EvaluationRequest) -> None:
    """Run an evaluation job end-to-end and persist a leaderboard entry on success."""
    db = new_session()
    try:
        _mark_running(db, job_id)
        logger.info("evaluation_job_started", job_id=job_id, model=request.model_name)

        from vla_eval.core.config import get_settings
        from vla_eval.data.preprocessing import make_synthetic_dataset, preprocess_dataset
        from vla_eval.evaluation.benchmark import run_offline_benchmark
        from vla_eval.evaluation.report import generate_markdown_report
        from vla_eval.models.registry import create_model
        from vla_eval.utils.device import resolve_device

        device = resolve_device(request.device)
        dataset = (
            make_synthetic_dataset()
            if request.dataset_name == "synthetic"
            else preprocess_dataset(request.dataset_name)
        )
        policy = create_model(request.model_name, device=device, action_dim=dataset.spec.action_dim)
        if request.checkpoint_path:
            policy.load(request.checkpoint_path)

        result = run_offline_benchmark(policy, dataset, max_samples=request.max_samples)
        report_path = generate_markdown_report(result, f"{get_settings().report_root}/{job_id}")

        entry = LeaderboardEntryDB(
            model_name=result.model_name,
            dataset_name=result.dataset_name,
            composite_score=result.composite_score,
            action_mse=result.action_metrics.mse,
            latency_p95_ms=result.latency.p95_ms,
            success_rate=result.success_rate,
            num_samples=result.num_samples,
            job_id=job_id,
        )
        db.add(entry)
        db.commit()

        result_dict = {
            "composite_score": result.composite_score,
            "action_mse": result.action_metrics.mse,
            "latency_p95_ms": result.latency.p95_ms,
            "report_path": str(report_path),
        }
        _mark_succeeded(db, job_id, result_dict)
        logger.info("evaluation_job_succeeded", job_id=job_id, score=result.composite_score)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "evaluation_job_failed", job_id=job_id, error=str(exc), traceback=traceback.format_exc()
        )
        _mark_failed(db, job_id, str(exc))
    finally:
        db.close()
