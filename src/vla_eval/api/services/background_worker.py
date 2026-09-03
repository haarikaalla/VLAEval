"""Standalone worker process: polls the `jobs` table and executes pending
training/evaluation jobs. Runs independently of the API process, allowing
compute-heavy work to be scaled out horizontally (see `docker-compose.yml`,
service `worker`).

Usage:
    python -m vla_eval.api.services.background_worker
"""

from __future__ import annotations

import json
import signal
import time
from types import FrameType

from vla_eval.api.db import init_db, new_session
from vla_eval.api.models_db import JobType
from vla_eval.api.schemas import EvaluationRequest, TrainingRequest
from vla_eval.api.services.job_manager import (
    _execute_evaluation_job,
    _execute_training_job,
    claim_pending_job,
)
from vla_eval.core.config import get_settings
from vla_eval.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 3.0
_shutdown_requested = False


def _handle_shutdown(signum: int, frame: FrameType | None) -> None:
    global _shutdown_requested
    logger.info("worker_shutdown_signal_received", signum=signum)
    _shutdown_requested = True


def run_worker_loop() -> None:
    configure_logging()
    get_settings()
    init_db()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    logger.info("worker_started", poll_interval_seconds=POLL_INTERVAL_SECONDS)

    while not _shutdown_requested:
        db = new_session()
        try:
            job = claim_pending_job(db)
        finally:
            db.close()

        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        payload = json.loads(job.payload)
        logger.info("worker_claimed_job", job_id=job.id, job_type=job.job_type)

        if job.job_type == JobType.TRAINING:
            _execute_training_job(job.id, TrainingRequest(**payload))
        elif job.job_type == JobType.EVALUATION:
            _execute_evaluation_job(job.id, EvaluationRequest(**payload))
        else:
            logger.warning("worker_unknown_job_type", job_id=job.id, job_type=job.job_type)

    logger.info("worker_stopped")


if __name__ == "__main__":
    run_worker_loop()
