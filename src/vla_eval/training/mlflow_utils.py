"""MLflow experiment tracking helpers.

Centralizes MLflow setup so training, evaluation, and the API layer log
metrics/artifacts consistently to the same tracking server and experiment.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from vla_eval.core.config import get_settings
from vla_eval.core.logging import get_logger

logger = get_logger(__name__)


def configure_mlflow(experiment_name: str | None = None) -> str:
    """Point the MLflow client at the configured tracking server/experiment.

    Returns:
        The resolved experiment name.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp_name = experiment_name or settings.mlflow_experiment_name
    mlflow.set_experiment(exp_name)
    return exp_name


@contextmanager
def mlflow_run(
    run_name: str,
    *,
    experiment_name: str | None = None,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Context manager wrapping an MLflow run with standard tags/params logged.

    Example:
        with mlflow_run("openvla-lora-pusht", params=cfg_dict) as run:
            mlflow.log_metric("loss", loss_value, step=step)
    """
    configure_mlflow(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        if params:
            # MLflow rejects param values >500 chars; stringify defensively.
            safe_params = {k: str(v)[:500] for k, v in params.items()}
            mlflow.log_params(safe_params)
        logger.info("mlflow_run_started", run_id=run.info.run_id, run_name=run_name)
        yield run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    mlflow.log_artifact(local_path, artifact_path=artifact_path)
