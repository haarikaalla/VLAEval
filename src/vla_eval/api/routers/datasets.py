"""Dataset discovery and download endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status

from vla_eval.api.dependencies import ApiKeyAuth
from vla_eval.api.schemas import DatasetDownloadRequest, DatasetInfo
from vla_eval.core.exceptions import DatasetDownloadError
from vla_eval.core.logging import get_logger

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = get_logger(__name__)


@router.get("", response_model=list[DatasetInfo], summary="List registered public datasets")
def list_datasets() -> list[DatasetInfo]:
    from vla_eval.data.registry import list_datasets as _list_datasets

    return [
        DatasetInfo(
            name=s.name,
            hub_repo_id=s.hub_repo_id,
            description=s.description,
            task_type=s.task_type,
            action_dim=s.action_dim,
            modalities=list(s.modalities),
            license=s.license,
            tags=list(s.tags),
        )
        for s in _list_datasets()
    ]


@router.get("/{name}", response_model=DatasetInfo, summary="Get metadata for a single dataset")
def get_dataset(name: str) -> DatasetInfo:
    from vla_eval.data.registry import get_dataset_spec

    s = get_dataset_spec(name)
    return DatasetInfo(
        name=s.name,
        hub_repo_id=s.hub_repo_id,
        description=s.description,
        task_type=s.task_type,
        action_dim=s.action_dim,
        modalities=list(s.modalities),
        license=s.license,
        tags=list(s.tags),
    )


@router.post(
    "/download",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger an asynchronous dataset download",
)
def download_dataset(
    request: DatasetDownloadRequest, background_tasks: BackgroundTasks, api_key: ApiKeyAuth
) -> dict:
    from vla_eval.core.config import get_settings
    from vla_eval.data.datasets import download_dataset as _download

    settings = get_settings()

    def _run() -> None:
        try:
            _download(request.name, hf_token=settings.hf_token)
        except DatasetDownloadError as exc:
            logger.error("dataset_download_task_failed", dataset=request.name, error=str(exc))

    background_tasks.add_task(_run)
    return {"message": f"Download started for dataset '{request.name}'", "dataset": request.name}
