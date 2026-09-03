"""Download and load public robotics datasets.

Two backends are supported:

1. `lerobot` (preferred, optional dependency): loads datasets natively as a
   `LeRobotDataset`, giving direct access to synchronized image/state/action
   tensors, episode boundaries, and camera streams.
2. `huggingface_hub` snapshot download (fallback): downloads the raw parquet
   / video shards to local disk so they can be inspected or converted, even
   when `lerobot` is not installed (e.g. in lightweight CI environments).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from vla_eval.core.exceptions import DatasetDownloadError
from vla_eval.core.logging import get_logger
from vla_eval.data.registry import DatasetSpec, get_dataset_spec

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("./data/cache")


@dataclass
class DownloadResult:
    dataset_name: str
    hub_repo_id: str
    local_path: Path
    backend: str
    num_bytes: int | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _snapshot_download(
    repo_id: str, local_dir: Path, hf_token: str | None, revision: str | None = None
) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=hf_token,
        revision=revision,
    )
    return Path(path)


def download_dataset(
    name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    hf_token: str | None = None,
    prefer_lerobot: bool = True,
    revision: str | None = None,
) -> DownloadResult:
    """Download a registered dataset by name.

    Args:
        name: Registry key (see `vla_eval.data.registry`), e.g. "pusht".
        cache_dir: Root directory under which the dataset is stored.
        hf_token: Optional Hugging Face access token for gated/private repos.
        prefer_lerobot: Try to use the `lerobot` package's dataset loader
            (which downloads + indexes episodes) before falling back to a
            raw snapshot download.
        revision: Optional Hugging Face Hub revision (branch, tag, or commit
            SHA) to pin the download to, for reproducibility. Defaults to
            the repo's default branch when not set.

    Returns:
        A `DownloadResult` describing where the data landed.

    Raises:
        DatasetDownloadError: if all download strategies fail.
    """
    spec: DatasetSpec = get_dataset_spec(name)
    local_dir = Path(cache_dir) / name
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("dataset_download_start", dataset=name, repo_id=spec.hub_repo_id)

    if prefer_lerobot:
        try:
            return _download_via_lerobot(spec, local_dir)
        except ImportError:
            logger.info("lerobot_not_installed_falling_back_to_snapshot", dataset=name)
        except Exception as exc:  # noqa: BLE001 - fall back deliberately
            logger.warning("lerobot_download_failed_falling_back", dataset=name, error=str(exc))

    try:
        path = _snapshot_download(spec.hub_repo_id, local_dir, hf_token, revision)
        num_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        logger.info(
            "dataset_download_complete", dataset=name, backend="hf_snapshot", path=str(path)
        )
        return DownloadResult(
            dataset_name=name,
            hub_repo_id=spec.hub_repo_id,
            local_path=path,
            backend="hf_snapshot",
            num_bytes=num_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        raise DatasetDownloadError(
            f"Failed to download dataset '{name}' ({spec.hub_repo_id}): {exc}"
        ) from exc


def _download_via_lerobot(spec: DatasetSpec, local_dir: Path) -> DownloadResult:
    """Load (and thereby cache-download) a dataset using the `lerobot` package."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(spec.hub_repo_id, root=str(local_dir))
    num_bytes = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
    logger.info(
        "dataset_download_complete",
        dataset=spec.name,
        backend="lerobot",
        num_episodes=getattr(dataset, "num_episodes", None),
        num_frames=len(dataset) if hasattr(dataset, "__len__") else None,
    )
    return DownloadResult(
        dataset_name=spec.name,
        hub_repo_id=spec.hub_repo_id,
        local_path=local_dir,
        backend="lerobot",
        num_bytes=num_bytes,
    )


def load_lerobot_dataset(name: str, *, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Any:
    """Load a previously downloaded dataset as a `LeRobotDataset` instance.

    Requires the optional `lerobot` dependency (`pip install vla-eval[lerobot]`).
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    spec = get_dataset_spec(name)
    local_dir = Path(cache_dir) / name
    return LeRobotDataset(spec.hub_repo_id, root=str(local_dir))
