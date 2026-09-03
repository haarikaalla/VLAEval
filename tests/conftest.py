"""Shared pytest fixtures.

Ensures tests run against an isolated, file-based SQLite database (rather
than the production `sqlite:///./vla_eval.db` or an in-memory DB that
wouldn't be shared across the background-worker threads used by job
execution), and resets the cached `Settings` singleton between tests that
mutate environment variables.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    db_path = tmp_path / "test_vla_eval.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("API_DEFAULT_API_KEY", "test-api-key")
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-key-not-for-production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Keep job artifacts inside the test's tmp dir instead of the repo root.
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("REPORT_ROOT", str(tmp_path / "reports"))

    from vla_eval.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def api_key() -> str:
    return "test-api-key"
