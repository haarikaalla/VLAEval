"""Unit tests for `vla_eval.core.config`."""

from __future__ import annotations

import pytest

from vla_eval.core.config import get_settings


def test_default_settings_load() -> None:
    settings = get_settings()
    assert settings.app_name == "vla-eval"
    assert settings.app_env == "test"


def test_cors_origins_list_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CORS_ORIGINS", "http://a.com, http://b.com")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.cors_origins_list == ["http://a.com", "http://b.com"]
    get_settings.cache_clear()


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()
