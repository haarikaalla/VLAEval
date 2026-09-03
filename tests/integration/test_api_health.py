"""Integration tests for unauthenticated health/readiness endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from vla_eval.api.main import app


def test_healthz_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "vla-eval"


def test_readyz_checks_database() -> None:
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
