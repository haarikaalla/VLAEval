"""Integration tests for the datasets and models listing endpoints, and auth."""

from __future__ import annotations

from fastapi.testclient import TestClient

from vla_eval.api.main import app


def test_list_datasets() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    names = {d["name"] for d in response.json()}
    assert "pusht" in names


def test_get_dataset_not_found() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/datasets/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "DatasetNotFoundError"


def test_list_models() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/models")
    assert response.status_code == 200
    names = {m["name"] for m in response.json()}
    assert "baseline-cnn" in names


def test_download_dataset_requires_api_key() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/datasets/download", json={"name": "pusht"})
    assert response.status_code == 401


def test_auth_token_requires_valid_api_key() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/token", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_auth_token_issued_with_valid_api_key(api_key: str) -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/token", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
