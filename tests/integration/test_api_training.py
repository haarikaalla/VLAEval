"""Integration tests covering the full training -> evaluation -> leaderboard flow."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from vla_eval.api.main import app


def _wait_for_job(client: TestClient, path: str, timeout_s: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.5)
    pytest.fail(f"Job at {path} did not complete within {timeout_s}s")


def test_training_job_submission_requires_api_key() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/training/jobs", json={"model_name": "baseline-cnn"})
    assert response.status_code == 401


def test_training_and_evaluation_end_to_end(api_key: str) -> None:
    headers = {"X-API-Key": api_key}
    with TestClient(app) as client:
        train_response = client.post(
            "/api/v1/training/jobs",
            headers=headers,
            json={
                "model_name": "baseline-cnn",
                "dataset_name": "synthetic",
                "num_epochs": 1,
                "batch_size": 16,
                "device": "cpu",
            },
        )
        assert train_response.status_code == 202
        train_job = train_response.json()
        assert train_job["status"] in ("pending", "running")

        finished_train_job = _wait_for_job(client, f"/api/v1/training/jobs/{train_job['id']}")
        assert finished_train_job["status"] == "succeeded", finished_train_job.get("error_message")

        eval_response = client.post(
            "/api/v1/evaluation/jobs",
            headers=headers,
            json={
                "model_name": "baseline-cnn",
                "dataset_name": "synthetic",
                "device": "cpu",
                "max_samples": 16,
            },
        )
        assert eval_response.status_code == 202
        eval_job = eval_response.json()

        finished_eval_job = _wait_for_job(client, f"/api/v1/evaluation/jobs/{eval_job['id']}")
        assert finished_eval_job["status"] == "succeeded", finished_eval_job.get("error_message")
        assert "composite_score" in finished_eval_job["result"]

        leaderboard_response = client.get("/api/v1/leaderboard")
        assert leaderboard_response.status_code == 200
        entries = leaderboard_response.json()
        assert any(e["model_name"] == "baseline-cnn" for e in entries)


def test_list_training_jobs(api_key: str) -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
