"""Centralized application configuration using pydantic-settings.

All runtime (service-level) configuration is defined here and sourced from
environment variables / `.env` files. Experiment configuration for training
and evaluation runs (hyperparameters, dataset selection, etc.) is handled
separately via Hydra YAML configs under `configs/` -- see
`vla_eval.core.experiment_config`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Environment variables are matched case-insensitively to field names.
    See `.env.example` for the full list of supported variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: Literal["development", "staging", "production", "test"] = "development"
    app_name: str = "vla-eval"
    app_log_level: str = "INFO"
    app_secret_key: str = Field(
        default="dev-insecure-secret-change-me",
        description="Used to sign JWTs. MUST be overridden in production.",
    )

    # --- API ---
    api_host: str = "0.0.0.0"  # nosec B104 -- intended for containerized deployment
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:5173"
    api_rate_limit_per_minute: int = 120
    api_jwt_algorithm: str = "HS256"
    api_jwt_expire_minutes: int = 60
    api_key_header_name: str = "X-API-Key"
    api_default_api_key: str = "dev-local-api-key-change-me"

    # --- Database ---
    database_url: str = "sqlite:///./vla_eval.db"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "vla-eval"
    mlflow_artifact_root: str = "./mlartifacts"

    # --- Hugging Face ---
    hf_token: str | None = None
    hf_home: str = "./.cache/huggingface"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Training defaults ---
    training_device: Literal["cuda", "cpu", "mps"] = "cpu"
    training_mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    training_num_workers: int = 4

    # --- Artifact output roots (used by API/worker jobs) ---
    checkpoint_root: str = "./models/checkpoints"
    report_root: str = "./reports/generated"

    @field_validator("app_secret_key")
    @classmethod
    def _warn_on_default_secret(cls, value: str) -> str:
        # Defense-in-depth: fail closed in production if the default secret is used.
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide `Settings` instance."""
    settings = Settings()
    if (
        settings.is_production
        and settings.app_secret_key
        == "dev-insecure-secret-change-me"  # nosec B105 - dev-only placeholder, enforced below
    ):
        raise RuntimeError("APP_SECRET_KEY must be set to a secure random value in production.")
    return settings
