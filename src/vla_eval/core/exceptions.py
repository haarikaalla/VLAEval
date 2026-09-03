"""Application-wide exception hierarchy.

Using a dedicated exception hierarchy allows the FastAPI layer to map
domain errors to consistent HTTP responses, and keeps business logic free
of framework-specific concerns.
"""

from __future__ import annotations


class VLAEvalError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(VLAEvalError):
    """Raised when configuration is missing or invalid."""


class DatasetNotFoundError(VLAEvalError):
    """Raised when a requested dataset is not registered or cannot be located."""


class DatasetDownloadError(VLAEvalError):
    """Raised when a dataset download or preprocessing step fails."""


class ModelNotFoundError(VLAEvalError):
    """Raised when a requested model is not registered or cannot be loaded."""


class ModelLoadError(VLAEvalError):
    """Raised when a model fails to load (checkpoint corruption, missing files, etc.)."""


class TrainingError(VLAEvalError):
    """Raised when a training run fails."""


class EvaluationError(VLAEvalError):
    """Raised when a benchmark/evaluation run fails."""


class JobNotFoundError(VLAEvalError):
    """Raised when a background job id does not exist."""


class AuthenticationError(VLAEvalError):
    """Raised when authentication credentials are missing or invalid."""


class AuthorizationError(VLAEvalError):
    """Raised when an authenticated principal lacks permission for an action."""
