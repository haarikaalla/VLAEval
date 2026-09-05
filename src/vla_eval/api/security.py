"""Authentication utilities: API-key verification and JWT issuance/validation.

Two complementary mechanisms are supported:

- **API key** (`X-API-Key` header): simple, static credential intended for
  service-to-service calls (CI pipelines, CLI, worker processes).
- **JWT bearer tokens**: short-lived tokens issued via `/api/v1/auth/token`
  after presenting a valid API key, intended for the browser-based
  dashboard (avoids sending the long-lived API key on every request).
"""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError

from vla_eval.core.config import Settings, get_settings
from vla_eval.core.logging import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def verify_api_key(
    api_key: Annotated[str | None, Depends(_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """FastAPI dependency validating the `X-API-Key` header.

    Raises:
        HTTPException 401 if the key is missing or invalid.
    """
    if not api_key or not _constant_time_eq(api_key, settings.api_default_api_key):
        logger.warning("api_key_auth_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


def create_access_token(
    subject: str, settings: Settings, expires_minutes: int | None = None
) -> str:
    """Issue a signed JWT for `subject` (e.g. an API key id or username)."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.api_jwt_expire_minutes
    )
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.app_secret_key, algorithm=settings.api_jwt_algorithm)


def verify_access_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """FastAPI dependency validating a `Bearer` JWT and returning its subject."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token."
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.app_secret_key,
            algorithms=[settings.api_jwt_algorithm],
        )
        subject: str = payload["sub"]
        return subject
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token."
        ) from exc
