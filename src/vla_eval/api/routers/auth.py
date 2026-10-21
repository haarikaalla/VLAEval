"""Auth endpoints: exchange a static API key for a short-lived JWT."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from vla_eval.api.dependencies import ApiKeyAuth, AppSettings, WriteAuth
from vla_eval.api.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str = "operator"


class CurrentUserResponse(BaseModel):
    subject: str
    role: str
    auth_type: str


@router.post("/token", response_model=TokenResponse, summary="Exchange an API key for a JWT")
def issue_token(api_key: ApiKeyAuth, settings: AppSettings) -> TokenResponse:
    token = create_access_token(subject="api-client", settings=settings)
    return TokenResponse(access_token=token, expires_in_minutes=settings.api_jwt_expire_minutes)


@router.get("/me", response_model=CurrentUserResponse, summary="Inspect the current identity")
def current_user(principal: WriteAuth) -> CurrentUserResponse:
    return CurrentUserResponse(
        subject=principal.subject,
        role=principal.role,
        auth_type=principal.auth_type,
    )
