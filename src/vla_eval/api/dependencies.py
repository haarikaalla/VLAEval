"""Shared FastAPI dependency aliases used across routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from vla_eval.api.db import get_db
from vla_eval.api.security import verify_api_key
from vla_eval.core.config import Settings, get_settings

DbSession = Annotated[Session, Depends(get_db)]
ApiKeyAuth = Annotated[str, Depends(verify_api_key)]
AppSettings = Annotated[Settings, Depends(get_settings)]
