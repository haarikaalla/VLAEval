"""Model registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from vla_eval.api.schemas import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo], summary="List registered model architectures")
def list_models() -> list[ModelInfo]:
    from vla_eval.models.registry import list_models as _list_models

    return [ModelInfo(name=name) for name in _list_models()]
