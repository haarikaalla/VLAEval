"""FastAPI application factory and top-level wiring.

Run locally with:
    uvicorn vla_eval.api.main:app --reload
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from vla_eval.__about__ import __version__
from vla_eval.api.db import init_db
from vla_eval.api.routers import auth, datasets, evaluation, health, leaderboard, models, training
from vla_eval.core.config import get_settings
from vla_eval.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DatasetDownloadError,
    DatasetNotFoundError,
    JobNotFoundError,
    ModelLoadError,
    ModelNotFoundError,
    VLAEvalError,
)
from vla_eval.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

_ERROR_STATUS_MAP: dict[type[VLAEvalError], int] = {
    DatasetNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    JobNotFoundError: status.HTTP_404_NOT_FOUND,
    DatasetDownloadError: status.HTTP_502_BAD_GATEWAY,
    ModelLoadError: status.HTTP_502_BAD_GATEWAY,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    logger.info(
        "app_startup", app_name=settings.app_name, environment=settings.app_env, version=__version__
    )
    init_db()
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="VLA-Eval API",
        description=(
            "REST API for downloading robotics datasets, training/fine-tuning "
            "Vision-Language-Action models, running benchmarks, and serving a "
            "public leaderboard."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=request_id,
        )
        return response

    @app.exception_handler(VLAEvalError)
    async def handle_domain_error(request: Request, exc: VLAEvalError) -> JSONResponse:
        status_code = _ERROR_STATUS_MAP.get(type(exc), status.HTTP_400_BAD_REQUEST)
        logger.warning("domain_error", error_type=type(exc).__name__, message=exc.message)
        return JSONResponse(
            status_code=status_code,
            content={"detail": exc.message, "error_code": type(exc).__name__},
        )

    api_prefix = "/api/v1"
    app.include_router(health.router, prefix="")  # unauthenticated, no version prefix
    app.include_router(auth.router, prefix=api_prefix)
    app.include_router(datasets.router, prefix=api_prefix)
    app.include_router(models.router, prefix=api_prefix)
    app.include_router(training.router, prefix=api_prefix)
    app.include_router(evaluation.router, prefix=api_prefix)
    app.include_router(leaderboard.router, prefix=api_prefix)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
