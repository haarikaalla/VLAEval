"""Structured logging configuration using `structlog`.

Provides JSON logs in production and human-readable console logs in
development, with consistent field names (timestamp, level, logger, event).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from vla_eval.core.config import get_settings


def configure_logging() -> None:
    """Configure stdlib logging + structlog for the whole application.

    Safe to call multiple times (idempotent).
    """
    settings = get_settings()
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.app_env in ("production", "staging"):
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog-bound logger, configuring logging on first use."""
    if not structlog.is_configured():
        configure_logging()
    return structlog.get_logger(name)
