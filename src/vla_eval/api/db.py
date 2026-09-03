"""SQLAlchemy engine/session setup.

Engine/session-factory construction is lazy and cached per `database_url`
(rather than a single eagerly-created module-level engine) so that:

- Tests can point at an isolated database per test run by setting
  `DATABASE_URL` before first use, without being stuck with whatever engine
  happened to be created first at import time.
- Any process that changes settings at runtime gets a correctly-scoped engine.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from vla_eval.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


@lru_cache
def _build_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


def get_engine() -> Engine:
    """Return the (cached) engine for the currently configured database URL."""
    return _build_engine(get_settings().database_url)


def get_sessionmaker() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def init_db() -> None:
    """Create all tables. In production, use Alembic migrations instead."""
    from vla_eval.api import models_db  # noqa: F401 - ensure models are registered

    Base.metadata.create_all(bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped database session."""
    session_factory = get_sessionmaker()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def new_session() -> Session:
    """Create a standalone session, e.g. for use inside background threads/workers."""
    return get_sessionmaker()()
