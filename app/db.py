"""Reusable SQLModel engine and request-scoped sessions."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine

from app.config import get_settings


def make_engine(database_url: str) -> Engine:
    """Build an engine without opening a connection at import time."""

    return create_engine(database_url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


def get_session() -> Generator[Session]:
    """Provide a session for future route and service dependencies."""

    with Session(get_engine()) as session:
        yield session
