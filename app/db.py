"""Reusable SQLModel engine and request-scoped sessions."""

import logging
from collections.abc import Generator
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

logger = logging.getLogger("uvicorn.error")


def make_engine(database_url: str) -> Engine:
    """Build an engine without opening a connection at import time."""

    return create_engine(database_url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


def initialize_database_schema(engine: Engine) -> None:
    """Create the registered development schema once at application startup."""

    from app.models import Chunk, Document, WikiPage

    model_types = (Document, WikiPage, Chunk)
    SQLModel.metadata.create_all(engine)
    logger.info(
        "database schema initialized tables=%s",
        ",".join(model.__tablename__ for model in model_types),
    )


def get_session() -> Generator[Session]:
    """Provide a request-scoped database session."""

    with Session(get_engine()) as session:
        yield session
