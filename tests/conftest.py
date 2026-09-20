"""Shared API test database fixtures."""

from collections.abc import Generator

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings
from app.db import get_session
from app.main import app


@pytest.fixture(autouse=True)
def disable_live_ocr_for_tests(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Keep local .env credentials from causing network calls in unit tests."""

    monkeypatch.setenv("OCR_PROVIDER", "disabled")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def api_database() -> Generator[Engine]:
    """Give API tests an isolated persistent-in-process SQLite database."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)

    def test_session() -> Generator[Session]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = test_session
    try:
        yield engine
    finally:
        app.dependency_overrides.pop(get_session, None)
        engine.dispose()
