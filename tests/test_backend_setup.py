"""Checks for Task 1.2 API and database wiring without an external database."""

import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlmodel import Session

from app import db, main
from app.config import Settings


def test_health_endpoint_does_not_require_a_database(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_engine", lambda: (_ for _ in ()).throw(AssertionError))

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_settings_load_env_file_and_environment_override(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=sqlite:///from-file.db\n")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert Settings(_env_file=env_file).database_url == "sqlite:///from-file.db"

    monkeypatch.setenv("DATABASE_URL", "sqlite:///from-environment.db")
    assert Settings(_env_file=env_file).database_url == "sqlite:///from-environment.db"


def test_engine_and_session_can_use_a_test_database(tmp_path, monkeypatch) -> None:
    engine = db.make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "get_engine", lambda: engine)

    try:
        session_provider = db.get_session()
        session = next(session_provider)
        assert isinstance(session, Session)
        assert session.exec(text("SELECT 1")).scalar_one() == 1
        session_provider.close()
    finally:
        engine.dispose()


def test_application_startup_creates_current_schema(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    engine = db.make_engine(f"sqlite:///{tmp_path / 'startup.db'}")
    monkeypatch.setattr(main, "get_engine", lambda: engine)

    try:
        assert inspect(engine).get_table_names() == []

        with (
            caplog.at_level(logging.INFO, logger="uvicorn.error"),
            TestClient(main.app) as client,
        ):
            assert client.get("/health").status_code == 200
            assert client.get("/health").status_code == 200

        inspector = inspect(engine)
        assert set(inspector.get_table_names()) == {"document", "wiki_page", "chunk"}
        assert "extraction_data" in {column["name"] for column in inspector.get_columns("document")}
        assert {
            constraint["name"] for constraint in inspector.get_check_constraints("document")
        } == {
            "ck_document_filename",
            "ck_document_page_count",
            "ck_document_completed_extraction_data",
        }
        assert frozenset({"storage_key"}) in {
            frozenset(constraint["column_names"])
            for constraint in inspector.get_unique_constraints("document")
        }
        assert (
            caplog.messages.count("database schema initialized tables=document,wiki_page,chunk")
            == 1
        )
    finally:
        engine.dispose()


def test_ready_reports_database_and_extension_status(tmp_path, monkeypatch) -> None:
    engine = db.make_engine(f"sqlite:///{tmp_path / 'ready.db'}")
    monkeypatch.setattr(main, "get_engine", lambda: engine)

    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE pg_extension (extname TEXT, extversion TEXT)"))

        client = TestClient(main.app)
        missing = client.get("/ready")
        assert missing.status_code == 503
        assert missing.json()["detail"] == "pgvector extension is not enabled"

        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO pg_extension (extname, extversion) VALUES ('vector', '0.8.6')")
            )

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ok", "database": "ok", "pgvector": "0.8.6"}
    finally:
        engine.dispose()
