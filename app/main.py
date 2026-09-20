"""Minimal FastAPI entry point and infrastructure checks."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.upload import router as upload_router
from app.api.wiki import router as wiki_router
from app.db import get_engine, initialize_database_schema


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize the development schema once before accepting requests."""

    initialize_database_schema(get_engine())
    yield


app = FastAPI(title="Special Problem Repository API", lifespan=lifespan)
app.include_router(upload_router)
app.include_router(wiki_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the API process can serve requests."""

    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: PostgreSQL responds and pgvector is enabled in this database."""

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
            version = connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    if version is None:
        raise HTTPException(status_code=503, detail="pgvector extension is not enabled")

    return {"status": "ok", "database": "ok", "pgvector": version}
