# Special Project Repository and Retrieval

This repository currently contains Fourth's Phase 1 backend foundation for a
PDF-to-Wiki application. It intentionally does **not** include OCR, LLM
generation, chunking logic, embeddings, semantic search, or frontend
code yet.

## Current Phase 1 flow

```text
POST /api/upload
        |
        v
Store digital PDF temporarily
        |
        v
PDF extractor (all pages, text layer only)
        |
        v
Return extraction data (persistence is future work)
        |
        v
Future AI Wiki Generator -> review -> publish -> chunk -> embed
```

The extractor accepts a filesystem path or a binary file-like object such as
`UploadFile.file`. It returns cleaned full text, page count, one-based per-page
text, and typed warnings for blank pages. It never performs database writes and
does not attempt OCR.

## Data model

- `Document` stores source-file provenance, student/project metadata, extracted
  raw text, page count, and extraction lifecycle state.
- `WikiPage` stores versioned Markdown and its review/publication state. Versions
  are unique within a document, and a database constraint keeps `status` and
  `is_published` consistent.
- `Chunk` belongs to a document and may belong to a particular Wiki version. Its
  zero-based `chunk_index` and optional one-based `source_page` preserve enough
  provenance for a future search result to link to the Wiki, original PDF, and
  source page.

The first schema uses portable string-backed enums so model tests run with
SQLite. No migration framework existed in the repository, so one has not been
introduced in this phase.

### pgvector handoff

The Docker PostgreSQL image includes pgvector, and the first-time database
initialization runs `CREATE EXTENSION IF NOT EXISTS vector`. `Chunk` still has no
embedding field. When an embedding model is selected, Jing should:

1. Add the `pgvector` Python dependency and an appropriately dimensioned
   `Vector(...)` column to `Chunk` after the embedding model is selected.
2. Add an HNSW or IVFFlat index after measuring the expected collection size and
   query behavior.
3. Introduce Alembic at that point to migrate the schema rather than relying on
   `SQLModel.metadata.create_all()` in production.

## Local setup and checks

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

For the API and PostgreSQL development stack, copy `.env.example` to `.env`
and run:

```bash
docker compose up --build -d
curl http://localhost:8000/health
curl http://localhost:8000/ready
docker compose exec db psql -U app -d app -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

`/health` checks the API process. `/ready` returns HTTP 200 only if the database
accepts a query and the `vector` extension is enabled; it returns HTTP 503
otherwise. The database initialization script runs only when the PostgreSQL data
volume is first created. On an existing database, enable the extension with
`CREATE EXTENSION IF NOT EXISTS vector;` using a role allowed to create extensions.

The default Docker credentials (`app`/`app`) are for local development. If you
change `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB` in `.env`, also set
`DATABASE_URL` there using the container hostname `db` and matching credentials.
For an API run directly on your host, set `DATABASE_URL` with hostname
`localhost` instead and run `uvicorn app.main:app --reload`. The API loads `.env`
if present. Neither the API startup nor these health checks create application
tables; schema migration work belongs to a later task.

PostgreSQL is published on host port `5433` by default to avoid common conflicts
on `5432`. Set `POSTGRES_PORT` to another free port if needed, and use the same
port in a host-run `DATABASE_URL`. Container-to-container connections always use
`db:5432`.

## Digital PDF upload

Send one `file` part as `multipart/form-data` to `POST /api/upload`. Its declared
content type must be `application/pdf`; the PDF header and extractor also validate
the content. The default maximum size is 20 MiB, configurable with
`MAX_UPLOAD_BYTES`. The upload is copied to a generated temporary directory and
removed after extraction. The original filename is used only as a display
basename in the response.

The JSON response contains `filename`, `page_count`, `raw_text`, `pages` (with
one-based `page_number` and `text`), and `warnings` (with `code`, `message`, and
optional `page_number`). A valid PDF without a text layer returns HTTP 200 with
empty `raw_text` and the extractor's warnings. Unsupported media type returns
415; oversize upload returns 413; invalid, encrypted, or zero-page PDFs return
422; an empty upload returns 400. Unexpected extraction or temporary-file errors
return 500. Error responses use FastAPI's `detail` field; a missing `file` part
receives its standard 422 validation response.

No upload or extraction result is persisted to the database in this task.
