# Special Project Repository and Retrieval

This repository contains the backend foundation and an in-memory PDF-to-Draft-Wiki
flow. It does not include OCR, Wiki publishing/persistence, chunking, embeddings,
semantic search, or frontend code yet.

## Current API flows

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
Future review -> publish -> chunk -> embed

POST /api/wiki/generate
        |
        v
Temporary PDF -> full extraction -> focused title/abstract/keyword source
        |
        v
Ollama/Qwen -> source-backed finalization -> Draft Wiki JSON (no persistence)
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

## Draft Wiki generation API

Send a digital PDF as the `file` part of a `multipart/form-data` request to
`POST /api/wiki/generate`. The route uses the same temporary-file, PDF validation,
and size-limit handling as `/api/upload`. It extracts all pages, selects only
front-matter Wiki source, calls the existing Ollama service, and returns a
validated draft without writing to PostgreSQL. The route and request schema are
visible at `/docs`.

The JSON response includes `status: "draft"`, `original_filename`, `page_count`, `selected_pages`
(one-based), `generated_markdown`, `structure_valid`, and extractor `warnings`.
Source-backed metadata fields are `title`, `english_title`, `students`,
`student_ids`, `advisor`, `academic_year`, and explicit `keywords`; missing values
remain `null` or empty arrays. No full PDF text is returned or sent to Ollama.

An empty upload returns 400, oversize 413, unsupported media type 415, and an
invalid PDF or PDF without usable front-matter text 422. Ollama runtime/model
unavailability returns 503, timeout 504, and invalid generated Wiki 502.
Unexpected extraction/generation failures return 500. Errors use FastAPI's
`detail` field.

For a real local smoke test, start the API and Ollama, pull the configured model,
restore a sample PDF under `data/sample/`, then run from the repository root:

```bash
ollama pull qwen2.5:7b-instruct
curl -X POST http://localhost:8000/api/wiki/generate \
  -F "file=@data/sample/document_071.pdf;type=application/pdf"
```

On PowerShell, run the request on one line:

```powershell
curl.exe -X POST http://localhost:8000/api/wiki/generate -F "file=@data/sample/document_071.pdf;type=application/pdf"
```

The response is a draft for review; the PDF and Markdown are not persisted.

## Reusable sample dataset and Wiki input

The small GroundTruth JSON records and mapping index live under `data/`.
Sample PDFs belong in `data/sample/` locally and are ignored by Git to keep the
repository small. The schema, archive layout, and restore instructions are in
[`docs/evaluation-dataset.md`](docs/evaluation-dataset.md).

Wiki generation uses a focused source prepared by
`app.services.wiki_source.prepare_wiki_source` from extracted title/front-matter,
abstract, keyword, and essential metadata pages. It does not send the entire
PDF text to the LLM by default. The full-document extractor and upload response
remain available for other uses.

## Local Wiki generation service

`app.services.llm_service.generate_wiki(source_text: str) -> str` builds the
existing Task 2.1 prompt, sends it to Ollama, and returns Markdown only when the
existing structure validator accepts it. For focused PDF sources, a provider-neutral
finalization step copies exact readable title and metadata from selected pages,
removes unsupported metadata bullets, and makes standalone missing-information
markers exact. Unsafe marker explanations cause a service error. The Ollama HTTP details live in
`OllamaClient`; callers can handle `LLMServiceError` subclasses for unavailable
runtime, missing model, timeout, empty output, invalid structure, and other
runtime errors. Structural validation checks headings, not factual grounding.

Before strict structure validation, finalization maps only known heading aliases
to the seven canonical headings, inserts wholly missing sections with the exact
missing-information marker, and merges distinct content under duplicate canonical
headings. A model-produced `คำสำคัญ` section is folded into the overview as a
normal keyword line and is not accepted as an eighth heading. Other unknown
headings remain so the strict validator can reject them. The smoke script and
draft API both use this same generation path.

The default model is `qwen2.5:7b-instruct`. Settings read `OLLAMA_BASE_URL`
(default `http://localhost:11434`), `OLLAMA_MODEL`,
`OLLAMA_TIMEOUT_SECONDS` (default 120), and `OLLAMA_TEMPERATURE` (default 0.2)
from environment variables or `.env`. The API container uses
`http://host.docker.internal:11434` by default to reach Ollama on the host;
set `OLLAMA_API_BASE_URL` for Compose if the runtime is elsewhere. The direct
host-run API reads `OLLAMA_BASE_URL`. Neither generation route writes to the
database.

To test manually after starting Ollama, install/pull the selected model and
pass a UTF-8 file containing extracted document text to the smoke script:

```bash
ollama pull qwen2.5:7b-instruct
python scripts/smoke_ollama.py path/to/extracted-text.txt
```

If the runtime is not already running, start it with `ollama serve` in a separate
terminal. Change `OLLAMA_MODEL` and the pull command together to use another
installed model.

To smoke-test the real PDF-to-Wiki pipeline, pass a local digital PDF to the
focused-source script from the repository root:

```bash
python scripts/smoke_pdf_to_wiki.py data/sample/document_071.pdf
```

The script extracts the full PDF, prints the selected title/metadata, abstract,
and keyword source text, and sends only that focused source to Ollama. It prints
the generated Markdown and checks it with the existing Wiki structure validator.
The source selector limits its scan to front matter; later chapters are not sent
to the model. No document or Markdown is persisted. Set `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS` in `.env` or your environment as
needed; start Ollama and pull the configured model before running the command.

## Local Wiki generation evaluation

With the local sample PDFs restored and Ollama running, evaluate the indexed
PDF/GroundTruth pairs and save a JSON report:

```bash
python scripts/evaluate_wiki_generation.py --output data/evaluation/wiki-report.json
```

Use `--ids document_007 document_123` or `--limit 3` for a smaller run. The
script extracts each PDF, selects front-matter Wiki source, calls the existing
LLM service, and continues after per-document failures. It prints generation
counts, structure pass rate, common failure categories, and per-document notes.
The JSON report includes generated Markdown and source-page numbers, but neither
the full PDF text nor GroundTruth is sent to the model. Generated reports under
`data/evaluation/` are ignored by Git. The checks and their limits are described
in [docs/wiki-generation-evaluation.md](docs/wiki-generation-evaluation.md).
Reports keep the raw Ollama reply and the source-backed final Markdown so
model omissions and finalization changes remain visible.
