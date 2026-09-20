# Task 1.1 data model

This document defines the SQLModel schema for source documents, generated Wiki
pages, and chunks. It intentionally covers data structure only; extraction,
generation, chunking, embeddings, vector indexes, routes, and search belong to
later Gantt tasks.

## Entity relationships

```text
+------------------+       1:N       +------------------+
| Document         |----------------<| WikiPage         |
|------------------|                 |------------------|
| PK id            |                 | PK id            |
| original_filename|                 | FK document_id   |
| metadata         |                 | markdown_content |
| storage_key      |                 | version / status |
| raw_text         |                 +--------+---------+
| extraction_data  |                          |
| extraction_status|                          |
+--------+---------+                          | 1:N
         |                                    |
         | 1:N                                |
         |                           +--------v---------+
         +--------------------------<| Chunk            |
                                     |------------------|
                                     | PK id            |
                                     | FK document_id   |
                                     | FK wiki_page_id? |
                                     | index / content  |
                                     | page / tokens    |
                                     +------------------+
```

A `Chunk` always belongs to one `Document`. It may also belong to one
`WikiPage`; when it does, the composite foreign key guarantees that both rows
refer to the same document.

## Entities

### Document

Stores source identity and metadata (`original_filename`, title, author,
student ID, academic year, and unique storage key), extracted `raw_text`, page
count, versioned page/warning JSON in `extraction_data` (including native/OCR text
and per-page `pymupdf`, `ocr`, or `mixed` provenance), extraction lifecycle
status, and timestamps.

### WikiPage

Stores a versioned Markdown representation of a document, its editorial status,
the optional generating model name, publication flag, and timestamps.

### Chunk

Stores ordered, non-empty text prepared for later embedding and search. The
optional one-based source page and non-negative token count preserve provenance
without introducing embedding or vector fields during Task 1.1.

## Database integrity

- Wiki versions are unique within a document and start at version 1.
- Chunk indexes are non-negative and unique within a Wiki page.
- Document-level chunks (those without a Wiki page) have unique indexes within
  their document.
- The `(wiki_page_id, document_id)` foreign key prevents a chunk from referring
  to a Wiki page owned by another document.
- Source pages are at least 1 when present; token counts are non-negative when
  present; trimmed chunk content cannot be empty.
- Publication status and `is_published` must agree.
- A completed Document must have both raw text and page-aware extraction data.
- ORM and database cascades remove dependent Wiki pages and chunks when their
  owning parent is deleted.
- Status enums are stored as portable validated strings. Timestamp columns are
  timezone-aware, allowing the same definitions to target PostgreSQL later.

`Chunk` deliberately has no embedding column or vector index. Those require an
embedding model and dimension decision and are deferred to Gantt Phase 3.
