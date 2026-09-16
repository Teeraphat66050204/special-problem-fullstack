# Wiki Markdown chunking

Task 3.1 exposes `app.services.wiki_chunker.chunk_wiki_markdown` for splitting
edited or published Wiki Markdown into deterministic, ordered text segments. It
does not call an LLM and never summarizes, normalizes, or otherwise rewrites the
source wording.

## Strategy and defaults

The default `chunk_size` is 1,200 characters and the default `chunk_overlap` is
150 characters. These are character limits rather than token limits because the
embedding model and its tokenizer will be selected in Task 3.3. Both values are
function arguments; overlap must be non-negative and smaller than the chunk
size.

The splitter first identifies Markdown headings from H1 through H6. Complete
adjacent sections are packed together when they fit within the size limit, so a
short Wiki remains one chunk. An oversized section is divided recursively using
the following boundary preference:

1. Markdown heading and section boundaries
2. Paragraph breaks
3. Line breaks
4. Sentence-ending punctuation where present
5. Spaces
6. Character position as the final fallback

This keeps headings with their following content where practical. Every piece
of a long section also carries its `section_title` and `headings` provenance,
even when only the first piece contains the literal Markdown heading.

## Output and provenance

Each immutable `WikiChunk` contains:

- zero-based `chunk_index`
- exact `content` copied from the input
- `section_title` when the chunk belongs to one identifiable section
- ordered `headings` covered by the chunk
- `char_count`
- half-open `start_offset` and `end_offset` into the original Markdown
- optional `document_id` and `wiki_page_id`

Chunks always follow source order. For every result, `content` equals
`markdown_content[start_offset:end_offset]`. Whitespace-only input returns an
empty tuple and never creates an empty database candidate.

## Overlap behavior

Overlap is applied only when an individual section must be split. The next
piece retains a suffix of the previous piece of at most `chunk_overlap`
characters. Its start is moved forward to a paragraph, line, sentence, or space
boundary when one exists inside that overlap window; character alignment is the
fallback. This can produce less overlap than requested, but never more. Source
offsets make the duplicated range explicit, and monotonically increasing start
offsets prevent duplicate full chunks.

Task 3.1 returns in-memory records only. It does not create embeddings, choose a
tokenizer, write `Chunk` rows, or access PostgreSQL. Those steps belong to later
Phase 3 tasks after publication and embedding configuration are defined.
