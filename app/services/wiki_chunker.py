"""Deterministically split finalized Wiki Markdown without persistence or embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150

_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+([^\n]+)(?:\n|$)")
_BOUNDARY_PATTERNS = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"\n"),
    re.compile(r"[.!?。！？](?=\s|$)"),
    re.compile(r"[ \t]+"),
)


@dataclass(frozen=True, slots=True)
class WikiChunk:
    """One ordered Markdown substring plus provenance for later persistence."""

    chunk_index: int
    content: str
    section_title: str | None
    headings: tuple[str, ...]
    char_count: int
    start_offset: int
    end_offset: int
    document_id: int | str | None = None
    wiki_page_id: int | None = None


@dataclass(frozen=True, slots=True)
class _Section:
    start: int
    end: int
    title: str | None


@dataclass(frozen=True, slots=True)
class _ChunkSpan:
    start: int
    end: int
    headings: tuple[str, ...]


def _sections(markdown: str) -> tuple[_Section, ...]:
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return (_Section(0, len(markdown), None),)

    sections: list[_Section] = []
    if matches[0].start() > 0:
        sections.append(_Section(0, matches[0].start(), None))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append(_Section(match.start(), end, match.group(2).strip()))
    return tuple(section for section in sections if section.end > section.start)


def _preferred_end(markdown: str, start: int, maximum: int, chunk_size: int) -> int:
    """Choose the highest-priority separator in the latter half of a size window."""

    minimum = min(maximum, start + max(1, chunk_size // 2))
    for pattern in _BOUNDARY_PATTERNS:
        candidates = [
            match.end()
            for match in pattern.finditer(markdown, start, maximum)
            if minimum <= match.end() <= maximum
        ]
        if candidates:
            return candidates[-1]
    return maximum


def _overlap_start(markdown: str, section_start: int, end: int, overlap: int) -> int:
    """Retain at most ``overlap`` characters and align to a readable boundary."""

    if overlap == 0:
        return end
    target = max(section_start, end - overlap)
    for pattern in _BOUNDARY_PATTERNS:
        candidates = [
            match.end()
            for match in pattern.finditer(markdown, target, end)
            if target <= match.end() < end
        ]
        if candidates:
            return candidates[0]
    return target


def _split_long_section(
    markdown: str,
    section: _Section,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkSpan]:
    spans: list[_ChunkSpan] = []
    start = section.start
    headings = (section.title,) if section.title else ()
    while section.end - start > chunk_size:
        end = _preferred_end(markdown, start, start + chunk_size, chunk_size)
        if end <= start:
            end = min(section.end, start + chunk_size)
        spans.append(_ChunkSpan(start, end, headings))
        following = _overlap_start(markdown, section.start, end, chunk_overlap)
        start = max(start + 1, following)
    if start < section.end:
        spans.append(_ChunkSpan(start, section.end, headings))
    return spans


def _span_headings(sections: list[_Section]) -> tuple[str, ...]:
    return tuple(section.title for section in sections if section.title is not None)


def _chunk_spans(
    markdown: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_ChunkSpan]:
    sections = _sections(markdown)
    if len(markdown) <= chunk_size:
        return [_ChunkSpan(0, len(markdown), _span_headings(list(sections)))]

    spans: list[_ChunkSpan] = []
    pending: list[_Section] = []

    def flush_pending() -> None:
        if not pending:
            return
        spans.append(
            _ChunkSpan(
                pending[0].start,
                pending[-1].end,
                _span_headings(pending),
            )
        )
        pending.clear()

    for section in sections:
        if section.end - section.start > chunk_size:
            flush_pending()
            spans.extend(_split_long_section(markdown, section, chunk_size, chunk_overlap))
            continue
        if pending and section.end - pending[0].start > chunk_size:
            flush_pending()
        pending.append(section)
    flush_pending()
    return spans


def chunk_wiki_markdown(
    markdown_content: str,
    *,
    document_id: int | str | None = None,
    wiki_page_id: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[WikiChunk, ...]:
    """Split Markdown in source order using headings and recursive text boundaries.

    Whole Markdown sections are packed when they fit. Oversized sections prefer
    paragraph, line, sentence, and space boundaries before character fallback.
    Subsequent pieces retain a boundary-aligned suffix no larger than
    ``chunk_overlap``. Whitespace-only input consistently returns an empty tuple.
    """

    if not isinstance(markdown_content, str):
        raise TypeError("markdown_content must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not markdown_content.strip():
        return ()

    spans = _chunk_spans(
        markdown_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[WikiChunk] = []
    for index, span in enumerate(spans):
        content = markdown_content[span.start : span.end]
        section_title = span.headings[0] if len(span.headings) == 1 else None
        chunks.append(
            WikiChunk(
                chunk_index=index,
                content=content,
                section_title=section_title,
                headings=span.headings,
                char_count=len(content),
                start_offset=span.start,
                end_offset=span.end,
                document_id=document_id,
                wiki_page_id=wiki_page_id,
            )
        )
    return tuple(chunks)
