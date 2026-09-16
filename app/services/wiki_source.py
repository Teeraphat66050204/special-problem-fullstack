"""Select front-matter text for Wiki prompts without changing PDF extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.pdf_extractor import PdfExtractionResult, PdfPageText

_THAI_ABSTRACT = re.compile(r"(?m)^[ \t]*บท[ \t]*คัด[ \t]*ย่อ[ \t]*$")
_ENGLISH_ABSTRACT = re.compile(r"(?im)^[ \t]*abstract[ \t]*$")
_KEYWORDS = re.compile(r"(?im)^[ \t]*(?:คำ[ \t]*สำ[ \t]*คัญ|keywords?\b)[ \t]*[:：]?")
_METADATA_MARKERS = (
    "ชื่อนักศึกษา",
    "อาจารย์ที่ปรึกษา",
    "ปีการศึกษา",
    "หัวข้อโครงงาน",
    "หัวข้อสหกิจศึกษา",
    "Students",
    "Advisor",
    "Academic Year",
)


@dataclass(frozen=True, slots=True)
class WikiSourcePolicy:
    """Front-matter limits that later page detectors can replace or tune."""

    front_matter_page_limit: int = 8
    title_page_count: int = 2
    max_source_pages: int = 6

    def __post_init__(self) -> None:
        if (
            min(
                self.front_matter_page_limit,
                self.title_page_count,
                self.max_source_pages,
            )
            < 1
        ):
            raise ValueError("Wiki source page limits must be positive")


@dataclass(frozen=True, slots=True)
class WikiSourceSelection:
    """Prompt-ready text and source-page provenance."""

    source_text: str
    selected_pages: tuple[int, ...]


def _keyword_section(text: str) -> str | None:
    """Keep a keyword label and comma-continued wrapped lines only."""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _KEYWORDS.search(line):
            section = [line.strip()]
            for continuation in lines[index + 1 :]:
                if not section[-1].rstrip().endswith((",", ";")):
                    break
                if not continuation.strip():
                    continue
                section.append(continuation.strip())
            return "\n".join(section)
    return None


def _keyword_continuation(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[0]) <= 3 and lines[0].isalnum():
        lines.pop(0)
    if not lines:
        return None
    section = [lines[0]]
    for line in lines[1:]:
        if not section[-1].rstrip().endswith((",", ";")):
            break
        section.append(line)
    return "\n".join(section)


def prepare_wiki_source(
    extraction: PdfExtractionResult,
    policy: WikiSourcePolicy | None = None,
) -> WikiSourceSelection:
    """Prefer title, metadata, abstract, and keyword pages over body chapters.

    Only front-matter pages are considered. An English abstract page suggests the
    preceding page may contain a Thai abstract when its heading extracted poorly.
    A page-four fallback is used when neither abstract heading is detected. The
    full-document ``extraction.full_text`` is deliberately never used here.
    """

    if policy is None:
        policy = WikiSourcePolicy()

    pages = {
        page.page_number: page
        for page in extraction.pages
        if page.page_number <= policy.front_matter_page_limit and page.text.strip()
    }
    if not pages:
        raise ValueError("No usable front-matter text was extracted for Wiki generation")

    selected: dict[int, tuple[str, str]] = {}

    def add_page(page: PdfPageText | None, role: str, text: str | None = None) -> None:
        if page is not None and page.page_number not in selected:
            selected[page.page_number] = (role, text if text is not None else page.text)

    for page_number in range(1, policy.title_page_count + 1):
        add_page(pages.get(page_number), "title page")

    thai_abstract = next(
        (
            page
            for page in pages.values()
            if page.page_number > 2 and _THAI_ABSTRACT.search(page.text)
        ),
        None,
    )
    english_abstract = next(
        (
            page
            for page in pages.values()
            if page.page_number > 2 and _ENGLISH_ABSTRACT.search(page.text)
        ),
        None,
    )
    add_page(thai_abstract, "Thai abstract")
    add_page(english_abstract, "English abstract")
    if english_abstract is not None and english_abstract.page_number >= 4:
        add_page(pages.get(english_abstract.page_number - 1), "possible Thai abstract")

    if thai_abstract is None and english_abstract is None:
        add_page(pages.get(4), "possible abstract")

    metadata_page = next(
        (
            page
            for page in pages.values()
            if page.page_number > policy.title_page_count
            and any(marker.casefold() in page.text.casefold() for marker in _METADATA_MARKERS)
        ),
        None,
    )
    add_page(metadata_page, "project metadata")

    for page in pages.values():
        if len(selected) >= policy.max_source_pages:
            break
        keyword_section = _keyword_section(page.text)
        if keyword_section:
            add_page(page, "keywords", keyword_section)
            if keyword_section.rstrip().endswith((",", ";")):
                following = pages.get(page.page_number + 1)
                continuation = _keyword_continuation(following.text) if following else None
                if continuation:
                    add_page(following, "keyword continuation", continuation)

    retained_numbers = set(list(selected)[: policy.max_source_pages])
    selected_pages = tuple(sorted(retained_numbers))
    source_text = "\n\n".join(
        f"Source page {page_number} ({selected[page_number][0]}):\n"
        f"{selected[page_number][1].strip()}"
        for page_number in selected_pages
    )
    return WikiSourceSelection(source_text=source_text, selected_pages=selected_pages)
