"""Deterministic document-wide abstract and Chapter 1 source selection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.services.pdf_extractor import PdfExtractionResult, PdfPageText, TextProvenance


def _plain(line: str) -> str:
    # Detection normalization only: never repair the retained source text.
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFC", line).replace("ํา", "ำ").strip().strip("#*_ ")
    ).strip()


def _compact(line: str) -> str:
    return re.sub(r"\s+", "", _plain(line)).casefold()


_CHAPTER = re.compile(r"^(?:บท(?:ที่)?([1-9๑-๙])|chapter([1-9]|one|two))(?=$|[^0-9๑-๙])", re.I)
_SECTION = re.compile(r"^1\.[0-9]+(?![0-9.])(?:\s|$)")
_LEADER = re.compile(r"\.{3,}|…{2,}|[·_]{3,}")
_RELEVANT = (
    "ที่มา",
    "ความสำคัญ",
    "ความเป็นมา",
    "ปัญหา",
    "หลักการ",
    "เหตุผล",
    "วัตถุประสงค์",
    "ขอบเขต",
    "ประโยชน์",
    "วิธี",
    "แนวทาง",
    "background",
    "problem",
    "rationale",
    "objective",
    "scope",
    "benefit",
    "method",
    "approach",
    "introduction",
)
_METADATA = re.compile(
    r"หัวข้อ(?:โครงงาน|สหกิจศึกษา|ปัญหาพิเศษ)|ชื่อ(?:โครงงาน|โครงการ|นักศึกษา)|"
    r"รหัสนักศึกษา|อาจารย์ที่ปรึกษา|ปีการศึกษา|"
    r"^(?:project title|english title|title|students?|student ids?|advisor|academic year)\b",
    re.I,
)
_KEYWORDS = re.compile(r"^(?:คำ\s*สำ\s*คัญ|keywords?)\s*[:：]?", re.I)


@dataclass(frozen=True, slots=True)
class WikiSourcePolicy:
    """Character budgets, independent of document length or chapter page number."""

    max_source_characters: int = 24000
    abstract_characters: int = 6500
    chapter_characters: int = 12000
    section_characters: int = 2400
    metadata_characters: int = 3500

    def __post_init__(self) -> None:
        if (
            min(
                self.max_source_characters,
                self.abstract_characters,
                self.chapter_characters,
                self.section_characters,
                self.metadata_characters,
            )
            < 1
        ):
            raise ValueError("Source character budgets must be positive")


@dataclass(frozen=True, slots=True)
class WikiSourcePage:
    page_number: int
    role: str
    text: str
    provenance: TextProvenance


@dataclass(frozen=True, slots=True)
class SourceDetection:
    """Physical PDF pages and deterministic signals, not probability estimates."""

    abstract_pages: tuple[int, ...] = ()
    chapter_start: int | None = None
    chapter_end: int | None = None
    chapter_two_start: int | None = None
    chapter_method: str | None = None
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WikiSourceSelection:
    source_text: str
    pages: tuple[WikiSourcePage, ...]
    detection: SourceDetection = SourceDetection()

    @property
    def selected_pages(self) -> tuple[int, ...]:
        return tuple(sorted({page.page_number for page in self.pages}))


def _chapter(line: str) -> int | None:
    match = _CHAPTER.match(_compact(line))
    if not match:
        return None
    value = match[1] or match[2]
    return {"one": 1, "two": 2}[value] if value in ("one", "two") else int(value)


def _toc(lines: list[str]) -> bool:
    if any(
        _compact(line) in ("สารบัญ", "สารบัญ(ต่อ)", "contents", "tableofcontents")
        for line in lines[:8]
    ):
        return True
    if sum(bool(_LEADER.search(line)) for line in lines) >= 3:
        return True
    if sum(_chapter(line) is not None for line in lines) >= 3:
        return True
    entries = sum(
        bool(re.match(r"^(?:1\.\d+|บท|chapter).+\s\d+\s*$", line, re.I)) and len(line) < 140
        for line in lines
    )
    return entries >= 4


def _abstract(line: str) -> str | None:
    return {"บทคัดย่อ": "Thai abstract", "abstract": "English abstract"}.get(_compact(line))


def _boundary(line: str) -> bool:
    return bool(
        _abstract(line)
        or _chapter(line)
        or _compact(line)
        in (
            "กิตติกรรมประกาศ",
            "สารบัญ",
            "สารบัญตาราง",
            "สารบัญรูป",
            "สารบัญภาพ",
            "acknowledgements",
            "acknowledgments",
            "contents",
            "tableofcontents",
            "references",
        )
    )


def _chapter_candidate(
    lines: list[list[str]],
    pi: int,
    li: int,
) -> tuple[str, ...]:
    line = lines[pi][li]
    chapter = _chapter(line)
    if (li > 7 and chapter == 1) or _LEADER.search(line) or _toc(lines[pi]):
        return ()
    match = _CHAPTER.match(_compact(line))
    assert match is not None
    if re.search(r"[0-9๑-๙]$", _compact(line)[match.end() :]):
        return ()  # Undotted TOC entry with a trailing page reference.
    after = lines[pi][li + 1 :]
    following = lines[pi + 1][:20] if pi + 1 < len(lines) and not _toc(lines[pi + 1]) else []
    lookahead = after + following
    section = any(re.match(rf"^{chapter}\.1(?:\s|$|[^0-9.])", _plain(x)) for x in lookahead)
    introduction = any(
        "บทนำ" in _plain(x) or "introduction" in x.casefold() for x in [line, *after[:4]]
    )
    body = sum(len(x) for x in lookahead if not _boundary(x) and not _LEADER.search(x)) >= 100
    signals = ["heading near page start" if li <= 7 else "chapter boundary within page"]
    if section:
        signals.append(f"section {chapter}.1 nearby")
    if introduction:
        signals.append("introduction nearby")
    if body:
        signals.append("substantial following text")
    return (
        tuple(signals) if body and (section or introduction or li <= 7 and len(after) >= 4) else ()
    )


def scan_source_headings(
    pages: list[PdfPageText],
) -> tuple[
    dict[str, list[tuple[int, int]]],
    dict[int, tuple[int, int, tuple[str, ...]]],
    list[str],
]:
    """Shared deterministic scan; returns page indices even without usable abstracts."""
    lines = [[line.strip() for line in page.text.splitlines() if line.strip()] for page in pages]
    abstracts: dict[str, list[tuple[int, int]]] = {}
    chapters: dict[int, tuple[int, int, tuple[str, ...]]] = {}
    signals: list[str] = []
    for pi, page_lines in enumerate(lines):
        if _toc(page_lines):
            signals.append(f"page {pages[pi].page_number}: rejected TOC")
            continue
        for li, line in enumerate(page_lines):
            role = _abstract(line)
            if role and any(len(x) >= 10 and not _boundary(x) for x in page_lines[li + 1 :]):
                abstracts.setdefault(role, []).append((pi, li))
                signals.append(f"page {pages[pi].page_number}: standalone {role} with body")
            number = _chapter(line)
            if number not in (1, 2) or number in chapters:
                continue
            if number == 2 and (1 not in chapters or (pi, li) <= chapters[1][:2]):
                continue
            reasons = _chapter_candidate(lines, pi, li)
            if reasons:
                chapters[number] = (pi, li, reasons)
                signals.append(
                    f"page {pages[pi].page_number}: Chapter {number}: {', '.join(reasons)}"
                )
            else:
                signals.append(f"page {pages[pi].page_number}: rejected Chapter {number} candidate")
    return abstracts, chapters, signals


def prepare_wiki_source(
    extraction: PdfExtractionResult,
    policy: WikiSourcePolicy | None = None,
) -> WikiSourceSelection:
    """Scan every final extracted page; retain bounded, provenance-labelled excerpts."""
    policy = policy or WikiSourcePolicy()
    pages = sorted(extraction.pages, key=lambda page: page.page_number)
    lines = [[line.strip() for line in page.text.splitlines() if line.strip()] for page in pages]
    abstracts, chapters, signals = scan_source_headings(pages)

    selected: list[WikiSourcePage] = []
    metadata_reserve = min(policy.metadata_characters, policy.max_source_characters // 6)
    remaining = policy.max_source_characters - metadata_reserve

    def add(pi: int, role: str, text: str, budget: int) -> int:
        nonlocal remaining
        page = pages[pi]
        overhead = len(f"Source page {page.page_number} ({role}):\n") + (2 if selected else 0)
        capacity = min(budget, remaining - overhead)
        if capacity <= 0:
            return 0
        excerpt = text.strip()[:capacity].rstrip()
        if not excerpt:
            return 0
        selected.append(WikiSourcePage(page.page_number, role, excerpt, page.provenance))
        remaining -= overhead + len(excerpt)
        return len(excerpt)

    def excerpt_text(pi: int, excerpt: list[str]) -> str:
        # Preserve original line spacing within contiguous excerpts (including OCR).
        if not excerpt:
            return ""
        raw = pages[pi].text
        begin = raw.find(excerpt[0])
        cursor = begin
        for line in excerpt:
            cursor = raw.find(line, cursor) + len(line)
        return raw[begin:cursor]

    def continuation(index: int, role: str) -> bool:
        content = " ".join(lines[index])
        letters = [char for char in content if char.isalpha()]
        if not letters:
            return False
        if role == "Thai abstract":
            return sum("ก" <= char <= "๛" for char in letters) / len(letters) > 0.3
        return sum(char.isascii() for char in letters) / len(letters) > 0.8

    abstract_pages: set[int] = set()
    for role in ("Thai abstract", "English abstract"):
        budget = policy.abstract_characters
        for pi, li in abstracts.get(role, []):
            abstract_pages.add(pages[pi].page_number)
            done = False
            keywords = False
            previous = ""
            for index in range(pi, len(pages)):
                if index > pi and (
                    _toc(lines[index])
                    or not continuation(index, role)
                    or any(_boundary(line) for line in lines[index][:4])
                ):
                    break
                excerpt: list[str] = []
                for line in lines[index][li if index == pi else 0 :]:
                    if _boundary(line) and not (
                        index == pi and not excerpt and _abstract(line) == role
                    ):
                        done = True
                        break
                    if keywords and previous and not previous.endswith((",", ";", ":", "：")):
                        done = True
                        break
                    if keywords and len(line) <= 3 and line.isalnum():
                        continue
                    excerpt.append(line)
                    if _KEYWORDS.match(_plain(line)):
                        keywords = True
                        previous = _plain(line).rstrip()
                        if _compact(line) in ("คำสำคัญ", "keywords", "keyword"):
                            previous = ":"
                    elif keywords:
                        previous = line.rstrip()
                if excerpt:
                    used = add(index, role, excerpt_text(index, excerpt), budget)
                    if used:
                        abstract_pages.add(pages[index].page_number)
                        budget -= used
                if done or budget <= 0:
                    break

    start, end = chapters.get(1), chapters.get(2)
    if start and not end:
        signals.append("Chapter 2 not detected: end unknown; bounded Chapter 1 excerpts only")
    chapter_end: int | None = None
    if start:
        chunks: list[tuple[str, list[tuple[int, str]]]] = [("", [])]
        for pi in range(start[0], (end[0] + 1) if end else len(pages)):
            first = start[1] if pi == start[0] else 0
            last = end[1] if end and pi == end[0] else len(lines[pi])
            for line in lines[pi][first:last]:
                # Page number immediately before Chapter 2 is not Chapter 1 content.
                if end and pi == end[0] and line.isdecimal():
                    continue
                chapter_end = pages[pi].page_number
                plain = _plain(line)
                if _SECTION.match(plain):
                    chunks.append((plain, []))
                elif chunks[-1][0] and _SECTION.fullmatch(chunks[-1][0]) and len(plain) < 140:
                    chunks[-1] = (chunks[-1][0] + " " + plain, chunks[-1][1])
                chunks[-1][1].append((pi, line))
        relevant = [
            chunk
            for chunk in chunks
            if (
                chunk[0].startswith("1.1 ") or any(word in _compact(chunk[0]) for word in _RELEVANT)
            )
        ]
        budget = min(policy.chapter_characters, remaining)
        chosen = relevant or chunks
        # Share space between recognized sections so a long background does not
        # crowd out late objectives, scope, or methods.
        per_section = min(policy.section_characters, max(1, budget // len(chosen)))
        for _, content in chosen:
            section_budget = min(per_section, budget) if relevant else budget
            groups: dict[int, list[str]] = {}
            for pi, line in content:
                groups.setdefault(pi, []).append(line)
            for pi, group in groups.items():
                used = add(pi, "Chapter 1", excerpt_text(pi, group), section_budget)
                section_budget -= used
                budget -= used
                if section_budget <= 0 or budget <= 0:
                    break
            if budget <= 0:
                break

    if not selected:
        raise ValueError("No usable abstract or real Chapter 1 was detected in extracted pages")

    # Cover context is optional metadata; abstract/chapter discovery has no page limit.
    remaining += metadata_reserve
    metadata_budget = policy.metadata_characters
    for pi, page_lines in enumerate(lines):
        if _toc(page_lines) or not page_lines:
            continue
        if pages[pi].page_number <= 2 and not any(_boundary(line) for line in page_lines):
            text, role = "\n".join(page_lines), "title page"
        else:
            indices = {
                i
                for i, line in enumerate(page_lines)
                if len(line) < 180 and _METADATA.match(_plain(line))
            }
            indices |= {
                i + 1
                for i in indices
                if i + 1 < len(page_lines) and not _boundary(page_lines[i + 1])
            }
            if not any(page.page_number == pages[pi].page_number for page in selected):
                for i, line in enumerate(page_lines):
                    if _KEYWORDS.match(_plain(line)):
                        indices.add(i)
                        while i + 1 < len(page_lines) and (
                            page_lines[i].endswith((",", ";", ":", "："))
                            or _compact(page_lines[i]) in ("คำสำคัญ", "keywords", "keyword")
                        ):
                            i += 1
                            if _boundary(page_lines[i]):
                                break
                            indices.add(i)
            text = "\n".join(page_lines[i] for i in sorted(indices))
            role = "project metadata"
        metadata_budget -= add(pi, role, text, metadata_budget)
        if metadata_budget <= 0:
            break

    source_text = "\n\n".join(
        f"Source page {page.page_number} ({page.role}):\n{page.text}" for page in selected
    )
    return WikiSourceSelection(
        source_text,
        tuple(selected),
        SourceDetection(
            abstract_pages=tuple(sorted(abstract_pages)),
            chapter_start=pages[start[0]].page_number if start else None,
            chapter_end=chapter_end if end else None,
            chapter_two_start=pages[end[0]].page_number if end else None,
            chapter_method=(
                "native" if pages[start[0]].provenance is TextProvenance.PYMUPDF else "ocr"
            )
            if start
            else None,
            signals=tuple(signals),
        ),
    )
