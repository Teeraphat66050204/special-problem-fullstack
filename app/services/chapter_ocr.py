"""Bounded OCR assistance for missing Chapter 1, using the existing OCR provider."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from app.config import Settings
from app.services.ocr_service import (
    OcrConfigurationError,
    OcrProvider,
    OcrServiceError,
    add_ocr_failure_warning,
    load_ocr_provider,
    run_ocr_pages,
)
from app.services.pdf_extractor import (
    PdfExtractionResult,
    PdfInput,
    PdfPageText,
    TextProvenance,
    normalize_text,
)
from app.services.text_quality import assess_text_quality
from app.services.wiki_source import _abstract, _plain, _toc, scan_source_headings


@dataclass(frozen=True, slots=True)
class ChapterOcrCandidate:
    page_number: int
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChapterOcrDiagnostics:
    status: str
    method: str | None = None
    selected_page: int | None = None
    chapter_two_page: int | None = None
    candidates: tuple[ChapterOcrCandidate, ...] = ()
    ocr_pages: tuple[int, ...] = ()
    retained_ocr_pages: tuple[int, ...] = ()
    rejections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChapterOcrResult:
    extraction: PdfExtractionResult
    diagnostics: ChapterOcrDiagnostics


def _lines(page: PdfPageText) -> list[str]:
    # Known layout filler is removed only for candidate ranking, never to create facts.
    return [_plain(line.replace("ǰ", " ")) for line in page.text.splitlines() if line.strip()]


def _toc_candidate(lines: list[str]) -> bool:
    return _toc(lines) or any(
        line.casefold().startswith(("สารบัญ", "ÿćøïćâ", "list of figures", "list of tables"))
        for line in lines[:6]
    )


def rank_chapter_candidates(
    extraction: PdfExtractionResult,
    *,
    chapter: int = 1,
    after: int = 0,
) -> tuple[tuple[ChapterOcrCandidate, ...], tuple[str, ...]]:
    """Cheap document-wide ranking; no PDF rendering, OCR, or heading repair."""
    pages = sorted(extraction.pages, key=lambda page: page.page_number)
    page_lines = {page.page_number: _lines(page) for page in pages}
    toc_pages = {number for number, lines in page_lines.items() if _toc_candidate(lines)}
    abstract_pages = {
        number for number, lines in page_lines.items() if any(_abstract(line) for line in lines)
    }
    candidates: list[ChapterOcrCandidate] = []
    rejected: list[str] = []
    for page in pages:
        number = page.page_number
        lines = page_lines[number]
        if number <= after:
            continue
        if number in toc_pages:
            rejected.append(f"page {number}: native TOC/list-of-figures layout; excluded from OCR")
            continue
        if number in abstract_pages:
            continue
        # Use nearby preceding front matter, not a fixed first-N PDF-page limit.
        preceding = [n for n in toc_pages | abstract_pages if n < number]
        distance = number - max(preceding) if preceding else None
        degraded = assess_text_quality(page.text).degraded
        section = any(re.match(rf"^{chapter}[.\s]+1(?:\s|$|[^0-9.])", line) for line in lines[:12])
        fragment = any(
            len(line) < 65 and re.match(r"^(?:บท|ïì|chapter\b)", line, re.I) for line in lines[:5]
        )
        introduction = chapter == 1 and any(
            "บทนำ" in line or "ïìî" in line.casefold() or "introduction" in line.casefold()
            for line in lines[:6]
        )
        page_one = chapter == 1 and bool(lines) and lines[0] in ("1", "๑")
        transition = degraded and distance is not None and distance <= 3
        if not (section or fragment or introduction or page_one or transition):
            continue
        score = 0
        reasons = []
        for flag, weight, reason in (
            (section, 8, f"first-section pattern {chapter}.1 near page top"),
            (fragment, 4, "short chapter-like/corrupted heading fragment near page top"),
            (introduction, 4, "introduction-like fragment near page top"),
            (page_one, 4, "printed body page numbering starts at 1"),
            (degraded, 2, "degraded native text"),
            (transition, 3, "near front-matter to body transition"),
        ):
            if flag:
                score += weight
                reasons.append(reason)
        candidates.append(ChapterOcrCandidate(number, score, tuple(reasons)))
    # Neighbours of strong candidates help when an otherwise plausible heading is blank.
    known = {candidate.page_number for candidate in candidates}
    for candidate in tuple(candidates):
        if candidate.score < 14:
            continue
        for number in (candidate.page_number - 1, candidate.page_number + 1):
            if (
                number > after
                and number in page_lines
                and number not in known | toc_pages | abstract_pages
            ):
                candidates.append(
                    ChapterOcrCandidate(
                        number,
                        candidate.score - 4,
                        ("neighbour of plausible opening",),
                    )
                )
                known.add(number)
    return tuple(sorted(candidates, key=lambda c: (-c.score, c.page_number))), tuple(rejected)


def recover_chapter_one(
    source: PdfInput,
    extraction: PdfExtractionResult,
    settings: Settings,
    *,
    provider: OcrProvider | None = None,
    allow_ocr: bool = True,
) -> ChapterOcrResult:
    """OCR at most CHAPTER_OCR_MAX_PAGES in total, including boundary/nearby pages.

    Existing successful detection returns immediately. The PDF must still be
    available (upload or local diagnostics); persisted page provenance requires no
    new storage schema. Provider failure never replaces usable existing text.
    """
    pages = sorted(extraction.pages, key=lambda page: page.page_number)
    _, chapters, _ = scan_source_headings(pages)
    if 1 in chapters:
        page = pages[chapters[1][0]]
        return ChapterOcrResult(
            extraction,
            ChapterOcrDiagnostics(
                "already_detected",
                "native" if page.provenance is TextProvenance.PYMUPDF else "existing_ocr",
                page.page_number,
                pages[chapters[2][0]].page_number if 2 in chapters else None,
            ),
        )
    budget = settings.chapter_ocr_max_pages
    ranked, native_rejections = rank_chapter_candidates(extraction)
    ranked = ranked[:budget]
    if not budget or not ranked or not allow_ocr:
        status = "disabled" if not budget else "not_run" if not allow_ocr else "no_candidates"
        return ChapterOcrResult(
            extraction,
            ChapterOcrDiagnostics(
                status,
                candidates=ranked,
                rejections=native_rejections,
            ),
        )
    try:
        provider = provider or load_ocr_provider(settings)
    except OcrServiceError as exc:
        return ChapterOcrResult(
            add_ocr_failure_warning(extraction, exc),
            ChapterOcrDiagnostics(
                "configuration_failed",
                candidates=ranked,
                rejections=native_rejections,
            ),
        )
    if provider is None:
        return ChapterOcrResult(
            extraction,
            ChapterOcrDiagnostics(
                "disabled",
                candidates=ranked,
                rejections=native_rejections,
            ),
        )

    attempted: list[int] = []
    examined: list[ChapterOcrCandidate] = []
    rejections = list(native_rejections)
    recovered: dict[int, PdfPageText] = {}
    failure: OcrServiceError | None = None
    positions = {page.page_number: i for i, page in enumerate(pages)}

    def try_page(candidate: ChapterOcrCandidate) -> bool:
        nonlocal failure
        number = candidate.page_number
        if number in attempted or len(attempted) >= budget:
            return False
        attempted.append(number)
        examined.append(candidate)
        try:
            response = run_ocr_pages(provider, source, (number,))
            text = normalize_text(response.get(number, ""))
        except OcrServiceError as exc:
            failure = exc
            rejections.append(f"page {number}: OCR unavailable; existing text preserved")
            return False
        if not text or _toc_candidate([_plain(line) for line in text.splitlines() if line.strip()]):
            rejections.append(f"page {number}: OCR returned empty text or a TOC; rejected")
            return False
        original = pages[positions[number]]
        recovered[number] = PdfPageText(
            number,
            text,
            native_text=original.native_text if original.native_text is not None else original.text,
            ocr_text=text,
            provenance=TextProvenance.OCR,
        )
        return True

    def rescan():
        current = [recovered.get(page.page_number, page) for page in pages]
        return current, scan_source_headings(current)[1]

    for candidate in ranked:
        try_page(candidate)
        current, chapters = rescan()
        if 1 in chapters or failure:
            break
        rejections.append(
            f"page {candidate.page_number}: no real Chapter 1 after shared heuristics"
        )

    start = current[chapters[1][0]].page_number if 1 in chapters else None
    if start and 2 not in chapters and not failure:
        boundary_candidates, _ = rank_chapter_candidates(extraction, chapter=2, after=start)
        for candidate in boundary_candidates:
            if len(attempted) >= budget:
                break
            try_page(candidate)
            current, chapters = rescan()
            if 2 in chapters or failure:
                break
            rejections.append(
                f"page {candidate.page_number}: no real Chapter 2 after shared heuristics"
            )

    end = current[chapters[2][0]].page_number if 2 in chapters else None
    if start and end and not failure:
        # Fill remaining budget with degraded Chapter 1 continuation pages only.
        for page in pages:
            if len(attempted) >= budget:
                break
            if start < page.page_number < end and assess_text_quality(page.text).degraded:
                try_page(
                    ChapterOcrCandidate(page.page_number, 0, ("degraded Chapter 1 continuation",))
                )
                if failure:
                    break

    current, chapters = rescan()
    start = current[chapters[1][0]].page_number if 1 in chapters else None
    end = current[chapters[2][0]].page_number if 2 in chapters else None
    # Retain OCR only inside a verified chapter span or at its Chapter 2 boundary.
    # Failed candidates before the opening cannot replace abstracts or front matter.
    retained = {
        number: page
        for number, page in recovered.items()
        if start is not None and number >= start and (number <= end if end else number <= start + 1)
    }
    final_pages = tuple(retained.get(page.page_number, page) for page in pages)
    # A title-only opening can depend on OCR body text on the immediately following
    # page. Validate the actual retained extraction, not discarded trial candidates.
    final_chapters = scan_source_headings(list(final_pages))[1]
    if start and 1 not in final_chapters:
        rejections.append(
            "OCR opening lost supporting evidence after filtering; native text preserved"
        )
        retained = {}
        final_pages = tuple(pages)
        start = end = None
    result = (
        replace(
            extraction,
            pages=final_pages,
            full_text=normalize_text("\n\n".join(page.text for page in final_pages)),
        )
        if retained
        else extraction
    )
    if failure:
        result = add_ocr_failure_warning(result, failure)
    return ChapterOcrResult(
        result,
        ChapterOcrDiagnostics(
            "recovered"
            if start
            else "configuration_failed"
            if isinstance(failure, OcrConfigurationError)
            else "ocr_failed"
            if failure
            else "not_found",
            "ocr_fallback" if start else None,
            start,
            end,
            tuple(examined),
            tuple(attempted),
            tuple(sorted(retained)),
            tuple(rejections),
        ),
    )
