"""Smoke-test native extraction and configured OCR without invoking an LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from app.config import Settings, get_settings
from app.datasets.groundtruth import DatasetEntry, load_dataset_index
from app.services.ocr_service import (
    OcrConfigurationError,
    OcrServiceError,
    add_ocr_failure_warning,
    apply_ocr_fallback,
    load_ocr_provider,
)
from app.services.pdf_extractor import (
    PdfExtractionError,
    PdfExtractionResult,
    PdfPageText,
    WarningCode,
    extract_pdf,
)
from app.services.text_quality import TextQualityAssessment, assess_front_matter_quality
from app.services.wiki_evidence import WikiSourceEvidence, collect_wiki_source_evidence
from app.services.wiki_source import WikiSourceSelection, prepare_wiki_source

_OCR_WARNING_CODES = frozenset((WarningCode.OCR_CONFIGURATION, WarningCode.OCR_FAILED))
_THAI_ABSTRACT_HEADING = re.compile(
    r"(?m)^\s*(?:#{1,6}\s+)?(?:\*{1,2}|_{1,2})?บท\s*คัด\s*ย่อ"
    r"(?:\*{1,2}|_{1,2})?\s*$"
)
_ENGLISH_ABSTRACT_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?(?:\*{1,2}|_{1,2})?abstract"
    r"(?:\*{1,2}|_{1,2})?\s*$"
)
_KEYWORD_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?(?:\*{1,2}|_{1,2})?"
    r"(?:คำ\s*สำคัญ|keywords?)\b"
)


def _find_document(data_dir: Path, document_id: str) -> DatasetEntry:
    index = load_dataset_index(data_dir / "index.json")
    entry = next(
        (entry for entry in index.documents if entry.document_id == document_id),
        None,
    )
    if entry is None:
        raise ValueError(f"Unknown document ID: {document_id}")
    return entry


def _front_matter(
    extraction: PdfExtractionResult,
    max_pages: int,
) -> tuple[PdfPageText, ...]:
    return tuple(page for page in extraction.pages if page.page_number <= max_pages)


def _focused_source(extraction: PdfExtractionResult) -> WikiSourceSelection | None:
    try:
        return prepare_wiki_source(extraction)
    except ValueError:
        return None


def _evidence(selection: WikiSourceSelection | None) -> WikiSourceEvidence | None:
    if selection is None:
        return None
    return collect_wiki_source_evidence(selection.source_text)


def _detected_abstract(
    selection: WikiSourceSelection | None,
    heading: re.Pattern[str],
) -> str | None:
    if selection is None:
        return None
    for page in selection.pages:
        match = heading.search(page.text)
        if match is None:
            continue
        content = page.text[match.end() :]
        keyword = _KEYWORD_HEADING.search(content)
        if keyword is not None:
            content = content[: keyword.start()]
        return content.strip() or None
    return None


def _print_evidence(label: str, evidence: WikiSourceEvidence | None) -> None:
    value = asdict(evidence) if evidence is not None else None
    print(f"{label}: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}")


def _print_diagnostics(
    document_id: str,
    quality: TextQualityAssessment,
    ocr_triggered: bool,
    result: PdfExtractionResult,
    focused_before: WikiSourceSelection | None,
    focused_after: WikiSourceSelection | None,
    before: WikiSourceEvidence | None,
    after: WikiSourceEvidence | None,
    max_pages: int,
) -> None:
    ocr_pages = [
        page.page_number
        for page in result.pages
        if page.page_number <= max_pages and page.ocr_text is not None
    ]
    print(f"document_id: {document_id}")
    print(f"native_quality: score={quality.score:.3f} degraded={str(quality.degraded).lower()}")
    print("degradation_reasons:")
    for reason in quality.reasons or ("none",):
        print(f"  - {reason}")
    print(f"ocr_triggered: {str(ocr_triggered).lower()}")
    print("pages_ocred: " + (", ".join(map(str, ocr_pages)) if ocr_pages else "none"))
    print("provenance:")
    for page in result.pages:
        if page.page_number <= max_pages:
            print(f"  page {page.page_number}: {page.provenance.value}")
    print("selected_source_provenance:")
    if focused_after is None:
        print("  none")
    else:
        for page in focused_after.pages:
            print(f"  page {page.page_number}: {page.provenance.value}")
    print("focused_source_before_ocr:")
    print(focused_before.source_text if focused_before is not None else "none")
    print("focused_source_after_ocr:")
    print(focused_after.source_text if focused_after is not None else "none")
    _print_evidence("evidence_before_ocr", before)
    _print_evidence("evidence_after_ocr", after)
    print(
        "detected_thai_abstract: "
        + json.dumps(
            _detected_abstract(focused_after, _THAI_ABSTRACT_HEADING),
            ensure_ascii=False,
        )
    )
    print(
        "detected_english_abstract: "
        + json.dumps(
            _detected_abstract(focused_after, _ENGLISH_ABSTRACT_HEADING),
            ensure_ascii=False,
        )
    )
    detected_keywords = list(after.keywords) if after is not None else []
    print("detected_keywords: " + json.dumps(detected_keywords, ensure_ascii=False))


def run_smoke_test(document_id: str, data_dir: Path, settings: Settings) -> int:
    """Run one indexed document through native extraction and conditional OCR."""

    try:
        entry = _find_document(data_dir, document_id)
        with (data_dir / entry.pdf).open("rb") as pdf_file:
            native = extract_pdf(pdf_file)
            quality = assess_front_matter_quality(_front_matter(native, settings.ocr_max_pages))
            focused_before = _focused_source(native)
            before = _evidence(focused_before)
            result = native
            ocr_triggered = False

            if quality.degraded:
                try:
                    provider = load_ocr_provider(settings)
                    if provider is None:
                        raise OcrConfigurationError(
                            "OCR_PROVIDER is disabled while degraded text requires OCR"
                        )
                    ocr_triggered = True
                    result = apply_ocr_fallback(
                        pdf_file,
                        native,
                        provider,
                        front_matter_page_limit=settings.ocr_max_pages,
                    )
                except OcrServiceError as error:
                    result = add_ocr_failure_warning(native, error)
    except (OSError, PdfExtractionError, ValueError) as error:
        print(f"OCR smoke test could not process {document_id}: {error}", file=sys.stderr)
        return 1

    focused_after = _focused_source(result)
    after = _evidence(focused_after)
    _print_diagnostics(
        document_id,
        quality,
        ocr_triggered,
        result,
        focused_before,
        focused_after,
        before,
        after,
        settings.ocr_max_pages,
    )

    ocr_warnings = [warning for warning in result.warnings if warning.code in _OCR_WARNING_CODES]
    if ocr_warnings:
        if any(warning.code is WarningCode.OCR_CONFIGURATION for warning in ocr_warnings):
            print("OCR configuration failure; native extraction was preserved.", file=sys.stderr)
            return 2
        print("OCR API/provider failure; native extraction was preserved.", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test conditional OCR for one indexed benchmark PDF without an LLM"
    )
    parser.add_argument("--document", required=True, help="Indexed ID such as document_064")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    arguments = parser.parse_args(argv)
    try:
        settings = get_settings()
    except ValueError:
        print("OCR configuration is invalid.", file=sys.stderr)
        return 2
    return run_smoke_test(arguments.document, arguments.data_dir, settings)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
