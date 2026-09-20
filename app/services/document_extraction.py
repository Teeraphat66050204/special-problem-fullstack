"""Serialize page-aware PDF extraction data stored on a Document."""

from __future__ import annotations

import json
from typing import Any

from app.models import Document
from app.services.pdf_extractor import (
    PdfExtractionResult,
    PdfExtractionWarning,
    PdfPageText,
    TextProvenance,
    WarningCode,
)

_FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = frozenset((1, _FORMAT_VERSION))


class StoredExtractionError(ValueError):
    """A completed Document lacks valid page-aware extraction data."""


def serialize_extraction(result: PdfExtractionResult) -> str:
    """Store only page provenance and warnings; raw text remains a model column."""

    payload = {
        "version": _FORMAT_VERSION,
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.text,
                "native_text": page.native_text,
                "ocr_text": page.ocr_text,
                "provenance": page.provenance.value,
            }
            for page in result.pages
        ],
        "warnings": [
            {
                "code": warning.code.value,
                "message": warning.message,
                "page_number": warning.page_number,
            }
            for warning in result.warnings
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _payload(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StoredExtractionError("Stored extraction data is invalid") from exc
    if not isinstance(payload, dict) or payload.get("version") not in _SUPPORTED_FORMAT_VERSIONS:
        raise StoredExtractionError("Stored extraction data has an unsupported format")
    return payload


def extraction_from_document(document: Document) -> PdfExtractionResult:
    """Reconstruct the extractor result required by focused source selection."""

    if document.raw_text is None or document.extraction_data is None:
        raise StoredExtractionError("Stored extraction data is incomplete")
    payload = _payload(document.extraction_data)
    try:
        version = int(payload["version"])
        pages = tuple(
            PdfPageText(
                page_number=int(item["page_number"]),
                text=str(item["text"]),
                native_text=(
                    str(item["native_text"])
                    if version >= 2 and item.get("native_text") is not None
                    else None
                ),
                ocr_text=(
                    str(item["ocr_text"])
                    if version >= 2 and item.get("ocr_text") is not None
                    else None
                ),
                provenance=(
                    TextProvenance(item["provenance"]) if version >= 2 else TextProvenance.PYMUPDF
                ),
            )
            for item in payload["pages"]
        )
        warnings = tuple(
            PdfExtractionWarning(
                code=WarningCode(item["code"]),
                message=str(item["message"]),
                page_number=(
                    int(item["page_number"]) if item.get("page_number") is not None else None
                ),
            )
            for item in payload["warnings"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StoredExtractionError("Stored extraction data is invalid") from exc

    expected_pages = tuple(range(1, document.page_count + 1))
    if tuple(page.page_number for page in pages) != expected_pages:
        raise StoredExtractionError("Stored extraction page provenance is inconsistent")
    return PdfExtractionResult(
        page_count=document.page_count,
        full_text=document.raw_text,
        pages=pages,
        warnings=warnings,
    )
