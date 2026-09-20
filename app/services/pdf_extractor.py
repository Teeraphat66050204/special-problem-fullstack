"""Text-layer extraction for digital PDF documents.

This module deliberately has no database or API dependencies. Callers decide
where files are stored and whether the returned text should be persisted.
"""

from __future__ import annotations

import os
import re
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, TypeAlias

import fitz

PdfInput: TypeAlias = str | os.PathLike[str] | BinaryIO

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHENATED_LATIN_LINE = re.compile(r"[A-Za-z]-$")
_LIST_ITEM = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s")
_MIN_WRAPPED_LINE_LENGTH = 40
_TERMINAL_PUNCTUATION = frozenset(".!?\u2026:;")


class PdfExtractionError(Exception):
    """Base exception for expected PDF extraction failures."""


class PdfInputError(PdfExtractionError):
    """Raised when a source path or file-like value cannot be read."""


class InvalidPdfError(PdfExtractionError):
    """Raised when the supplied data is not a readable PDF."""


class EncryptedPdfError(PdfExtractionError):
    """Raised when a PDF requires a password."""


class EmptyPdfError(PdfExtractionError):
    """Raised when a valid PDF contains no pages."""


class WarningCode(StrEnum):
    """Non-fatal conditions encountered during extraction."""

    PAGE_HAS_NO_TEXT = "page_has_no_text"
    DOCUMENT_HAS_NO_TEXT = "document_has_no_text"
    OCR_CONFIGURATION = "ocr_configuration"
    OCR_FAILED = "ocr_failed"


class TextProvenance(StrEnum):
    """Origin of the selected text retained for one PDF page."""

    PYMUPDF = "pymupdf"
    OCR = "ocr"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class PdfPageText:
    """Cleaned text and one-based provenance for a source page."""

    page_number: int
    text: str
    native_text: str | None = None
    ocr_text: str | None = None
    provenance: TextProvenance = TextProvenance.PYMUPDF


@dataclass(frozen=True, slots=True)
class PdfExtractionWarning:
    """A non-fatal extraction warning, optionally tied to a page."""

    code: WarningCode
    message: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class PdfExtractionResult:
    """Complete text-layer extraction result."""

    page_count: int
    full_text: str
    pages: tuple[PdfPageText, ...]
    warnings: tuple[PdfExtractionWarning, ...]


def normalize_text(text: str) -> str:
    """Conservatively normalize extracted text without rewriting language.

    NFC preserves the semantic content of Thai text while normalizing equivalent
    Unicode sequences. Zero-width joiners and non-joiners are intentionally kept
    because removing them can change the meaning of some writing systems.
    """

    normalized = unicodedata.normalize("NFC", text).lstrip("\ufeff")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")
    )
    normalized = _EXCESSIVE_BLANK_LINES.sub("\n\n", normalized)
    return _join_wrapped_lines(normalized).strip()


def _join_wrapped_lines(text: str) -> str:
    """Join only line breaks that strongly resemble prose wrapping."""

    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        lines = paragraph.splitlines()
        if not lines:
            continue

        joined_lines = [lines[0]]
        for following in lines[1:]:
            current = joined_lines[-1]
            if _is_hyphenated_wrap(current, following):
                joined_lines[-1] = current[:-1] + following
            elif _is_prose_wrap(current, following):
                joined_lines[-1] = f"{current} {following}"
            else:
                joined_lines.append(following)
        paragraphs.append("\n".join(joined_lines))

    return "\n\n".join(paragraphs)


def _is_hyphenated_wrap(current: str, following: str) -> bool:
    return (
        len(current) >= _MIN_WRAPPED_LINE_LENGTH
        and bool(_HYPHENATED_LATIN_LINE.search(current))
        and bool(following)
        and following[0].islower()
    )


def _is_prose_wrap(current: str, following: str) -> bool:
    return (
        len(current) >= _MIN_WRAPPED_LINE_LENGTH
        and bool(following)
        and current[-1] not in _TERMINAL_PUNCTUATION
        and not _LIST_ITEM.match(following)
        and following[0].islower()
    )


def extract_pdf(source: PdfInput) -> PdfExtractionResult:
    """Extract and normalize text from every page of a digital PDF.

    ``source`` may be a path or a binary file-like object such as
    ``UploadFile.file``. Password-protected documents are rejected and OCR is not
    attempted. Pages without a usable text layer remain present in ``pages`` and
    produce warnings so their page numbers are not lost.
    """

    document = _open_pdf(source)
    try:
        if document.needs_pass:
            raise EncryptedPdfError("The PDF is encrypted and requires a password.")

        page_count = document.page_count
        if page_count == 0:
            raise EmptyPdfError("The PDF contains no pages.")

        pages: list[PdfPageText] = []
        warnings: list[PdfExtractionWarning] = []

        for page_index in range(page_count):
            page_number = page_index + 1
            try:
                raw_text = document.load_page(page_index).get_text("text", sort=True)
            except (RuntimeError, ValueError) as exc:
                raise PdfExtractionError(
                    f"Could not extract text from PDF page {page_number}."
                ) from exc

            text = normalize_text(raw_text)
            pages.append(PdfPageText(page_number=page_number, text=text))
            if not text:
                warnings.append(
                    PdfExtractionWarning(
                        code=WarningCode.PAGE_HAS_NO_TEXT,
                        page_number=page_number,
                        message=f"Page {page_number} has no usable native PDF text layer.",
                    )
                )

        full_text = normalize_text("\n\n".join(page.text for page in pages))
        if not full_text:
            warnings.append(
                PdfExtractionWarning(
                    code=WarningCode.DOCUMENT_HAS_NO_TEXT,
                    message="The PDF has no usable native text layer on any page.",
                )
            )

        return PdfExtractionResult(
            page_count=page_count,
            full_text=full_text,
            pages=tuple(pages),
            warnings=tuple(warnings),
        )
    finally:
        document.close()


def _open_pdf(source: PdfInput) -> fitz.Document:
    try:
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if not path.is_file():
                raise PdfInputError(f"PDF file does not exist: {path}")
            return fitz.open(filename=path)

        data = _read_binary_source(source)
        if not data:
            raise InvalidPdfError("The supplied PDF data is empty.")
        return fitz.open(stream=data, filetype="pdf")
    except PdfExtractionError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError, TypeError) as exc:
        raise InvalidPdfError("The supplied file is not a valid readable PDF.") from exc


def _read_binary_source(source: BinaryIO) -> bytes:
    if not hasattr(source, "read"):
        raise PdfInputError("PDF source must be a filesystem path or binary file-like object.")

    original_position: int | None = None
    try:
        if hasattr(source, "tell"):
            original_position = source.tell()
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
    except (OSError, ValueError) as exc:
        raise PdfInputError("The PDF source could not be read.") from exc
    finally:
        if original_position is not None and hasattr(source, "seek"):
            with suppress(OSError, ValueError):
                source.seek(original_position)

    if not isinstance(data, bytes):
        raise PdfInputError("The PDF file-like source must return bytes.")
    return data
