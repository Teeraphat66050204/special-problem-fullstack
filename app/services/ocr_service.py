"""Provider-neutral, quality-gated OCR enrichment for PDF front matter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from importlib import import_module
from typing import Protocol, runtime_checkable

from app.config import Settings
from app.services.pdf_extractor import (
    PdfExtractionResult,
    PdfExtractionWarning,
    PdfInput,
    PdfPageText,
    TextProvenance,
    WarningCode,
    normalize_text,
)
from app.services.text_quality import assess_front_matter_quality, assess_text_quality


class OcrServiceError(RuntimeError):
    """OCR configuration or provider execution failed."""


class OcrConfigurationError(OcrServiceError):
    """OCR was needed but its provider configuration was incomplete."""


@runtime_checkable
class OcrProvider(Protocol):
    """Adapter implemented by a Typhoon or other OCR integration."""

    def extract_pages(
        self,
        source: PdfInput,
        page_numbers: Sequence[int],
    ) -> Mapping[int, str]:
        """OCR exactly the requested one-based PDF pages and return exact text."""


def load_ocr_provider(settings: Settings) -> OcrProvider | None:
    """Load the built-in Typhoon adapter or an optional provider factory."""

    provider_path = settings.ocr_provider.strip()
    if provider_path.casefold() in {"", "disabled", "none", "off"}:
        return None
    if provider_path.casefold() == "typhoon":
        from app.services.typhoon_ocr import TyphoonOcrProvider

        return TyphoonOcrProvider.from_settings(settings)
    if ":" not in provider_path:
        raise OcrConfigurationError(
            "OCR_PROVIDER must be 'disabled', 'typhoon', or a module:factory path"
        )
    module_name, factory_name = provider_path.rsplit(":", 1)
    try:
        factory = getattr(import_module(module_name), factory_name)
        provider = factory(settings)
    except Exception as exc:
        raise OcrConfigurationError("Could not load the configured OCR provider") from exc
    if not isinstance(provider, OcrProvider):
        raise OcrConfigurationError("Configured OCR factory did not return an OCR provider")
    return provider


def _restore_source_position(source: PdfInput, original_position: int | None) -> None:
    if original_position is not None and hasattr(source, "seek"):
        with suppress(OSError, ValueError):
            source.seek(original_position)


def _run_provider(
    provider: OcrProvider,
    source: PdfInput,
    page_numbers: Sequence[int],
) -> Mapping[int, str]:
    original_position: int | None = None
    try:
        if hasattr(source, "tell"):
            original_position = source.tell()
        if hasattr(source, "seek"):
            source.seek(0)
        result = provider.extract_pages(source, page_numbers)
    except OcrServiceError:
        raise
    except Exception as exc:
        raise OcrServiceError("OCR provider failed for selected front-matter pages") from exc
    finally:
        _restore_source_position(source, original_position)
    if not isinstance(result, Mapping):
        raise OcrServiceError("OCR provider returned an invalid page mapping")
    return result


def add_ocr_failure_warning(
    extraction: PdfExtractionResult,
    error: OcrServiceError,
) -> PdfExtractionResult:
    """Preserve native extraction and add a safe, non-secret OCR warning."""

    if isinstance(error, OcrConfigurationError):
        code = WarningCode.OCR_CONFIGURATION
        message = (
            "Typhoon OCR is enabled but not configured; native PDF text was preserved. "
            "Set TYPHOON_API_KEY before retrying OCR."
        )
    else:
        code = WarningCode.OCR_FAILED
        message = "OCR fallback failed; native PDF text was preserved."
    return PdfExtractionResult(
        page_count=extraction.page_count,
        full_text=extraction.full_text,
        pages=extraction.pages,
        warnings=(*extraction.warnings, PdfExtractionWarning(code=code, message=message)),
    )


def _select_page_text(page: PdfPageText, ocr_text: str) -> PdfPageText:
    native_text = page.native_text if page.native_text is not None else page.text
    normalized_ocr = normalize_text(ocr_text)
    if not normalized_ocr:
        return PdfPageText(
            page_number=page.page_number,
            text=page.text,
            native_text=native_text,
            ocr_text=normalized_ocr,
            provenance=TextProvenance.PYMUPDF,
        )

    expected_thai = page.page_number in {1, 3, 4}
    native_quality = assess_text_quality(page.text, expected_thai_metadata=expected_thai)
    ocr_quality = assess_text_quality(normalized_ocr, expected_thai_metadata=expected_thai)
    if ocr_quality.score >= native_quality.score + 0.1 and (
        native_quality.degraded or not ocr_quality.degraded
    ):
        return PdfPageText(
            page_number=page.page_number,
            text=normalized_ocr,
            native_text=native_text,
            ocr_text=normalized_ocr,
            provenance=TextProvenance.OCR,
        )

    combined = normalize_text(f"{page.text}\n\n{normalized_ocr}")
    combined_quality = assess_text_quality(combined, expected_thai_metadata=expected_thai)
    if (
        native_quality.degraded
        and ocr_quality.degraded
        and combined_quality.score >= max(native_quality.score, ocr_quality.score) + 0.1
    ):
        return PdfPageText(
            page_number=page.page_number,
            text=combined,
            native_text=native_text,
            ocr_text=normalized_ocr,
            provenance=TextProvenance.MIXED,
        )

    return PdfPageText(
        page_number=page.page_number,
        text=page.text,
        native_text=native_text,
        ocr_text=normalized_ocr,
        provenance=TextProvenance.PYMUPDF,
    )


def apply_ocr_fallback(
    source: PdfInput,
    extraction: PdfExtractionResult,
    provider: OcrProvider | None,
    *,
    front_matter_page_limit: int = 6,
) -> PdfExtractionResult:
    """OCR degraded front matter only and retain the better exact page text."""

    if provider is None:
        return extraction
    if not 1 <= front_matter_page_limit <= 6:
        raise ValueError("OCR front-matter page limit must be between 1 and 6")

    front_matter = tuple(
        page for page in extraction.pages if page.page_number <= front_matter_page_limit
    )
    quality = assess_front_matter_quality(front_matter)
    if not quality.degraded:
        return extraction

    requested_pages = quality.degraded_page_numbers
    try:
        ocr_pages = _run_provider(provider, source, requested_pages)
        unexpected = set(ocr_pages) - set(requested_pages)
        if unexpected:
            raise OcrServiceError("OCR provider returned unrequested page numbers")
        if any(not isinstance(value, str) for value in ocr_pages.values()):
            raise OcrServiceError("OCR provider returned non-text page content")
    except OcrServiceError as error:
        return add_ocr_failure_warning(extraction, error)

    selected_pages: list[PdfPageText] = []
    for page in extraction.pages:
        if page.page_number not in ocr_pages:
            selected_pages.append(page)
            continue
        selected_pages.append(_select_page_text(page, ocr_pages[page.page_number]))

    full_text = normalize_text("\n\n".join(page.text for page in selected_pages))
    usable_ocr_pages = {
        page.page_number
        for page in selected_pages
        if page.provenance is not TextProvenance.PYMUPDF and page.text
    }
    warnings = tuple(
        warning
        for warning in extraction.warnings
        if not (
            warning.code is WarningCode.PAGE_HAS_NO_TEXT and warning.page_number in usable_ocr_pages
        )
        and not (warning.code is WarningCode.DOCUMENT_HAS_NO_TEXT and full_text)
    )
    return PdfExtractionResult(
        page_count=extraction.page_count,
        full_text=full_text,
        pages=tuple(selected_pages),
        warnings=warnings,
    )
