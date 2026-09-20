"""Typhoon OCR API adapter for the provider-neutral OCR fallback service."""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import fitz
import httpx
from pydantic import SecretStr

from app.config import Settings
from app.services.ocr_service import OcrConfigurationError, OcrServiceError
from app.services.pdf_extractor import PdfInput

_MAX_OCR_PAGES = 6
_TARGET_IMAGE_DIMENSION = 1800
_OCR_PROMPT = """Extract all text from this document page.

Return only clean Markdown with no explanation or surrounding code fence. Include all visible
information in reading order. Preserve names, identifiers, spelling, and wording exactly as shown;
do not translate, correct, infer, or invent text. Use Markdown for headings and tables where the
layout clearly supports them."""


@dataclass(slots=True)
class TyphoonOcrProvider:
    """Call Typhoon's OpenAI-compatible OCR endpoint one PDF page at a time."""

    api_key: SecretStr | None
    base_url: str
    model: str
    timeout_seconds: float
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> TyphoonOcrProvider:
        """Build a provider without requiring the key until OCR is actually needed."""

        return cls(
            api_key=settings.typhoon_api_key,
            base_url=settings.typhoon_base_url,
            model=settings.typhoon_ocr_model,
            timeout_seconds=settings.ocr_timeout_seconds,
        )

    def extract_pages(
        self,
        source: PdfInput,
        page_numbers: Sequence[int],
    ) -> Mapping[int, str]:
        """Render and OCR only the requested one-based front-matter pages."""

        api_key = self._required_api_key()
        requested_pages = tuple(dict.fromkeys(page_numbers))
        if any(page < 1 or page > _MAX_OCR_PAGES for page in requested_pages):
            raise OcrServiceError("Typhoon OCR page numbers must be between 1 and 6")
        if not requested_pages:
            return {}

        pdf_bytes = _read_pdf_bytes(source)
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise OcrServiceError("Typhoon OCR could not read the PDF") from exc

        try:
            if any(page > document.page_count for page in requested_pages):
                raise OcrServiceError("Typhoon OCR page number exceeds the PDF page count")
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                return {
                    page_number: self._ocr_page(client, document, page_number, api_key)
                    for page_number in requested_pages
                }
        finally:
            document.close()

    def _required_api_key(self) -> str:
        value = self.api_key.get_secret_value().strip() if self.api_key is not None else ""
        if not value:
            raise OcrConfigurationError(
                "Typhoon OCR is enabled but TYPHOON_API_KEY is not configured"
            )
        return value

    def _ocr_page(
        self,
        client: httpx.Client,
        document: fitz.Document,
        page_number: int,
        api_key: str,
    ) -> str:
        image = _render_page(document, page_number)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(image).decode()}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 16384,
            "temperature": 0.1,
            "top_p": 0.6,
            "repetition_penalty": 1.1,
        }
        try:
            response = client.post(
                _chat_completions_url(self.base_url),
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty OCR response")
            return content
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise OcrServiceError("Typhoon OCR request failed") from exc


def _read_pdf_bytes(source: PdfInput) -> bytes:
    if isinstance(source, (str, os.PathLike)):
        try:
            return Path(source).read_bytes()
        except OSError as exc:
            raise OcrServiceError("Typhoon OCR could not read the PDF") from exc
    try:
        value = source.read()
    except (OSError, ValueError) as exc:
        raise OcrServiceError("Typhoon OCR could not read the PDF") from exc
    if not isinstance(value, bytes):
        raise OcrServiceError("Typhoon OCR requires a binary PDF source")
    return value


def _render_page(document: fitz.Document, page_number: int) -> bytes:
    page = document.load_page(page_number - 1)
    longest_dimension = max(page.rect.width, page.rect.height)
    scale = _TARGET_IMAGE_DIMENSION / longest_dimension if longest_dimension else 1.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    return pixmap.tobytes("png")


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise OcrConfigurationError("TYPHOON_BASE_URL must not be empty")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"
