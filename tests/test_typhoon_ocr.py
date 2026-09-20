"""Tests for the real Typhoon OCR API provider and safe fallback behavior."""

from __future__ import annotations

import logging
from io import BytesIO

import fitz
import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.ocr_service import OcrServiceError, apply_ocr_fallback, load_ocr_provider
from app.services.pdf_extractor import (
    PdfExtractionResult,
    PdfPageText,
    TextProvenance,
    WarningCode,
)
from app.services.typhoon_ocr import TyphoonOcrProvider

_SECRET = "test-typhoon-key-must-stay-secret"
_CORRUPTED_THAI = (
    "Ã–Ä‡Ã¸Ã³Ä†Ã§Ã®Ä‡Ä‘Ä‚Ã–Ã¿Ä‡Ã¸Ä•Ã¬Ã· ÃŸÄŒÄ™Ä‚Ã®Ä†Ã–Ã½Ä‹Ã–Ã¾Ä‡ Ä‚Ä‡ÃÄ‡Ã¸Ã·Å¤Ã¬ÄŠÄ™Ã°Ã¸Ä‹Ã–Ã¾Ä‡"
)
_CLEAN_THAI = (
    "หัวข้อโครงงาน ระบบตรวจสอบคุณภาพเอกสารภาษาไทย\n"
    "ชื่อนักศึกษา นาย สมชาย ใจดี 63050001\n"
    "อาจารย์ที่ปรึกษา ดร.ตัวอย่าง ทดสอบ\n"
    "ปีการศึกษา 2566\n"
    "บทคัดย่อ\nระบบนี้ตรวจสอบข้อความจากเอกสารภาษาไทย\n"
    "คำสำคัญ: เอกสาร, ภาษาไทย"
)


def _pdf_bytes(page_count: int = 1) -> bytes:
    document = fitz.open()
    try:
        for page_number in range(1, page_count + 1):
            document.new_page().insert_text((72, 72), f"Page {page_number}")
        return document.tobytes()
    finally:
        document.close()


def _extraction(text: str, page_count: int = 1) -> PdfExtractionResult:
    pages = tuple(PdfPageText(page_number, text) for page_number in range(1, page_count + 1))
    return PdfExtractionResult(
        page_count=page_count,
        full_text="\n\n".join(page.text for page in pages),
        pages=pages,
        warnings=(),
    )


def _provider(transport: httpx.BaseTransport, key: str = _SECRET) -> TyphoonOcrProvider:
    return TyphoonOcrProvider(
        api_key=SecretStr(key),
        base_url="https://api.opentyphoon.invalid/v1",
        model="typhoon-ocr",
        timeout_seconds=120,
        transport=transport,
    )


def test_typhoon_api_key_and_configuration_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "typhoon")
    monkeypatch.setenv("TYPHOON_API_KEY", _SECRET)
    monkeypatch.setenv("TYPHOON_BASE_URL", "https://ocr.example/v1")
    monkeypatch.setenv("TYPHOON_OCR_MODEL", "typhoon-ocr-test")
    monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("OCR_MAX_PAGES", "6")

    settings = Settings(_env_file=None)
    provider = load_ocr_provider(settings)

    assert isinstance(provider, TyphoonOcrProvider)
    assert provider.api_key is not None
    assert provider.api_key.get_secret_value() == _SECRET
    assert provider.base_url == "https://ocr.example/v1"
    assert provider.model == "typhoon-ocr-test"
    assert provider.timeout_seconds == 120
    assert settings.ocr_max_pages == 6
    assert _SECRET not in repr(settings)
    assert _SECRET not in repr(provider)


def test_clean_native_text_does_not_call_typhoon() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Typhoon must not be called for clean text: {request.url}")

    provider = _provider(httpx.MockTransport(unexpected_request))
    native = _extraction(_CLEAN_THAI)

    result = apply_ocr_fallback(BytesIO(b"not-read"), native, provider)

    assert result is native


def test_degraded_text_calls_typhoon_and_records_ocr_provenance() -> None:
    requests: list[httpx.Request] = []

    def successful_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _CLEAN_THAI}}]},
        )

    native = _extraction(_CORRUPTED_THAI)
    result = apply_ocr_fallback(
        BytesIO(_pdf_bytes()),
        native,
        _provider(httpx.MockTransport(successful_request)),
    )

    assert len(requests) == 1
    assert requests[0].url == "https://api.opentyphoon.invalid/v1/chat/completions"
    assert requests[0].headers["authorization"].endswith(_SECRET)
    assert result.pages[0].text == _CLEAN_THAI
    assert result.pages[0].native_text == _CORRUPTED_THAI
    assert result.pages[0].ocr_text == _CLEAN_THAI
    assert result.pages[0].provenance is TextProvenance.OCR
    assert result.warnings == ()


def test_typhoon_failure_preserves_native_text_and_never_exposes_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failed_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, text="provider unavailable")

    native = _extraction(_CORRUPTED_THAI)
    with caplog.at_level(logging.DEBUG):
        result = apply_ocr_fallback(
            BytesIO(_pdf_bytes()),
            native,
            _provider(httpx.MockTransport(failed_request)),
        )

    assert result.full_text == native.full_text
    assert result.pages == native.pages
    assert result.warnings[-1].code is WarningCode.OCR_FAILED
    assert _SECRET not in result.warnings[-1].message
    assert _SECRET not in caplog.text


def test_missing_typhoon_key_warns_only_when_degraded_text_needs_ocr() -> None:
    provider = TyphoonOcrProvider(
        api_key=None,
        base_url="https://api.opentyphoon.invalid/v1",
        model="typhoon-ocr",
        timeout_seconds=120,
    )

    clean = _extraction(_CLEAN_THAI)
    assert apply_ocr_fallback(BytesIO(b"not-read"), clean, provider) is clean

    degraded = _extraction(_CORRUPTED_THAI)
    result = apply_ocr_fallback(BytesIO(_pdf_bytes()), degraded, provider)
    assert result.pages == degraded.pages
    assert result.warnings[-1].code is WarningCode.OCR_CONFIGURATION
    assert "TYPHOON_API_KEY" in result.warnings[-1].message


def test_typhoon_provider_rejects_pages_after_six_without_api_call() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Out-of-scope OCR request was sent: {request.url}")

    provider = _provider(httpx.MockTransport(unexpected_request))

    with pytest.raises(OcrServiceError, match="between 1 and 6"):
        provider.extract_pages(BytesIO(_pdf_bytes(page_count=7)), (7,))
