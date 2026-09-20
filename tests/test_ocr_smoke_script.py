"""CLI-level behavior for the OCR fallback smoke-test script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.pdf_extractor import PdfExtractionResult, PdfPageText
from scripts import test_ocr_fallback as smoke

_CORRUPTED = (
    "Ã–Ä‡Ã¸Ã³Ä†Ã§Ã®Ä‡Ä‘Ä‚Ã–Ã¿Ä‡Ã¸Ä•Ã¬Ã· ÃŸÄŒÄ™Ä‚Ã®Ä†Ã–Ã½Ä‹Ã–Ã¾Ä‡ Ä‚Ä‡ÃÄ‡Ã¸Ã·Å¤Ã¬ÄŠÄ™Ã°Ã¸Ä‹Ã–Ã¾Ä‡"
)
_CLEAN = (
    "หัวข้อโครงงาน ระบบตรวจสอบคุณภาพเอกสารภาษาไทย\n"
    "ชื่อนักศึกษา นาย สมชาย ใจดี 63050001\n"
    "อาจารย์ที่ปรึกษา ดร.ตัวอย่าง ทดสอบ\n"
    "ปีการศึกษา 2566\nบทคัดย่อ\nข้อความจากเอกสาร\nคำสำคัญ: เอกสาร"
)


def _data_dir(tmp_path: Path) -> Path:
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "document_064.pdf").write_bytes(b"placeholder")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [
                    {
                        "document_id": "document_064",
                        "pdf": "sample/document_064.pdf",
                        "groundtruth": "groundtruth/document_064.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _extraction(text: str) -> PdfExtractionResult:
    return PdfExtractionResult(
        page_count=1,
        full_text=text,
        pages=(PdfPageText(1, text),),
        warnings=(),
    )


def test_clean_document_skips_provider_and_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(smoke, "extract_pdf", lambda source: _extraction(_CLEAN))
    monkeypatch.setattr(
        smoke,
        "load_ocr_provider",
        lambda settings: pytest.fail("clean text must not load the OCR provider"),
    )

    status = smoke.run_smoke_test(
        "document_064",
        _data_dir(tmp_path),
        Settings(_env_file=None, ocr_provider="disabled"),
    )

    output = capsys.readouterr()
    assert status == 0
    assert "degraded=false" in output.out
    assert "ocr_triggered: false" in output.out
    assert "pages_ocred: none" in output.out
    assert "page 1: pymupdf" in output.out


def test_degraded_document_runs_provider_and_reports_ocr_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Provider:
        def extract_pages(self, source, page_numbers):
            assert tuple(page_numbers) == (1,)
            return {1: _CLEAN}

    monkeypatch.setattr(smoke, "extract_pdf", lambda source: _extraction(_CORRUPTED))
    monkeypatch.setattr(smoke, "load_ocr_provider", lambda settings: Provider())

    status = smoke.run_smoke_test(
        "document_064",
        _data_dir(tmp_path),
        Settings(_env_file=None, ocr_provider="typhoon"),
    )

    output = capsys.readouterr()
    assert status == 0
    assert "degraded=true" in output.out
    assert "ocr_triggered: true" in output.out
    assert "pages_ocred: 1" in output.out
    assert "page 1: ocr" in output.out
    assert "focused_source_before_ocr:" in output.out
    assert "focused_source_after_ocr:" in output.out
    assert "evidence_before_ocr:" in output.out
    assert "evidence_after_ocr:" in output.out
    assert "detected_thai_abstract:" in output.out
    assert "detected_english_abstract:" in output.out
    assert 'detected_keywords: ["เอกสาร"]' in output.out


def test_provider_failure_is_safe_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "SECRET_MUST_NOT_BE_PRINTED"

    class Provider:
        def extract_pages(self, source, page_numbers):
            del source, page_numbers
            raise RuntimeError(secret)

    monkeypatch.setattr(smoke, "extract_pdf", lambda source: _extraction(_CORRUPTED))
    monkeypatch.setattr(smoke, "load_ocr_provider", lambda settings: Provider())

    status = smoke.run_smoke_test(
        "document_064",
        _data_dir(tmp_path),
        Settings(_env_file=None, ocr_provider="typhoon"),
    )

    output = capsys.readouterr()
    assert status == 3
    assert "ocr_triggered: true" in output.out
    assert "page 1: pymupdf" in output.out
    assert "OCR API/provider failure" in output.err
    assert secret not in output.out
    assert secret not in output.err
