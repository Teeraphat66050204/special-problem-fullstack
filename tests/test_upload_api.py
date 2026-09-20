"""Endpoint checks for digital-PDF uploads with isolated persistence."""

import logging
import re
from pathlib import Path
from typing import BinaryIO

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.api import upload
from app.config import Settings
from app.db import get_session
from app.main import app
from app.models import Document, ExtractionStatus
from app.services.document_extraction import extraction_from_document
from app.services.pdf_extractor import (
    PdfExtractionError,
    PdfExtractionResult,
    PdfPageText,
    TextProvenance,
    extract_pdf,
)

pytestmark = pytest.mark.usefixtures("api_database")


def make_pdf(*page_texts: str | None) -> bytes:
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text is not None:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def make_encrypted_pdf() -> bytes:
    document = fitz.open()
    try:
        document.new_page().insert_text((72, 72), "Protected text")
        return document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    finally:
        document.close()


def post_pdf(data: bytes, filename: str = "project.pdf", content_type: str = "application/pdf"):
    return TestClient(app).post("/api/upload", files={"file": (filename, data, content_type)})


def test_successful_upload_creates_document_and_returns_id(api_database: Engine) -> None:
    response = post_pdf(make_pdf("Digital project text"), "../../project.pdf")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "document_id": data["document_id"],
        "filename": "project.pdf",
        "page_count": 1,
        "extraction_status": "completed",
        "warnings": [],
    }
    assert isinstance(data["document_id"], int)

    with Session(api_database) as session:
        document = session.get(Document, data["document_id"])
        assert document is not None
        assert document.original_filename == "project.pdf"
        assert document.page_count == 1
        assert document.raw_text == "Digital project text"
        assert document.extraction_status is ExtractionStatus.COMPLETED
        assert extraction_from_document(document).pages == (
            PdfPageText(page_number=1, text="Digital project text"),
        )


def test_upload_persists_supported_document_metadata(
    monkeypatch: pytest.MonkeyPatch, api_database: Engine
) -> None:
    title = "REMOTE CONSULTATION AND CONSULTATION APPLICATION ON iOS"
    extraction = PdfExtractionResult(
        page_count=4,
        full_text=f"{title}\nJutharat Tuayjan\n61050154\n2563",
        pages=(
            PdfPageText(1, f"{title}\n2563"),
            PdfPageText(2, f"{title}\nJutharat Tuayjan"),
            PdfPageText(3, "Student ID 61050154\nAdvisor: Dr Smith"),
            PdfPageText(4, "Abstract\nProject uses iOS."),
        ),
        warnings=(),
    )
    monkeypatch.setattr(upload, "extract_pdf", lambda source: extraction)

    response = post_pdf(make_pdf("Placeholder"))

    with Session(api_database) as session:
        document = session.get(Document, response.json()["document_id"])
        assert document is not None
        assert document.title == title
        assert document.author == "Jutharat Tuayjan"
        assert document.student_id == "61050154"
        assert document.academic_year == 2563


def test_upload_applies_configured_ocr_before_metadata_extraction(
    monkeypatch: pytest.MonkeyPatch, api_database: Engine
) -> None:
    corrupted = "ÖćøóĆçîćđĂÖÿćøĕì÷ ßČęĂîĆÖýċÖþć ĂćÝćø÷ŤìĊęðøċÖþć"
    ocr_text = (
        "หัวข้อโครงงาน ระบบตรวจสอบคุณภาพเอกสารภาษาไทย\n"
        "ชื่อนักศึกษา นาย สมชาย ใจดี 63050001\n"
        "อาจารย์ที่ปรึกษา ดร.ตัวอย่าง ทดสอบ\n"
        "ปีการศึกษา 2566\nบทคัดย่อ\nข้อความจากต้นฉบับ\nคำสำคัญ: เอกสาร"
    )

    class Provider:
        def extract_pages(self, source, page_numbers):
            assert tuple(page_numbers) == (1,)
            return {1: ocr_text}

    monkeypatch.setattr(
        upload,
        "extract_pdf",
        lambda source: PdfExtractionResult(
            page_count=1,
            full_text=corrupted,
            pages=(PdfPageText(1, corrupted),),
            warnings=(),
        ),
    )
    monkeypatch.setattr(upload, "load_ocr_provider", lambda settings: Provider())

    response = post_pdf(make_pdf("placeholder"), "ocr-project.pdf")

    assert response.status_code == 200
    with Session(api_database) as session:
        document = session.get(Document, response.json()["document_id"])
        assert document is not None
        assert document.title == "ระบบตรวจสอบคุณภาพเอกสารภาษาไทย"
        assert document.author == "นาย สมชาย ใจดี"
        assert document.student_id == "63050001"
        extraction = extraction_from_document(document)
        assert extraction.pages[0].native_text == corrupted
        assert extraction.pages[0].ocr_text == ocr_text
        assert extraction.pages[0].provenance is TextProvenance.OCR


def test_upload_preserves_native_text_and_returns_safe_warning_when_ocr_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    api_database: Engine,
) -> None:
    secret = "OCR_SECRET_MUST_NOT_APPEAR"
    native_text = "short title"

    class FailingProvider:
        def extract_pages(self, source, page_numbers):
            del source, page_numbers
            raise RuntimeError(f"provider failed with {secret}")

    monkeypatch.setattr(
        upload,
        "extract_pdf",
        lambda source: PdfExtractionResult(
            page_count=1,
            full_text=native_text,
            pages=(PdfPageText(1, native_text),),
            warnings=(),
        ),
    )
    monkeypatch.setattr(upload, "load_ocr_provider", lambda settings: FailingProvider())

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        response = post_pdf(make_pdf("placeholder"), "ocr-failure.pdf")

    assert response.status_code == 200
    assert response.json()["warnings"][-1]["code"] == "ocr_failed"
    assert secret not in response.text
    assert secret not in caplog.text
    assert any(
        message == "wiki.upload OCR fallback warning filename=ocr-failure.pdf code=ocr_failed"
        for message in caplog.messages
    )
    with Session(api_database) as session:
        document = session.get(Document, response.json()["document_id"])
        assert document is not None
        extraction = extraction_from_document(document)
        assert extraction.full_text == native_text
        assert extraction.pages == (PdfPageText(1, native_text),)


def test_multi_page_upload_preserves_pages_and_warnings(api_database: Engine) -> None:
    response = post_pdf(make_pdf("First page", None, "Third page"))

    assert response.status_code == 200
    data = response.json()
    assert data["page_count"] == 3
    assert data["warnings"][0]["code"] == "page_has_no_text"
    assert data["warnings"][0]["page_number"] == 2
    with Session(api_database) as session:
        document = session.get(Document, data["document_id"])
        assert document is not None
        extraction = extraction_from_document(document)
        assert extraction.full_text == "First page\n\nThird page"
        assert extraction.pages == (
            PdfPageText(page_number=1, text="First page"),
            PdfPageText(page_number=2, text=""),
            PdfPageText(page_number=3, text="Third page"),
        )


def test_upload_logs_upload_extraction_and_persistence_timings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = post_pdf(make_pdf("Timed extraction"), "timed.pdf")

    assert response.status_code == 200
    for stage in ("upload", "extraction", "persistence", "total"):
        assert any(
            re.fullmatch(rf"wiki\.upload filename=timed\.pdf {stage}=\d+\.\d{{3}}s", message)
            for message in caplog.messages
        )
    assert not any(
        stage in message
        for message in caplog.messages
        for stage in ("source_selection=", "llm_generation=", "validation=")
    )


def test_persistence_failure_rolls_back_logs_exception_and_returns_safe_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "RAW_PDF_TEXT_MUST_NOT_BE_LOGGED"
    failure = SQLAlchemyError("forced persistence failure")

    class FailingSession:
        rollback_called = False

        def add(self, document: Document) -> None:
            pass

        def commit(self) -> None:
            raise failure

        def refresh(self, document: Document) -> None:
            pytest.fail("refresh should not run after a failed commit")

        def rollback(self) -> None:
            self.rollback_called = True

    failing_session = FailingSession()

    def override_session():
        yield failing_session

    monkeypatch.setattr(
        upload,
        "extract_pdf",
        lambda source: PdfExtractionResult(
            page_count=1,
            full_text=raw_text,
            pages=(PdfPageText(page_number=1, text=raw_text),),
            warnings=(),
        ),
    )
    previous_override = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = override_session
    try:
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            response = post_pdf(make_pdf("placeholder"), "safe-name.pdf")
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_session, None)
        else:
            app.dependency_overrides[get_session] = previous_override

    assert failing_session.rollback_called is True
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not persist extracted document"}
    assert "forced persistence failure" not in response.text

    exception_records = [
        record
        for record in caplog.records
        if record.getMessage() == "wiki.upload persistence failed filename=safe-name.pdf"
    ]
    assert len(exception_records) == 1
    assert exception_records[0].exc_info is not None
    assert exception_records[0].exc_info[1] is failure
    assert raw_text not in caplog.text
    assert any(
        re.fullmatch(
            r"wiki\.upload filename=safe-name\.pdf persistence=\d+\.\d{3}s",
            message,
        )
        for message in caplog.messages
    )


def test_missing_file_is_rejected() -> None:
    response = TestClient(app).post("/api/upload")

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "file"]


def test_non_pdf_content_type_is_rejected() -> None:
    response = post_pdf(make_pdf("Valid PDF bytes"), content_type="text/plain")

    assert response.status_code == 415
    assert response.json() == {"detail": "Only application/pdf uploads are supported"}


def test_empty_pdf_upload_is_rejected() -> None:
    response = post_pdf(b"")

    assert response.status_code == 400
    assert response.json() == {"detail": "The uploaded file is empty"}


@pytest.mark.parametrize("data", [b"not a PDF", b"%PDF-1.7\ncorrupt data"])
def test_invalid_pdf_is_rejected(data: bytes) -> None:
    response = post_pdf(data)

    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


def test_encrypted_pdf_is_rejected() -> None:
    response = post_pdf(make_encrypted_pdf())

    assert response.status_code == 422
    assert "encrypted" in response.json()["detail"]


def test_upload_size_limit_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(
        upload, "get_settings", lambda: Settings(_env_file=None, max_upload_bytes=10)
    )

    response = post_pdf(make_pdf("Too large"))

    assert response.status_code == 413
    assert response.json() == {"detail": "PDF exceeds the maximum upload size"}


def test_upload_size_setting_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4096")

    assert Settings(_env_file=None).max_upload_bytes == 4096


def test_valid_pdf_without_text_returns_extractor_warnings() -> None:
    response = post_pdf(make_pdf(None))

    assert response.status_code == 200
    assert [warning["code"] for warning in response.json()["warnings"]] == [
        "page_has_no_text",
        "document_has_no_text",
    ]


def test_temporary_file_is_removed_after_success(monkeypatch) -> None:
    captured_path: Path | None = None

    def inspect_path(source: BinaryIO) -> PdfExtractionResult:
        nonlocal captured_path
        path = Path(source.name)
        captured_path = path
        assert path.is_file()
        assert path.name == "upload.pdf"
        return extract_pdf(source)

    monkeypatch.setattr(upload, "extract_pdf", inspect_path)

    assert post_pdf(make_pdf("Temporary text")).status_code == 200
    assert captured_path is not None
    assert not captured_path.exists()
    assert not captured_path.parent.exists()


def test_temporary_file_is_removed_after_extraction_failure(monkeypatch) -> None:
    captured_path: Path | None = None

    def fail_extraction(source: BinaryIO) -> PdfExtractionResult:
        nonlocal captured_path
        path = Path(source.name)
        captured_path = path
        assert path.is_file()
        raise PdfExtractionError("Unexpected page failure")

    monkeypatch.setattr(upload, "extract_pdf", fail_extraction)

    response = post_pdf(make_pdf("Temporary text"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Could not extract PDF text"}
    assert captured_path is not None
    assert not captured_path.exists()


def test_thai_text_is_preserved_in_persisted_extraction(monkeypatch, api_database: Engine) -> None:
    thai_text = "การพัฒนาระบบ\u200bค้นคืนข้อมูล"

    def extract_thai(source: BinaryIO) -> PdfExtractionResult:
        path = Path(source.name)
        assert path.is_file()
        return PdfExtractionResult(
            page_count=1,
            full_text=thai_text,
            pages=(PdfPageText(page_number=1, text=thai_text),),
            warnings=(),
        )

    monkeypatch.setattr(upload, "extract_pdf", extract_thai)

    response = post_pdf(make_pdf("Source text"))
    assert response.status_code == 200
    with Session(api_database) as session:
        document = session.get(Document, response.json()["document_id"])
        assert document is not None
        extraction = extraction_from_document(document)
        assert extraction.full_text == thai_text
        assert extraction.pages[0].text == thai_text
