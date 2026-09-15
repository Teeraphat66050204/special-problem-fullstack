"""Endpoint checks for digital-PDF uploads without a database."""

from pathlib import Path
from typing import BinaryIO

import fitz
import pytest
from fastapi.testclient import TestClient

from app.api import upload
from app.config import Settings
from app.main import app
from app.services.pdf_extractor import (
    PdfExtractionError,
    PdfExtractionResult,
    PdfPageText,
    extract_pdf,
)


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


def test_successful_upload_uses_safe_display_filename() -> None:
    response = post_pdf(make_pdf("Digital project text"), "../../project.pdf")

    assert response.status_code == 200
    assert response.json() == {
        "filename": "project.pdf",
        "page_count": 1,
        "raw_text": "Digital project text",
        "pages": [{"page_number": 1, "text": "Digital project text"}],
        "warnings": [],
    }


def test_multi_page_upload_preserves_pages_and_warnings() -> None:
    response = post_pdf(make_pdf("First page", None, "Third page"))

    assert response.status_code == 200
    data = response.json()
    assert data["page_count"] == 3
    assert data["raw_text"] == "First page\n\nThird page"
    assert data["pages"] == [
        {"page_number": 1, "text": "First page"},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "Third page"},
    ]
    assert data["warnings"][0]["code"] == "page_has_no_text"
    assert data["warnings"][0]["page_number"] == 2


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
    assert response.json()["raw_text"] == ""
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


def test_thai_text_is_preserved_in_api_response(monkeypatch) -> None:
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
    assert response.json()["raw_text"] == thai_text
    assert response.json()["pages"][0]["text"] == thai_text
