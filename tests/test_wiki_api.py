"""Persisted-document Draft Wiki API tests without a live Ollama service."""

import logging
import re
from pathlib import Path
from typing import BinaryIO

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.api import upload, wiki
from app.main import app
from app.models import Document, ExtractionStatus
from app.prompts import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS
from app.services import llm_service
from app.services.document_extraction import serialize_extraction
from app.services.pdf_extractor import (
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
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def post_upload(data: bytes, filename: str = "project.pdf", content_type: str = "application/pdf"):
    return TestClient(app).post("/api/upload", files={"file": (filename, data, content_type)})


def post_document(document_id: int):
    return TestClient(app).post("/api/wiki/generate", json={"document_id": document_id})


def post_pdf(data: bytes, filename: str = "project.pdf", content_type: str = "application/pdf"):
    upload_response = post_upload(data, filename, content_type)
    if upload_response.status_code != 200:
        return upload_response
    return post_document(upload_response.json()["document_id"])


def wiki_markdown(title: str, *, overview: str = MISSING_INFORMATION_MARKER) -> str:
    sections = "\n\n".join(
        f"{heading}\n{overview if index == 0 else MISSING_INFORMATION_MARKER}"
        for index, heading in enumerate(REQUIRED_WIKI_HEADINGS)
    )
    return f"# {title}\n\n{sections}"


def sample_pdf() -> bytes:
    title = "REMOTE CONSULTATION AND CONSULTATION APPLICATION ON iOS"
    return make_pdf(
        f"{title}\n2563",
        f"{title}\nJutharat Tuayjan",
        "Student ID 61050154\nAdvisor: Dr Smith",
        "Abstract\nProject uses iOS and Swift.\nKeywords: iOS, Swift",
        None,
        "Chapter 8\nFULL_REPORT_ONLY_SENTINEL",
    )


def test_pdf_to_draft_wiki_uses_focused_source_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    title = "REMOTE CONSULTATION AND CONSULTATION APPLICATION ON iOS"
    prompts: list[str] = []

    def fake_ollama(self: llm_service.OllamaClient, prompt: str) -> str:
        prompts.append(prompt)
        return wiki_markdown("Invented title", overview=f"{MISSING_INFORMATION_MARKER}.")

    monkeypatch.setattr(llm_service.OllamaClient, "generate", fake_ollama)
    response = post_pdf(sample_pdf(), "../../project.pdf")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "draft"
    assert isinstance(data["document_id"], int)
    assert data["original_filename"] == "project.pdf"
    assert data["page_count"] == 6
    assert data["selected_pages"] == [1, 2, 3, 4]
    assert data["structure_valid"] is True
    assert data["generated_markdown"].startswith(f"# {title}\n")
    assert f"{MISSING_INFORMATION_MARKER}." not in data["generated_markdown"]
    assert "- นักศึกษา: Jutharat Tuayjan" in data["generated_markdown"]
    assert data["title"] == title
    assert data["english_title"] == title
    assert data["students"] == ["Jutharat Tuayjan"]
    assert data["student_ids"] == ["61050154"]
    assert data["advisor"] == "Dr Smith"
    assert data["academic_year"] == "2563"
    assert data["keywords"] == ["iOS", "Swift"]
    assert "- อาจารย์ที่ปรึกษา: Dr Smith" in data["generated_markdown"]
    assert "คำสำคัญ: iOS, Swift" in data["generated_markdown"]
    assert data["warnings"][0]["page_number"] == 5
    assert "FULL_REPORT_ONLY_SENTINEL" not in prompts[0]
    assert "Source page 4 (English abstract)" in prompts[0]


def test_persisted_ocr_abstract_reaches_source_passed_to_generation(
    monkeypatch: pytest.MonkeyPatch,
    api_database: Engine,
) -> None:
    corrupted_native = "CORRUPTED_NATIVE_ABSTRACT_MUST_NOT_REACH_LLM"
    thai_ocr = "## บทคัดย่อ\nเนื้อหา OCR ภาษาไทยที่ต้องส่งต่อโดยคงถ้อยคำจากต้นฉบับ\n\n## คำสำคัญ\nโอซีอาร์, เอกสาร"
    english_ocr = "## Abstract\nFaithfully recovered English abstract text."
    extraction = PdfExtractionResult(
        page_count=5,
        full_text="\n\n".join(
            (
                "OCR ABSTRACT PROJECT",
                "OCR ABSTRACT PROJECT",
                "Student ID 63050001\nAcademic Year 2566",
                thai_ocr,
                english_ocr,
            )
        ),
        pages=(
            PdfPageText(1, "OCR ABSTRACT PROJECT"),
            PdfPageText(2, "OCR ABSTRACT PROJECT"),
            PdfPageText(3, "Student ID 63050001\nAcademic Year 2566"),
            PdfPageText(
                4,
                thai_ocr,
                native_text=corrupted_native,
                ocr_text=thai_ocr,
                provenance=TextProvenance.OCR,
            ),
            PdfPageText(
                5,
                english_ocr,
                native_text="corrupted English abstract",
                ocr_text=english_ocr,
                provenance=TextProvenance.OCR,
            ),
        ),
        warnings=(),
    )
    with Session(api_database) as session:
        document = Document(
            original_filename="ocr-abstract.pdf",
            storage_key="extractions/ocr-abstract.json",
            raw_text=extraction.full_text,
            extraction_data=serialize_extraction(extraction),
            page_count=extraction.page_count,
            extraction_status=ExtractionStatus.COMPLETED,
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        assert document.id is not None
        document_id = document.id

    received_sources: list[str] = []

    def generate(source_text: str) -> str:
        received_sources.append(source_text)
        return wiki_markdown("OCR ABSTRACT PROJECT")

    monkeypatch.setattr(wiki, "generate_wiki", generate)

    response = post_document(document_id)

    assert response.status_code == 200, response.text
    assert len(received_sources) == 1
    assert thai_ocr in received_sources[0]
    assert english_ocr in received_sources[0]
    assert corrupted_native not in received_sources[0]
    assert response.json()["keywords"] == ["โอซีอาร์", "เอกสาร"]


def test_timing_logs_cover_pipeline_without_changing_response(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        llm_service.OllamaClient,
        "generate",
        lambda self, prompt: wiki_markdown("Generated title"),
    )
    upload_response = post_upload(sample_pdf(), "timed-project.pdf")
    document_id = upload_response.json()["document_id"]

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = post_document(document_id)

    assert response.status_code == 200
    data = response.json()
    assert set(data) == set(wiki.DraftWikiResponse.model_fields)
    assert not any(key.endswith("_seconds") or "timing" in key for key in data)
    assert "wiki.generate" not in response.text

    messages = [record.getMessage() for record in caplog.records]
    for stage in (
        "source_selection",
        "evidence_extraction",
        "llm_generation",
        "output_finalization",
        "validation",
        "total",
    ):
        assert any(
            re.fullmatch(
                rf"wiki\.generate document_id={document_id} {stage}=\d+\.\d{{3}}s",
                message,
            )
            for message in messages
        )
    assert not any(" upload=" in message or " extraction=" in message for message in messages)


def test_failed_generation_keeps_api_error_and_logs_total(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(source: str) -> str:
        raise llm_service.OllamaTimeoutError("slow")

    monkeypatch.setattr(wiki, "generate_wiki", fail)
    document_id = post_upload(sample_pdf(), "timeout.pdf").json()["document_id"]
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = post_document(document_id)

    assert response.status_code == 504
    assert response.json() == {"detail": "Wiki generation timed out"}
    assert any(
        re.fullmatch(rf"wiki\.generate document_id={document_id} total=\d+\.\d{{3}}s", message)
        for message in caplog.messages
    )


def test_route_is_documented_in_openapi_and_docs() -> None:
    client = TestClient(app)
    operation = client.get("/openapi.json").json()["paths"]["/api/wiki/generate"]["post"]
    assert operation["requestBody"]["content"]["application/json"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["$ref"].endswith("/DraftWikiRequest")
    assert operation["responses"]["200"]
    assert client.get("/docs").status_code == 200


def test_api_uses_shared_finalizer_for_document_071_heading_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        "# Invented title\n\n"
        "## ภาพรวมโครงงาน\nข้อมูลในบทคัดย่อ\n\n"
        "## วัตถุประสงค์\nวัตถุประสงค์ตามต้นฉบับ\n\n"
        "## วิธีการ\nวิธีจากเอกสาร\n\n"
        "## คำสำคัญ\nระบบผู้เชี่ยวชาญ, การวินิจฉัยโรค\n\n"
        "## ผลการศึกษา\nผลตามต้นฉบับ 12.5%\n\n"
        "## สรุป\nสรุปตามต้นฉบับ"
    )
    monkeypatch.setattr(llm_service.OllamaClient, "generate", lambda self, prompt: raw)

    response = post_pdf(sample_pdf())

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["structure_valid"] is True
    markdown = data["generated_markdown"]
    assert "## ผลการศึกษา" not in markdown
    assert "## วิธีการ" not in markdown
    assert "## คำสำคัญ" not in markdown
    assert "คำสำคัญ: iOS, Swift" in markdown
    assert "ระบบผู้เชี่ยวชาญ, การวินิจฉัยโรค" not in markdown
    assert "## วิธีดำเนินงาน\n\nวิธีจากเอกสาร" in markdown
    assert "## ผลลัพธ์\n\nผลตามต้นฉบับ 12.5%" in markdown
    for heading in (
        "## ปัญหาและที่มา",
        "## เครื่องมือและเทคโนโลยี",
    ):
        assert f"{heading}\n\n{MISSING_INFORMATION_MARKER}" in markdown
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == list(
        REQUIRED_WIKI_HEADINGS
    )


def test_generate_requires_document_id_json() -> None:
    assert TestClient(app).post("/api/wiki/generate").status_code == 422
    response = TestClient(app).post(
        "/api/wiki/generate",
        files={"file": ("project.pdf", b"not-a-second-upload", "application/pdf")},
    )
    assert response.status_code == 422


def test_generate_reuses_persisted_document_without_second_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_calls = 0
    real_extract = upload.extract_pdf

    def counted_extract(source: BinaryIO) -> PdfExtractionResult:
        nonlocal extraction_calls
        extraction_calls += 1
        return real_extract(source)

    monkeypatch.setattr(upload, "extract_pdf", counted_extract)
    monkeypatch.setattr(
        wiki,
        "generate_wiki",
        lambda source: wiki_markdown("REMOTE CONSULTATION AND CONSULTATION APPLICATION ON iOS"),
    )
    upload_response = post_upload(sample_pdf())
    assert upload_response.status_code == 200

    response = post_document(upload_response.json()["document_id"])

    assert response.status_code == 200
    assert extraction_calls == 1


def test_unknown_document_id_returns_404() -> None:
    response = post_document(999_999)

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_failed_extraction_state_returns_409(api_database: Engine) -> None:
    with Session(api_database) as session:
        document = Document(
            original_filename="failed.pdf",
            storage_key="extractions/failed.json",
            extraction_status=ExtractionStatus.FAILED,
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        assert document.id is not None
        document_id = document.id

    response = post_document(document_id)

    assert response.status_code == 409
    assert response.json() == {"detail": "Document extraction is not complete"}


def test_blank_pdf_has_no_focused_source_and_never_calls_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wiki, "generate_wiki", lambda source: pytest.fail("Ollama must not be called")
    )
    response = post_pdf(make_pdf(None))
    assert response.status_code == 422
    assert "front-matter" in response.json()["detail"]


def test_metadata_is_empty_when_focused_source_has_no_explicit_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wiki, "generate_wiki", lambda source: wiki_markdown(MISSING_INFORMATION_MARKER)
    )
    response = post_pdf(make_pdf("General prose without explicit title or metadata"))
    assert response.status_code == 200
    data = response.json()
    assert data["title"] is None
    assert data["english_title"] is None
    assert data["students"] == []
    assert data["student_ids"] == []
    assert data["advisor"] is None
    assert data["academic_year"] is None
    assert data["keywords"] == []


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (llm_service.OllamaUnavailableError("offline"), 503, "Ollama is unavailable"),
        (
            llm_service.OllamaModelNotFoundError("missing"),
            503,
            "Configured Ollama model is unavailable",
        ),
        (llm_service.OllamaTimeoutError("slow"), 504, "Wiki generation timed out"),
        (
            llm_service.InvalidWikiMarkdownError(()),
            502,
            "Generated Wiki is invalid",
        ),
        (
            llm_service.InvalidWikiOutputError("unsafe marker", "raw"),
            502,
            "Generated Wiki is invalid",
        ),
        (llm_service.EmptyModelResponseError("empty"), 502, "Generated Wiki is invalid"),
    ],
)
def test_ollama_and_invalid_output_errors_are_mapped(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int, detail: str
) -> None:
    def fail(source: str) -> str:
        raise error

    monkeypatch.setattr(wiki, "generate_wiki", fail)
    response = post_pdf(sample_pdf())
    assert response.status_code == status
    assert response.json()["detail"] == detail


def test_invalid_structure_from_fake_generator_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wiki, "generate_wiki", lambda source: "# Wrong\nNo required sections")
    response = post_pdf(sample_pdf())
    assert response.status_code == 502
    assert response.json()["detail"] == "Generated Wiki has invalid structure"


def test_thai_text_and_keywords_survive_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    thai_title = "การพัฒนาระบบค้นคืนข้อมูลภาษาไทย"
    thai_abstract = "บทคัดย่อ\nระบบรองรับการค้นคืนข้อมูลภาษาไทย"
    extraction = PdfExtractionResult(
        page_count=4,
        full_text=f"{thai_title}\n{thai_abstract}",
        pages=(
            PdfPageText(1, thai_title),
            PdfPageText(2, ""),
            PdfPageText(3, ""),
            PdfPageText(4, f"{thai_abstract}\nคำสำคัญ: ภาษาไทย, ข้อมูล"),
        ),
        warnings=(),
    )
    received: list[str] = []
    monkeypatch.setattr(upload, "extract_pdf", lambda source: extraction)

    def fake_generate(source: str) -> str:
        received.append(source)
        return wiki_markdown(thai_title, overview="ระบบรองรับการค้นคืนข้อมูลภาษาไทย")

    monkeypatch.setattr(wiki, "generate_wiki", fake_generate)
    response = post_pdf(make_pdf("Placeholder PDF"))
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == thai_title
    assert data["english_title"] is None
    assert data["keywords"] == ["ภาษาไทย", "ข้อมูล"]
    assert data["selected_pages"] == [1, 4]
    assert thai_abstract in received[0]
    assert "ระบบรองรับการค้นคืนข้อมูลภาษาไทย" in data["generated_markdown"]


def test_document_071_api_metadata_and_markdown_share_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample" / "document_071.pdf"
    extraction = extract_pdf(fixture)
    keywords = [
        "กรดไขมันของแอลกอฮอล์เอทอกซีเลเต็ดฟอสเฟตเอสเทอร์",
        "กระจกฉากกั้นห้องอาบน้ำ",
        "ค่าประสิทธิภาพในการทำความสะอาด",
        "โซเดียมลอริลซัลเฟต",
        "น้ำยาขจัดคราบ",
        "สารลดแรงตึงผิว",
    ]
    monkeypatch.setattr(upload, "extract_pdf", lambda source: extraction)
    monkeypatch.setattr(
        llm_service.OllamaClient,
        "generate",
        lambda self, prompt: wiki_markdown("Invented title"),
    )

    response = post_pdf(make_pdf("Placeholder PDF"), "document_071.pdf")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["advisor"] == "รศ.ดร.อิทธิพล แจ้งชัด"
    assert data["keywords"] == keywords
    assert "- อาจารย์ที่ปรึกษา: รศ.ดร.อิทธิพล แจ้งชัด" in data["generated_markdown"]
    assert f"คำสำคัญ: {', '.join(keywords)}" in data["generated_markdown"]


def test_uploaded_temp_file_is_gone_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Path | None = None

    def inspect(source: BinaryIO) -> PdfExtractionResult:
        nonlocal captured
        captured = Path(source.name)
        assert captured.is_file()
        return extract_pdf(source)

    monkeypatch.setattr(upload, "extract_pdf", inspect)

    def fake_generate(source: str) -> str:
        assert captured is not None and not captured.exists()
        return wiki_markdown("REMOTE CONSULTATION AND CONSULTATION APPLICATION ON iOS")

    monkeypatch.setattr(wiki, "generate_wiki", fake_generate)
    assert post_pdf(sample_pdf()).status_code == 200
