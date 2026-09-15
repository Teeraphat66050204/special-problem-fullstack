"""The command-line smoke test wires existing services without requiring Ollama."""

from pathlib import Path

import pytest

from app.prompts import REQUIRED_WIKI_HEADINGS, validate_wiki_markdown
from app.services.llm_service import (
    InvalidWikiMarkdownError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.pdf_extractor import InvalidPdfError, PdfExtractionResult, PdfPageText
from scripts import smoke_pdf_to_wiki


def _extraction() -> PdfExtractionResult:
    pages = (
        PdfPageText(1, "ชื่อโครงการ: ระบบค้นคืนข้อมูล\nอาจารย์ที่ปรึกษา: ดร.สมชาย"),
        PdfPageText(3, "บทคัดย่อ\nงานนี้พัฒนาระบบค้นคืนข้อมูล\nคำสำคัญ: ค้นคืนข้อมูล"),
        PdfPageText(9, "บทที่ 1\nเนื้อหาทั้งเล่มที่ไม่ควรส่งให้ Ollama"),
    )
    return PdfExtractionResult(
        page_count=9,
        full_text="\n".join(page.text for page in pages),
        pages=pages,
        warnings=(),
    )


def _valid_markdown() -> str:
    sections = "\n\n".join(f"{heading}\nข้อมูลจากเอกสาร" for heading in REQUIRED_WIKI_HEADINGS)
    return f"# ระบบค้นคืนข้อมูล\n\n{sections}"


@pytest.fixture
def pdf_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(smoke_pdf_to_wiki, "extract_pdf", lambda pdf_file: _extraction())
    return path


def test_smoke_passes_only_focused_source_to_generation(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received: list[str] = []

    def generate(source_text: str) -> str:
        received.append(source_text)
        return _valid_markdown()

    monkeypatch.setattr(smoke_pdf_to_wiki, "generate_wiki", generate)

    assert smoke_pdf_to_wiki.main([str(pdf_path)]) == 0

    output = capsys.readouterr().out
    assert len(received) == 1
    assert "ชื่อโครงการ" in received[0]
    assert "บทคัดย่อ" in received[0]
    assert "คำสำคัญ" in received[0]
    assert "เนื้อหาทั้งเล่ม" not in received[0]
    assert received[0] in output
    assert _valid_markdown() in output
    assert "Wiki Markdown structure: valid" in output


def test_missing_pdf_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_pdf_to_wiki.main([str(tmp_path / "missing.pdf")]) == 1
    assert "PDF not found" in capsys.readouterr().err


def test_extraction_failure_is_reported(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_extraction(pdf_file):
        del pdf_file
        raise InvalidPdfError("corrupt PDF")

    monkeypatch.setattr(smoke_pdf_to_wiki, "extract_pdf", fail_extraction)
    assert smoke_pdf_to_wiki.run(pdf_path) == 1
    assert "PDF extraction failed: corrupt PDF" in capsys.readouterr().err


def test_no_suitable_wiki_source_is_reported(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        smoke_pdf_to_wiki,
        "extract_pdf",
        lambda pdf_file: PdfExtractionResult(
            page_count=1,
            full_text="",
            pages=(PdfPageText(1, ""),),
            warnings=(),
        ),
    )
    assert smoke_pdf_to_wiki.run(pdf_path) == 1
    assert "No suitable Wiki source: No usable front-matter text" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OllamaUnavailableError("connection refused"), "Ollama unavailable"),
        (OllamaTimeoutError("timed out"), "Ollama timeout"),
    ],
)
def test_ollama_failures_are_reported(
    pdf_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    message: str,
) -> None:
    def fail_generation(source_text: str) -> str:
        del source_text
        raise error

    monkeypatch.setattr(smoke_pdf_to_wiki, "generate_wiki", fail_generation)
    assert smoke_pdf_to_wiki.run(pdf_path) == 1
    assert message in capsys.readouterr().err


def test_invalid_markdown_from_llm_service_is_reported(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = validate_wiki_markdown("# Title only")

    def fail_generation(source_text: str) -> str:
        del source_text
        raise InvalidWikiMarkdownError(invalid.issues)

    monkeypatch.setattr(smoke_pdf_to_wiki, "generate_wiki", fail_generation)
    assert smoke_pdf_to_wiki.run(pdf_path) == 1
    assert "Invalid Wiki Markdown returned by Ollama" in capsys.readouterr().err


def test_explicit_markdown_validation_rejects_invalid_output(
    pdf_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(smoke_pdf_to_wiki, "generate_wiki", lambda text: "# Title only")
    assert smoke_pdf_to_wiki.run(pdf_path) == 1
    output = capsys.readouterr()
    assert "Generated Wiki Markdown:\n# Title only" in output.out
    assert "Invalid Wiki Markdown:" in output.err
