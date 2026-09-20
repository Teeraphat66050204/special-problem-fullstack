"""Conditional front-matter OCR tests without an external OCR provider."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

import pytest

from app.config import Settings
from app.models import Document, ExtractionStatus
from app.services.document_extraction import extraction_from_document, serialize_extraction
from app.services.ocr_service import apply_ocr_fallback
from app.services.pdf_extractor import (
    PdfExtractionResult,
    PdfPageText,
    TextProvenance,
)
from app.services.text_quality import assess_front_matter_quality
from app.services.wiki_evidence import collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source

_CORRUPTED_THAI = (
    "Ã–Ä‡Ã¸Ã³Ä†Ã§Ã®Ä‡Ä‘Ä‚Ã–Ã¿Ä‡Ã¸Ä•Ã¬Ã· ÃŸÄŒÄ™Ä‚Ã®Ä†Ã–Ã½Ä‹Ã–Ã¾Ä‡ Ä‚Ä‡ÃÄ‡Ã¸Ã·Å¤Ã¬ÄŠÄ™Ã°Ã¸Ä‹Ã–Ã¾Ä‡"
)


def _extraction(*texts: str) -> PdfExtractionResult:
    return PdfExtractionResult(
        page_count=len(texts),
        full_text="\n\n".join(texts),
        pages=tuple(PdfPageText(index, text) for index, text in enumerate(texts, 1)),
        warnings=(),
    )


class FakeOcrProvider:
    def __init__(self, pages: dict[int, str]) -> None:
        self.pages = pages
        self.calls: list[tuple[int, ...]] = []

    def extract_pages(self, source: object, page_numbers: Sequence[int]) -> dict[int, str]:
        del source
        requested = tuple(page_numbers)
        self.calls.append(requested)
        return {number: self.pages[number] for number in requested if number in self.pages}


def _clean_thai_front_matter() -> str:
    return (
        "หัวข้อโครงงาน ระบบตรวจสอบคุณภาพเอกสารภาษาไทย\n"
        "ชื่อนักศึกษา นาย สมชาย ใจดี 63050001\n"
        "อาจารย์ที่ปรึกษา ดร.ตัวอย่าง ทดสอบ\n"
        "ปีการศึกษา 2566\n"
        "บทคัดย่อ\nระบบนี้ตรวจสอบข้อความจากเอกสาร\n"
        "คำสำคัญ: เอกสาร, ภาษาไทย"
    )


def test_clean_native_text_does_not_call_ocr() -> None:
    native = _extraction(_clean_thai_front_matter())
    provider = FakeOcrProvider({1: "must not be used"})

    result = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)

    assert result is native
    assert provider.calls == []


def test_corrupted_thai_text_calls_ocr_only_for_front_matter_pages() -> None:
    corrupted = "ÖćøóĆçîćđĂÖÿćøĕì÷ ßČęĂîĆÖýċÖþć ĂćÝćø÷ŤìĊęðøċÖþć"
    native = _extraction(corrupted, "Clean English title", "", "", "", "", corrupted)
    provider = FakeOcrProvider({1: _clean_thai_front_matter()})

    result = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)

    assert provider.calls
    assert all(1 <= number <= 6 for number in provider.calls[0])
    assert 7 not in provider.calls[0]
    assert result.pages[0].provenance is TextProvenance.OCR


def test_ocr_page_limit_cannot_exceed_front_matter_scope() -> None:
    native = _extraction("short title")
    provider = FakeOcrProvider({1: "extra value"})

    with pytest.raises(ValueError, match="between 1 and 6"):
        apply_ocr_fallback(
            BytesIO(b"pdf"),
            native,
            provider,
            front_matter_page_limit=7,
        )

    assert provider.calls == []


def test_ocr_improves_title_and_student_evidence_extraction() -> None:
    corrupted = "ÖćøóĆçîćđĂÖÿćøĕì÷ ßČęĂîĆÖýċÖþć ĂćÝćø÷ŤìĊęðøċÖþć"
    native = _extraction(corrupted)
    provider = FakeOcrProvider({1: _clean_thai_front_matter()})

    enriched = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)
    evidence = collect_wiki_source_evidence(prepare_wiki_source(enriched).source_text)

    assert evidence.title == "ระบบตรวจสอบคุณภาพเอกสารภาษาไทย"
    assert evidence.students == ("นาย สมชาย ใจดี",)
    assert evidence.student_ids == ("63050001",)
    assert evidence.advisor == "ดร.ตัวอย่าง ทดสอบ"


def test_corrupted_native_abstract_is_replaced_in_focused_source_by_better_ocr() -> None:
    corrupted = _CORRUPTED_THAI
    ocr_abstract = (
        "## บทคัดย่อ\nเนื้อหาบทคัดย่อภาษาไทยที่กู้คืนจากหน้าต้นฉบับโดยไม่สรุปข้อความ\n\n## คำสำคัญ\nโอซีอาร์, เอกสาร"
    )
    native = _extraction(
        _clean_thai_front_matter(),
        "DOCUMENT QUALITY SYSTEM",
        "ชื่อนักศึกษา นาย สมชาย ใจดี\nปีการศึกษา 2566",
        corrupted,
        "## Abstract\nClean English abstract text.",
    )
    provider = FakeOcrProvider({4: ocr_abstract})

    enriched = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)
    selection = prepare_wiki_source(enriched)

    assert 4 in provider.calls[0]
    assert corrupted not in selection.source_text
    assert ocr_abstract in selection.source_text
    selected_page = next(page for page in selection.pages if page.page_number == 4)
    assert selected_page.provenance is TextProvenance.OCR
    assert enriched.pages[3].native_text == corrupted
    assert enriched.pages[3].ocr_text == ocr_abstract


def test_clean_native_abstract_remains_selected_when_another_page_needs_ocr() -> None:
    corrupted_title = _CORRUPTED_THAI
    native_abstract = "บทคัดย่อ\nเนื้อหาบทคัดย่อภาษาไทยจาก text layer ที่สมบูรณ์"
    native = _extraction(
        corrupted_title,
        "DOCUMENT QUALITY SYSTEM",
        "ชื่อนักศึกษา นาย สมชาย ใจดี\nปีการศึกษา 2566",
        native_abstract,
        "Abstract\nClean English abstract text.",
    )
    provider = FakeOcrProvider({1: _clean_thai_front_matter(), 4: "worse OCR"})

    enriched = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)
    selection = prepare_wiki_source(enriched)

    assert 4 not in provider.calls[0]
    assert native_abstract in selection.source_text
    selected_page = next(page for page in selection.pages if page.page_number == 4)
    assert selected_page.provenance is TextProvenance.PYMUPDF
    assert enriched.pages[3].ocr_text is None


def test_worse_ocr_text_does_not_replace_native_text() -> None:
    corrupted_but_usable = "โครงการภาษาไทย ÖćøóĆçîć พร้อมรายละเอียดที่ยังอ่านได้"
    native = _extraction(corrupted_but_usable)
    provider = FakeOcrProvider({1: "� � �"})

    result = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)

    assert provider.calls == [(1,)]
    assert result.pages[0].text == corrupted_but_usable
    assert result.pages[0].native_text == corrupted_but_usable
    assert result.pages[0].ocr_text == "� � �"
    assert result.pages[0].provenance is TextProvenance.PYMUPDF


def test_complementary_degraded_text_can_be_retained_as_mixed_provenance() -> None:
    native = _extraction("short title")
    provider = FakeOcrProvider({1: "extra value"})

    result = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)

    assert result.pages[0].text == "short title\n\nextra value"
    assert result.pages[0].native_text == "short title"
    assert result.pages[0].ocr_text == "extra value"
    assert result.pages[0].provenance is TextProvenance.MIXED


def test_page_provenance_and_both_text_versions_survive_persistence() -> None:
    corrupted = "ÖćøóĆçîćđĂÖÿćøĕì÷ ßČęĂîĆÖýċÖþć ĂćÝćø÷ŤìĊęðøċÖþć"
    clean_english = "PROJECT DOCUMENT TITLE AND CLEAN ENGLISH METADATA"
    native = _extraction(corrupted, clean_english)
    provider = FakeOcrProvider({1: _clean_thai_front_matter()})
    enriched = apply_ocr_fallback(BytesIO(b"pdf"), native, provider)
    document = Document(
        original_filename="project.pdf",
        storage_key="extractions/test.json",
        raw_text=enriched.full_text,
        extraction_data=serialize_extraction(enriched),
        page_count=enriched.page_count,
        extraction_status=ExtractionStatus.COMPLETED,
    )

    restored = extraction_from_document(document)

    assert restored.pages == enriched.pages
    assert restored.pages[0].native_text == corrupted
    assert restored.pages[0].ocr_text == _clean_thai_front_matter()
    assert restored.pages[0].provenance is TextProvenance.OCR
    assert restored.pages[1].provenance is TextProvenance.PYMUPDF


def test_quality_reasons_and_ocr_settings_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = assess_front_matter_quality((PdfPageText(1, "ÖćøóĆçîćđĂÖÿćøĕì÷ ßČęĂîĆÖýċÖþć"),))
    assert quality.degraded is True
    assert quality.reasons
    assert quality.degraded_page_numbers == (1,)

    monkeypatch.setenv("OCR_PROVIDER", "vendor.typhoon:create_provider")
    monkeypatch.setenv("OCR_FRONT_MATTER_PAGE_LIMIT", "5")
    monkeypatch.setenv("OCR_ENDPOINT", "https://ocr.invalid")
    monkeypatch.setenv("OCR_API_KEY", "secret-value")
    settings = Settings(_env_file=None)
    assert settings.ocr_provider == "vendor.typhoon:create_provider"
    assert settings.ocr_front_matter_page_limit == 5
    assert settings.ocr_endpoint == "https://ocr.invalid"
    assert settings.ocr_api_key is not None
    assert settings.ocr_api_key.get_secret_value() == "secret-value"
