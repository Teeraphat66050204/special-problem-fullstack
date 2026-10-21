"""Targeted chapter recovery with fake OCR; no external services."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models import Document, ExtractionStatus
from app.services import chapter_ocr
from app.services.chapter_ocr import rank_chapter_candidates, recover_chapter_one
from app.services.document_extraction import extraction_from_document, serialize_extraction
from app.services.ocr_service import OcrServiceError
from app.services.pdf_extractor import PdfExtractionResult, PdfPageText, TextProvenance, extract_pdf
from app.services.wiki_source import prepare_wiki_source

BODY = "โครงงานนี้พัฒนาระบบค้นคืนข้อมูลภาษาไทยเพื่อให้ผู้ใช้สามารถค้นหาเอกสารที่ต้องการ " * 4
CORRUPT = "ïììĊę 1\nïìîĞć\n1.1 ÙüćöđðŨîöć\n" + "ÖćøóĆçîćđĂÖÿćøĕì÷ " * 20
CHAPTER_ONE = "# บทที่ 1\n## บทนำ\n### 1.1 ความเป็นมา\n" + BODY
CHAPTER_TWO = "# บทที่ 2\n## ทฤษฎีที่เกี่ยวข้อง\n### 2.1 แนวคิด\n" + BODY


def extraction(texts=None):
    texts = texts or {
        5: "Abstract\nA project to retrieve Thai documents.",
        15: CORRUPT,
        16: "1.2 วัตถุประสงค์\n" + BODY,
        17: CHAPTER_TWO,
    }
    pages = tuple(PdfPageText(i, texts.get(i, "")) for i in range(1, max(texts) + 1))
    return PdfExtractionResult(len(pages), "\n\n".join(p.text for p in pages), pages, ())


class Provider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def extract_pages(self, source, page_numbers):
        self.calls.extend(page_numbers)
        assert source.tell() == 0
        source.read(1)
        return {number: self.responses.get(number, "") for number in page_numbers}


def recover(native, provider, maximum=6):
    source = BytesIO(b"pdf")
    source.seek(2)
    result = recover_chapter_one(
        source, native, Settings(_env_file=None, chapter_ocr_max_pages=maximum), provider=provider
    )
    assert source.tell() == 2
    return result


def test_damaged_heading_recovers_ocr_and_preserves_source_provenance():
    native = extraction()
    provider = Provider({15: CHAPTER_ONE})
    result = recover(native, provider)
    selection = prepare_wiki_source(result.extraction)
    assert provider.calls == [15]
    assert result.diagnostics.method == "ocr_fallback"
    assert result.diagnostics.selected_page == 15
    assert result.diagnostics.chapter_two_page == 17
    assert result.diagnostics.retained_ocr_pages == (15,)
    assert selection.detection.chapter_method == "ocr"
    assert selection.detection.chapter_end == 16
    assert 15 in selection.selected_pages
    assert "บทที่ 2" not in selection.source_text
    page = result.extraction.pages[14]
    assert page.provenance is TextProvenance.OCR
    assert page.native_text == CORRUPT
    assert page.ocr_text == CHAPTER_ONE.strip()


def test_native_success_never_loads_provider_or_invokes_ocr(monkeypatch):
    native = extraction({5: "Abstract\nA useful abstract about documents.", 15: CHAPTER_ONE})
    monkeypatch.setattr(
        chapter_ocr, "load_ocr_provider", lambda settings: pytest.fail("Loaded OCR")
    )
    result = recover_chapter_one(BytesIO(b"pdf"), native, Settings(_env_file=None))
    assert result.extraction is native
    assert result.diagnostics.method == "native"
    assert result.diagnostics.ocr_pages == ()
    assert result.diagnostics.candidates == ()


def test_ocr_toc_is_rejected_and_next_candidate_can_recover():
    native = extraction(
        {
            5: "Abstract\nA useful abstract about documents.",
            12: CORRUPT,
            15: CORRUPT,
            17: CHAPTER_TWO,
        }
    )
    provider = Provider({12: "สารบัญ\nบทที่ 1 บทนำ .... 1\n1.1 ความเป็นมา .... 1", 15: CHAPTER_ONE})
    result = recover(native, provider)
    assert result.diagnostics.selected_page == 15
    assert result.extraction.pages[11] == native.pages[11]
    assert any("page 12" in reason and "TOC" in reason for reason in result.diagnostics.rejections)


@pytest.mark.parametrize("budget", [0, 1, 2, 4, 6])
def test_total_budget_respected_even_when_no_candidates_succeed(budget):
    native = extraction({i: CORRUPT for i in range(10, 31)})
    provider = Provider({})
    result = recover(native, provider, budget)
    assert len(provider.calls) <= budget
    assert len(set(provider.calls)) == len(provider.calls)
    assert result.extraction is native
    assert len(result.diagnostics.ocr_pages) <= budget


def test_boundary_and_continuation_share_total_budget():
    native = extraction(
        {
            5: "Abstract\nA useful abstract about documents.",
            15: CORRUPT,
            16: "1.2 " + CORRUPT,
            17: "ïììĊę 2\n2.1 ทฤษฎี\n" + CORRUPT,
        }
    )
    provider = Provider({15: CHAPTER_ONE, 16: "1.2 วัตถุประสงค์\n" + BODY, 17: CHAPTER_TWO})
    result = recover(native, provider, 3)
    assert provider.calls == [15, 17, 16]
    assert result.diagnostics.chapter_two_page == 17
    assert result.diagnostics.retained_ocr_pages == (15, 16, 17)
    assert prepare_wiki_source(result.extraction).detection.chapter_end == 16


def test_ocr_failure_is_nonfatal_and_secret_is_not_exposed():
    class FailingProvider:
        def extract_pages(self, *args):
            raise RuntimeError("SECRET_TOKEN")

    native = extraction()
    result = recover(native, FailingProvider())
    assert result.extraction.pages == native.pages
    assert result.diagnostics.status == "ocr_failed"
    assert "SECRET_TOKEN" not in repr(result)
    assert "Abstract" in prepare_wiki_source(result.extraction).source_text
    assert result.extraction.warnings[-1].code == "ocr_failed"


def test_failure_after_recovered_start_keeps_recovery():
    class FailBoundary(Provider):
        def extract_pages(self, source, page_numbers):
            if 17 in page_numbers:
                raise OcrServiceError("unavailable")
            return super().extract_pages(source, page_numbers)

    native = extraction(
        {
            5: "Abstract\nA useful abstract about documents.",
            15: CORRUPT,
            17: "ïììĊę 2\n2.1 ทฤษฎี\n" + CORRUPT,
        }
    )
    result = recover(native, FailBoundary({15: CHAPTER_ONE}))
    assert result.diagnostics.selected_page == 15
    assert result.diagnostics.chapter_two_page is None
    assert prepare_wiki_source(result.extraction).detection.chapter_start == 15


def test_ocr_provenance_survives_existing_document_serialization():
    result = recover(extraction(), Provider({15: CHAPTER_ONE})).extraction
    document = Document(
        original_filename="test.pdf",
        storage_key="test.json",
        raw_text=result.full_text,
        extraction_data=serialize_extraction(result),
        page_count=result.page_count,
        extraction_status=ExtractionStatus.COMPLETED,
    )
    restored = extraction_from_document(document)
    assert restored.pages == result.pages
    assert prepare_wiki_source(restored).detection.chapter_method == "ocr"


@pytest.mark.parametrize("identifier, expected", [("064", 11), ("123", 20), ("254", 12)])
def test_real_damaged_fixture_candidate_ranking(identifier, expected):
    path = Path(__file__).parent / "fixtures" / "sample" / f"document_{identifier}.pdf"
    candidates, _ = rank_chapter_candidates(extract_pdf(path))
    assert candidates[0].page_number == expected
    assert len(candidates[0].reasons) >= 3


def test_missing_abstract_can_still_recover_chapter():
    result = recover(extraction({15: CORRUPT, 17: CHAPTER_TWO}), Provider({15: CHAPTER_ONE}))
    assert prepare_wiki_source(result.extraction).detection.chapter_start == 15


def test_environment_budget_and_hard_cap(monkeypatch):
    monkeypatch.setenv("CHAPTER_OCR_MAX_PAGES", "4")
    assert Settings(_env_file=None).chapter_ocr_max_pages == 4
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chapter_ocr_max_pages=7)


def test_dry_run_does_not_load_provider(monkeypatch):
    monkeypatch.setattr(
        chapter_ocr, "load_ocr_provider", lambda settings: pytest.fail("Loaded OCR")
    )
    result = recover_chapter_one(
        BytesIO(b"pdf"), extraction(), Settings(_env_file=None), allow_ocr=False
    )
    assert result.diagnostics.status == "not_run"
    assert result.diagnostics.ocr_pages == ()


def test_existing_ocr_success_does_not_call_provider_again():
    native = extraction({15: CHAPTER_ONE, 17: CHAPTER_TWO})
    pages = tuple(
        replace(page, provenance=TextProvenance.OCR) if page.page_number == 15 else page
        for page in native.pages
    )
    result = recover(replace(native, pages=pages), Provider({}))
    assert result.diagnostics.method == "existing_ocr"
    assert result.diagnostics.ocr_pages == ()


def test_title_only_opening_retains_ocr_support_on_next_page_when_end_unknown():
    native = extraction({15: "ïììĊę 1\nïìîĞć", 16: "1.1 x"})
    provider = Provider({15: "บทที่ 1\nบทนำ", 16: "1.1 ความเป็นมา\n" + BODY})
    result = recover(native, provider, 2)
    assert result.diagnostics.retained_ocr_pages == (15, 16)
    assert prepare_wiki_source(result.extraction).detection.chapter_start == 15
