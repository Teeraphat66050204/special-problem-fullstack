"""Production document-wide detection, boundaries, budgeting, and provenance."""

from pathlib import Path

import pytest

from app.services.pdf_extractor import PdfExtractionResult, PdfPageText, TextProvenance
from app.services.wiki_source import WikiSourcePolicy, prepare_wiki_source

BODY = "โครงงานนี้ศึกษาปัญหาการค้นหาเอกสารและพัฒนาระบบสำหรับค้นคืนข้อมูลภาษาไทย " * 5


def extraction(pages: dict[int, str]) -> PdfExtractionResult:
    return PdfExtractionResult(
        max(pages),
        "DO NOT SEND FULL TEXT",
        tuple(PdfPageText(number, text) for number, text in pages.items()),
        (),
    )


@pytest.mark.parametrize("start", [5, 15, 20])
@pytest.mark.parametrize(
    "heading",
    [
        "บทที่ 1",
        "บท ที่ 1",
        "บทที่ ๑",
        "บท 1",
        "Chapter 1",
        "CHAPTER 1",
        "Chapter One",
    ],
)
def test_real_chapter_anywhere_after_toc(start, heading):
    result = prepare_wiki_source(
        extraction(
            {
                2: "สารบัญ\nบทคัดย่อ .... ก\nบทที่ 1 บทนำ ........ 1\nบทที่ 2 ทฤษฎี .... 5",
                start: f"{heading}\nบทนำ\n1.1 ความเป็นมา\n{BODY}",
                start + 1: f"1.2 วัตถุประสงค์\n{BODY}",
                start + 2: f"บทที่ 2\nทฤษฎี\n2.1 ความรู้พื้นฐาน\n{BODY}\nCHAPTER_TWO_SENTINEL",
            }
        )
    )
    assert result.detection.chapter_start == start
    assert result.detection.chapter_end == start + 1
    assert result.detection.chapter_two_start == start + 2
    assert {start, start + 1} <= set(result.selected_pages)
    assert 2 not in result.selected_pages
    assert "CHAPTER_TWO_SENTINEL" not in result.source_text
    assert "DO NOT SEND FULL TEXT" not in result.source_text


def test_toc_without_leaders_and_fake_chapter_two_are_rejected():
    result = prepare_wiki_source(
        extraction(
            {
                3: "บทที่ 1 บทนำ 1\n1.1 ความเป็นมา 1\n1.2 วัตถุประสงค์ 3\nบทที่ 2 ทฤษฎี 4",
                15: f"บทที่ 1\nบทนำ\n1.1 ความเป็นมา\n{BODY}",
                16: "บทที่ 2 ทฤษฎี ........ 4\n" + BODY,
                17: "1.2 วัตถุประสงค์\nLATE_OBJECTIVE\n" + BODY,
                18: "Chapter Two\nTheory\n2.1 Concepts\n" + BODY,
            }
        )
    )
    assert result.detection.chapter_start == 15
    assert result.detection.chapter_two_start == 18
    assert "LATE_OBJECTIVE" in result.source_text


@pytest.mark.parametrize("heading", ["บทคัดย่อ", "บท คัด ย่อ", "## **บทคัดย่อ**", "Abstract"])
def test_abstract_outside_front_matter_without_chapter(heading):
    result = prepare_wiki_source(
        extraction(
            {
                2: "สารบัญ\nบทคัดย่อ .... 12",
                12: heading + "\n" + BODY,
                13: "กิตติกรรมประกาศ\nDO_NOT_INCLUDE_ACKNOWLEDGEMENTS",
            }
        )
    )
    assert result.detection.abstract_pages == (12,)
    assert result.detection.chapter_start is None
    assert "DO_NOT_INCLUDE" not in result.source_text


def test_abstract_continues_and_stops_at_keywords():
    result = prepare_wiki_source(
        extraction(
            {
                12: "บทคัดย่อ\n" + BODY,
                13: BODY + "\nคำสำคัญ: เอกสาร,\nภาษาไทย\nDO_NOT_INCLUDE",
                14: "Abstract\nEnglish supplementary abstract evidence.",
                15: "Acknowledgements\nDO_NOT_INCLUDE",
            }
        )
    )
    assert result.detection.abstract_pages == (12, 13, 14)
    assert "DO_NOT_INCLUDE" not in result.source_text


@pytest.mark.parametrize("heading", ["บทคัดยอ", "บทคัดย'อ", "บ ท คั ด ย่ อ"])
def test_noisy_thai_abstract_heading_requires_body_and_rejects_toc(heading):
    result = prepare_wiki_source(
        extraction(
            {
                2: f"สารบัญ\n{heading} ........ ก\nบทที่ 1 บทนำ ........ 1",
                4: f"{heading}\nผลการทดลองภาษาไทยที่มีเนื้อหาสาระเพียงพอสำหรับบทคัดย่อจริง\n{BODY}",
            }
        )
    )

    assert result.detection.abstract_pages == (4,)
    assert result.summary_selected_pages == (4,)
    assert "ผลการทดลองภาษาไทย" in result.summary_source_text


def test_summary_source_prefers_thai_abstract_and_excludes_metadata_noise():
    result = prepare_wiki_source(
        extraction(
            {
                1: "ชื่อโครงงานที่ไม่ควรส่งเข้าโมเดล\nชื่อนักศึกษา สมชาย ใจดี",
                4: "บทคัดย่อ\nผลการทดลองภาษาไทยใช้สูตร A 10% และให้ผลสูงสุด",
                5: "Abstract\nEnglish duplicate of the experimental result.",
                10: "บทที่ 1\nบทนำ\n1.1 ความเป็นมา\n" + BODY,
                11: "1.2 วัตถุประสงค์\nพัฒนาระบบทดสอบ\n" + BODY,
                12: "บทที่ 2\n2.1 ทฤษฎี\n" + BODY,
            }
        )
    )

    assert "ผลการทดลองภาษาไทย" in result.summary_source_text
    assert "English duplicate" not in result.summary_source_text
    assert "ชื่อโครงงานที่ไม่ควรส่งเข้าโมเดล" not in result.summary_source_text
    assert "ชื่อนักศึกษา" not in result.summary_source_text
    assert "1.1 ความเป็นมา" in result.summary_source_text
    assert "1.2 วัตถุประสงค์" in result.summary_source_text
    assert result.selected_pages == tuple(sorted({page.page_number for page in result.pages}))


def test_summary_source_falls_back_to_english_abstract():
    result = prepare_wiki_source(
        extraction(
            {
                1: "Title page metadata",
                4: "Abstract\nActual English experimental result with value 42.",
            }
        )
    )

    assert "Actual English experimental result" in result.summary_source_text
    assert "Title page metadata" not in result.summary_source_text


def test_budgets_prefer_thai_and_late_relevant_sections():
    result = prepare_wiki_source(
        extraction(
            {
                12: "บทคัดย่อ\n" + BODY,
                13: "Abstract\n" + "English supplemental evidence. " * 400,
                20: "บทที่ 1\nบทนำ\n1.1 ความเป็นมา\n" + BODY * 100,
                21: "1.2 วัตถุประสงค์\nIMPORTANT_OBJECTIVE\n" + BODY,
                22: "1.3 ตารางเวลา\nIRRELEVANT_SCHEDULE\n" + BODY,
                23: "1.4 ขอบเขต\nIMPORTANT_SCOPE\n" + BODY,
                24: "บทที่ 2\n2.1 ทฤษฎี\n" + BODY,
            }
        ),
        WikiSourcePolicy(
            max_source_characters=5000, abstract_characters=500, section_characters=500
        ),
    )
    assert result.source_text.startswith("Source page 12 (Thai abstract)")
    assert len(result.source_text) <= 5000
    assert "IMPORTANT_OBJECTIVE" in result.source_text
    assert "IMPORTANT_SCOPE" in result.source_text
    assert "IRRELEVANT_SCHEDULE" not in result.source_text


def test_focused_chapter_excludes_recognized_inventory_and_admin_sections():
    result = prepare_wiki_source(
        extraction(
            {
                10: f"บทที่ 1\nบทนำ\n1.1. ความเป็นมา\nBACKGROUND\n{BODY}",
                11: f"1.2. วัตถุประสงค์\nOBJECTIVE\n{BODY}",
                12: (
                    f"1.4. ขอบเขต\nSCOPE DPU YOLOv4 Visual Search\n{BODY}\n"
                    "1.6. อุปกรณ์และเครื่องมือที่ใช้\n"
                    "INVENTORY Apple MacBook Dell PowerEdge Nvidia Tesla V100"
                ),
                13: "1.7. เงื่อนไข/ข้อจำกัด\nADMINISTRATIVE_PROJECT_SUPERVISOR_CONDITIONS",
                14: f"บทที่ 2\n2.1 ทฤษฎี\n{BODY}",
            }
        )
    )

    assert "BACKGROUND" in result.summary_source_text
    assert "OBJECTIVE" in result.summary_source_text
    assert "SCOPE DPU YOLOv4 Visual Search" in result.summary_source_text
    assert "INVENTORY" not in result.summary_source_text
    assert "ADMINISTRATIVE_PROJECT_SUPERVISOR_CONDITIONS" not in result.summary_source_text


def test_no_useful_source_fails_even_with_metadata():
    with pytest.raises(ValueError, match="No usable abstract or real Chapter 1"):
        prepare_wiki_source(extraction({1: "Project title: no abstract", 20: "Unrelated body"}))


def test_weak_subsections_retain_bounded_chapter_and_ocr_provenance():
    page = PdfPageText(
        20, "บทที่ 1\nบทนำ\n" + BODY * 20, native_text="CORRUPT", provenance=TextProvenance.OCR
    )
    result = prepare_wiki_source(
        PdfExtractionResult(20, "FULL", (page,), ()), WikiSourcePolicy(chapter_characters=700)
    )
    assert result.pages[0].provenance == TextProvenance.OCR
    assert len(result.pages[0].text) <= 700
    assert "CORRUPT" not in result.source_text


def test_chapter_two_mid_page_preserves_only_preceding_chapter_one():
    result = prepare_wiki_source(
        extraction(
            {
                20: "บทที่ 1\nบทนำ\n1.1 ความเป็นมา\n" + BODY,
                21: "CHAPTER_ONE_END\nบทที่ 2\n2.1 ทฤษฎี\n" + BODY + "\nEXCLUDED",
            }
        )
    )
    assert result.detection.chapter_end == 21
    assert "CHAPTER_ONE_END" in result.source_text
    assert "EXCLUDED" not in result.source_text


def test_separate_number_and_heading_lines_and_late_chapter_boundary():
    result = prepare_wiki_source(
        extraction(
            {
                3: "บทที่ 1 บทนำ 1\n" + BODY,
                20: "บทที่ 1\nบทนำ\n1.1\nความเป็นมา\n" + BODY,
                21: "1.2\nวัตถุประสงค์\n" + BODY,
                22: "\n".join(["Continued body text"] * 9) + "\nบทที่ 2\n2.1\n" + BODY,
            }
        )
    )
    assert result.detection.chapter_start == 20
    assert result.detection.chapter_two_start == 22
    assert "วัตถุประสงค์" in result.source_text


def test_budget_counts_labels_and_allows_more_than_six_contributing_pages():
    pages = {10: "บทที่ 1\nบทนำ\n1.1 ความเป็นมา\n" + BODY}
    pages.update({i: f"1.{i - 9} วัตถุประสงค์\n{BODY}" for i in range(11, 19)})
    pages[19] = "บทที่ 2\n2.1 ทฤษฎี\n" + BODY
    result = prepare_wiki_source(extraction(pages), WikiSourcePolicy(max_source_characters=5000))
    assert len(result.selected_pages) > 6
    assert len(result.source_text) <= 5000


def test_sparse_pages_do_not_treat_late_body_as_cover_metadata():
    result = prepare_wiki_source(
        extraction(
            {
                12: "บทคัดย่อ\n" + BODY + "\nคำสำคัญ: เอกสาร",
                20: "DO_NOT_INCLUDE_UNRELATED_BODY",
            }
        )
    )
    assert result.selected_pages == (12,)


def test_real_fixture_chapter_one_after_long_front_matter():
    from app.services.pdf_extractor import extract_pdf

    path = Path(__file__).parent / "fixtures" / "sample" / "document_251.pdf"
    result = prepare_wiki_source(extract_pdf(path))
    assert result.detection.chapter_start == 25
    assert result.detection.chapter_end == 27
    assert result.detection.chapter_two_start == 28
    assert {25, 26, 27} <= set(result.selected_pages)


def test_document_339_prefers_noisy_thai_abstract_and_focuses_chapter_one():
    from app.services.pdf_extractor import extract_pdf

    path = Path(__file__).parent / "fixtures" / "sample" / "document_339.pdf"
    result = prepare_wiki_source(extract_pdf(path))

    assert result.detection.abstract_pages == (4, 5)
    assert result.summary_pages[0].role == "Thai abstract"
    assert result.summary_selected_pages == (4, 14, 15, 16)
    assert len(result.summary_source_text) < 8969
    assert "การปฏิบัติสหกิจในครั้งนี้" in result.summary_source_text
    assert all(term in result.summary_source_text for term in ("DPU", "YOLOv4", "Visual Search"))
    assert "Apple MacBook" not in result.summary_source_text
    assert "Dell PowerEdge" not in result.summary_source_text
    assert "ผู้ดูแลโครงการ" not in result.summary_source_text
    assert "1.6." not in result.summary_source_text
    assert "1.7." not in result.summary_source_text
