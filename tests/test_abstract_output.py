import pytest

from app.prompts import MISSING_INFORMATION_MARKER, WIKI_MARKDOWN_SKELETON
from app.prompts.abstract_generation import (
    build_abstract_generation_prompt,
    normalize_abstract,
    validate_abstract,
)

PARAGRAPH = (
    "โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยเพื่อช่วยให้ผู้ใช้เข้าถึงข้อมูล "
    "โดยใช้การค้นหาคำสำคัญและจัดแสดงเอกสารที่ตรงกับคำค้น"
)


@pytest.mark.parametrize(
    "text", [PARAGRAPH, "ระบบรองรับการค้นหาเอกสารภาษาไทย", PARAGRAPH + " (12.5%)"]
)
def test_valid_paragraph(text):
    assert validate_abstract(text).is_valid


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n ",
        WIKI_MARKDOWN_SKELETON,
        "# title",
        "## วัตถุประสงค์",
        "- รายการ",
        "• รายการ",
        "1. รายการ",
        "๑) รายการ",
        MISSING_INFORMATION_MARKER,
        "นี่คือบทคัดย่อ " + PARAGRAPH,
        "จากข้อมูลที่ให้มา " + PARAGRAPH,
        PARAGRAPH + "\n\n" + PARAGRAPH,
        "วัตถุประสงค์:\n" + PARAGRAPH,
        "**" + PARAGRAPH + "**",
        "บทคัดย่อ: " + PARAGRAPH,
        "English abstract only.",
        "ก" * 4001,
        "โครงงานนี้มีวัตถุประสงค์: ค้นหาเอกสาร ผลลัพธ์: แสดงเอกสาร",
        "*ข้อความภาษาไทย*",
        "•รายการภาษาไทย",
    ],
)
def test_invalid_output(text):
    assert not validate_abstract(text).is_valid


def test_minimal_normalization_preserves_facts_and_does_not_merge_paragraphs():
    assert normalize_abstract(" \nบทคัดย่อ: " + PARAGRAPH + " \n") == PARAGRAPH
    assert normalize_abstract(PARAGRAPH + "\n\n" + PARAGRAPH).count("\n\n") == 1


def test_prompt_only_contains_source_and_abstract_instructions():
    prompt = build_abstract_generation_prompt("SOURCE SENTINEL")
    assert "exactly one" in prompt
    assert "No headings, bullets, numbered lists, Markdown" in prompt
    assert "Omit unavailable information naturally" in prompt
    assert "<SOURCE_DOCUMENT>\nSOURCE SENTINEL\n</SOURCE_DOCUMENT>" in prompt
    assert WIKI_MARKDOWN_SKELETON not in prompt
    assert MISSING_INFORMATION_MARKER not in prompt
