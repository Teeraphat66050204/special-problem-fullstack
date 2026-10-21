import pytest

from app.prompts.project_summary import (
    build_project_summary_prompt,
    normalize_project_summary,
    validate_project_summary,
)

_DETAIL = (
    "โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยเพื่อแก้ปัญหาการค้นหาข้อมูลจากรายงานจำนวนมาก "
    "โดยมีวัตถุประสงค์ให้ผู้ใช้เข้าถึงเนื้อหาที่เกี่ยวข้องได้สะดวกขึ้น ทีมงานศึกษาการสกัดข้อความ "
    "การตรวจจับบทคัดย่อและบทที่ 1 และนำ Python กับฐานข้อมูลมาใช้สร้างต้นแบบ "
    "ระบบทดสอบกับเอกสารตัวอย่างและเก็บข้อมูลหน้าที่เป็นแหล่งอ้างอิงเพื่อให้ตรวจสอบผลได้ "
    "ผลที่รายงานแสดงว่าต้นแบบเลือกเนื้อหาสำคัญและคงคำศัพท์ทางเทคนิคจากต้นฉบับ "
    "จึงสามารถใช้ช่วยเตรียมข้อมูลโครงการสำหรับการค้นหาและการทบทวนรายงานได้"
)


def _summary_at_least(length: int) -> str:
    return " ".join(_DETAIL for _ in range((length // len(_DETAIL)) + 1))


def test_plain_text_summary_is_valid_without_json():
    summary = _summary_at_least(750)
    assert not summary.startswith("{")
    assert validate_project_summary(summary).is_valid


@pytest.mark.parametrize(
    ("summary", "issue"),
    [
        ("", "empty_summary"),
        ("  \n", "empty_summary"),
        ("ระบบค้นคืนเอกสารโครงงานภาษาไทย", "summary_too_short"),
        ("โครงงานนี้พัฒนาระบบค้นหาเอกสาร", "summary_too_short"),
        ("English project summary " * 30, "summary_not_thai"),
        (
            "จากข้อมูลที่คุณให้มา " + _summary_at_least(700),
            "assistant_or_meta_commentary",
        ),
        ("คำถามนี้ " + _summary_at_least(700), "assistant_or_meta_commentary"),
        ("ภาพที่แสดง " + _summary_at_least(700), "assistant_or_meta_commentary"),
        ("ไม่มีข้อมูลเพียงพอ " + _summary_at_least(700), "assistant_or_meta_commentary"),
        ("หากคุณต้องการ " + _summary_at_least(700), "assistant_or_meta_commentary"),
        ("กรุณาตรวจสอบ " + _summary_at_least(700), "assistant_or_meta_commentary"),
        ("SOURCE_DOCUMENT " + _summary_at_least(700), "assistant_or_meta_commentary"),
        (
            "- ปัญหาของโครงการ\n- วัตถุประสงค์\n" + _summary_at_least(700),
            "markdown_heavy_summary",
        ),
        ("```\n" + _summary_at_least(700) + "\n```", "markdown_heavy_summary"),
        (
            "ไม่พบข้อมูลในเอกสารต้นฉบับ " + _summary_at_least(700),
            "missing_information_marker",
        ),
    ],
)
def test_obvious_summary_failures_are_invalid(summary, issue):
    validation = validate_project_summary(summary)
    assert not validation.is_valid
    assert issue in validation.issues


def test_normal_preferred_length_is_valid():
    summary = _summary_at_least(750)
    assert 700 <= len(summary) <= 1200
    assert validate_project_summary(summary).is_valid


@pytest.mark.parametrize("target", [650, 1250])
def test_slightly_outside_preferred_length_is_still_valid(target):
    assert validate_project_summary(_summary_at_least(target)).is_valid


def test_prompt_interpolates_source_and_requests_plain_text_only():
    prompt = build_project_summary_prompt("FOCUSED SOURCE SENTINEL")
    assert "<SOURCE_DOCUMENT>\nFOCUSED SOURCE SENTINEL\n</SOURCE_DOCUMENT>" in prompt
    assert "..." not in prompt
    assert "Return only the Thai project summary as plain text" in prompt
    assert "Do not return JSON or metadata" in prompt
    assert "Experimental results, actual outcomes, formulas" in prompt
    assert "Do not use Markdown headings, bullet or numbered lists" in prompt
    assert "Do not invent advantages, safety claims, environmental" in prompt
    assert "Do not expand acronyms or chemical names" in prompt
    assert "Clearly distinguish" in prompt
    assert "actual results from objectives and expected benefits" in prompt
    assert '"generated_summary"' not in prompt


def test_normalization_only_trims_transport_whitespace():
    summary = _summary_at_least(700)
    assert normalize_project_summary(" \n" + summary + " \n") == summary
