"""Deterministic focused-source metadata evidence regression tests."""

from pathlib import Path

from app.services.pdf_extractor import extract_pdf
from app.services.wiki_evidence import collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source


def _source(*pages: str) -> str:
    return "\n\n".join(
        f"Source page {number} (project metadata):\n{text}"
        for number, text in enumerate(pages, start=1)
    )


def test_thai_advisor_is_extracted_and_missing_advisor_stays_null() -> None:
    evidence = collect_wiki_source_evidence(_source("อาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัด"))
    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"

    missing = collect_wiki_source_evidence(_source("ไม่มีข้อมูลผู้ให้คำปรึกษา"))
    assert missing.advisor is None


def test_multiline_thai_keywords_stop_at_section_boundary_and_deduplicate() -> None:
    evidence = collect_wiki_source_evidence(
        _source("คำสำคัญ: หนึ่ง, สอง,\nสาม, สี่, สอง,\nห้า, หก\n\nบทคัดย่อ\nเจ็ด")
    )
    assert evidence.keywords == ("หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก")


def test_keyword_list_can_continue_on_the_next_focused_page() -> None:
    evidence = collect_wiki_source_evidence(
        _source("คำสำคัญ: หนึ่ง, สอง,", "ข\nสาม, สี่\n\nTitle New section")
    )
    assert evidence.keywords == ("หนึ่ง", "สอง", "สาม", "สี่")


def test_repeated_clean_metadata_wins_without_spell_correction() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "โครงงานสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดดกระบัง\nอาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัต",
            "หัวข้อสหกิจศึกษา โครงงานสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง\n"
            "อาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัด",
            "หัวข้อสหกิจศึกษา โครงงานสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง\n"
            "อาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัด",
        )
    )
    assert evidence.title == "โครงงานสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง"
    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"


def test_document_071_has_advisor_and_all_six_source_keywords() -> None:
    pdf_path = Path(__file__).parent / "fixtures" / "sample" / "document_071.pdf"
    selection = prepare_wiki_source(extract_pdf(pdf_path))
    evidence = collect_wiki_source_evidence(selection.source_text)

    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"
    assert evidence.keywords == (
        "กรดไขมันของแอลกอฮอล์เอทอกซีเลเต็ดฟอสเฟตเอสเทอร์",
        "กระจกฉากกั้นห้องอาบน้ำ",
        "ค่าประสิทธิภาพในการทำความสะอาด",
        "โซเดียมลอริลซัลเฟต",
        "น้ำยาขจัดคราบ",
        "สารลดแรงตึงผิว",
    )
