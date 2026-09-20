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


def test_multiline_thai_title_continues_without_joining_student_metadata() -> None:
    title = "การระบุตำแหน่งและการนำทางโดยใช้เครื่องมือ LIDAR และการรับรู้ความลึกของวัตถุ"
    evidence = collect_wiki_source_evidence(
        _source(
            "การระบุตำแหน่งและการนำทางโดยใช้เครื่องมือ LIDAR และ\n"
            "การรับรู้ความลึกของวัตถุ\n\nนายธิษณบดินท์ ทรายเพชร์",
            "หัวข้อสหกิจศึกษา การระบุตำแหน่งและการนำทางโดยใช้เครื่องมือ LIDAR และ\n"
            "การรับรู้ความลึกของวัตถุ\n"
            "ชื่อนักศึกษา นาย ธิษณบดินท์ ทรายเพชร์ 63050148",
        )
    )

    assert evidence.title == title
    assert evidence.students == ("นาย ธิษณบดินท์ ทรายเพชร์",)
    assert evidence.student_ids == ("63050148",)


def test_multiline_mixed_case_english_title_beats_submission_boilerplate() -> None:
    title = "Mobile Robot Localization and Navigation using LIDAR and Depth Perception"
    evidence = collect_wiki_source_evidence(
        _source(
            "Mobile Robot Localization and Navigation using LIDAR\n"
            "and Depth Perception\n\nAlex Example\n\n"
            "A COOPERATIVE EDUCATION SUBMITTED IN PARTIAL\n"
            "FULFILLMENT OF THE REQUIREMENT FOR",
            "Title Mobile Robot Localization and Navigation using LIDAR\n"
            "and Depth Perception\n"
            "Students Alex Example",
        )
    )

    assert evidence.title_en == title
    assert evidence.students == ("Alex Example",)


def test_multiple_labeled_students_keep_associated_id_order() -> None:
    evidence = collect_wiki_source_evidence(
        _source("Students\nAlice Example 63050001\nBob Sample 63050002\nAdvisor: Dr Mentor")
    )

    assert evidence.students == ("Alice Example", "Bob Sample")
    assert evidence.student_ids == ("63050001", "63050002")


def test_attached_student_labels_are_removed_without_changing_names() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "Students\n"
            "นางสาวเพชรรุ้ง นาถสกุลมงคล รหัสนักศึกษา 61051004\n"
            "นางสาววนิดา สุวรรณเมฆ Student 61051007\n"
            "Advisor: Dr Mentor"
        )
    )

    assert evidence.students == (
        "นางสาวเพชรรุ้ง นาถสกุลมงคล",
        "นางสาววนิดา สุวรรณเมฆ",
    )
    assert evidence.student_ids == ("61051004", "61051007")


def test_four_line_uppercase_english_title_includes_short_final_line() -> None:
    title = (
        "ACTINOMYCETE FROM ORGANIC RICE FIELD SOILS IN PATHUM THANI AND NAKHON SAWAN "
        "WITH INHIBITORY EFFECT ON RICE DISEASE AND RICE GROWTH PROMOTING"
    )
    evidence = collect_wiki_source_evidence(
        _source(
            "ACTINOMYCETE FROM ORGANIC RICE FIELD SOILS IN\n"
            "PATHUM THANI AND NAKHON SAWAN WITH INHIBITORY\n"
            "EFFECT ON RICE DISEASE AND RICE GROWTH\n"
            "PROMOTING\n\nAlice Example"
        )
    )

    assert evidence.title_en == title


def test_trailing_thai_conjunction_fragment_is_not_selected_as_a_title() -> None:
    english_title = "Species identification of Crotalaria sp. and evaluation of phytochemical"
    evidence = collect_wiki_source_evidence(
        _source(
            f"ข้อความชื่อบรรทัดแรกที่อ่านไม่ได้ 1\nและการประเมินพฤกษเคมี\n\n{english_title}",
            english_title,
        )
    )

    assert evidence.title == english_title
    assert evidence.title_en == english_title


def test_degree_and_submission_boilerplate_are_not_titles() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "A SPECIAL PROJECT SUBMITTED IN PARTIAL\n"
            "FULFILLMENT OF THE REQUIREMENT FOR\n"
            "THE DEGREE OF BACHELOR OF SCIENCE\n"
            "DEPARTMENT OF COMPUTER SCIENCE"
        )
    )

    assert evidence.title is None
    assert evidence.title_en is None


def test_document_295_focused_source_metadata_regression() -> None:
    pdf_path = Path(__file__).parent / "fixtures" / "sample" / "document_295.pdf"
    selection = prepare_wiki_source(extract_pdf(pdf_path))
    evidence = collect_wiki_source_evidence(selection.source_text)

    assert evidence.title == ("การระบุตำแหน่งและการนำทางโดยใช้เครื่องมือ LIDAR และการรับรู้ความลึกของวัตถุ")
    assert evidence.title_en == (
        "Mobile robot localization and navigation using LIDAR and Depth Perception"
    )
    assert evidence.students == ("นาย ธิษณบดินท์ ทรายเพชร์",)
    assert evidence.student_ids == ("63050148",)
    assert evidence.advisor == "ดร.ประพจน์ ศรีนุวัตติวงศ์"
    assert evidence.academic_year == "2566"
    assert evidence.keywords == ("SLAM", "PYQT5", "PostgreSQL", "Flask", "requests", "yaw")


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
