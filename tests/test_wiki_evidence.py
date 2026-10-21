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


def _role_source(*pages: tuple[str, str]) -> str:
    return "\n\n".join(
        f"Source page {number} ({role}):\n{text}"
        for number, (role, text) in enumerate(pages, start=1)
    )


def test_thai_advisor_is_extracted_and_missing_advisor_stays_null() -> None:
    evidence = collect_wiki_source_evidence(_source("อาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัด"))
    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"

    missing = collect_wiki_source_evidence(_source("ไม่มีข้อมูลผู้ให้คำปรึกษา"))
    assert missing.advisor is None


def test_project_metadata_schema_is_built_only_from_deterministic_source_evidence() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "Title: Deterministic Document Retrieval\n"
            "Students: Alice Example\n"
            "Student ID: 63050001\n"
            "Academic Year: 2566\n"
            "Advisor: Dr Mentor\n"
            "Department: Computer Science\n"
            "Faculty: Science and Technology\n"
            "Institution: Example University\n"
            "Project Type: Special Project"
        )
    )

    assert evidence.project_metadata() == {
        "title_th": None,
        "title_en": "Deterministic Document Retrieval",
        "students": ["Alice Example"],
        "student_ids": ["63050001"],
        "academic_year": "2566",
        "advisor": "Dr Mentor",
        "department": "Computer Science",
        "faculty": "Science and Technology",
        "institution": "Example University",
        "project_type": "Special Project",
    }


def test_bounded_thai_metadata_fields_do_not_consume_following_labels() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "ภาควิชา เคมี\n"
            "คณะ วิทยาศาสตร์\n"
            "มหาวิทยาลัย สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง\n"
            "ปีการศึกษา 2564\n"
            "อาจารย์ที่ปรึกษา รศ.ดร.อิทธิพล แจ้งชัด"
        )
    )

    assert evidence.department == "เคมี"
    assert evidence.faculty == "วิทยาศาสตร์"
    assert evidence.institution == "สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง"
    assert evidence.academic_year == "2564"
    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"


def test_metadata_fields_stop_before_inline_labels_and_approval_boilerplate() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "คณะ วิทยาศาสตร์ มหาวิทยาลัย มหาวิทยาลัยตัวอย่าง ปีการศึกษา 2564\n"
            "คณะ วิทยาศาตร์ สถาบันเทคโนโลยีตัวอย่าง อนุมัติ\n"
            "มหาวิทยาลัย สถาบันเทคโนโลยีตัวอย่าง อาจารย์ที่ปรึกษา ดร. ตัวอย่าง"
        )
    )

    assert evidence.faculty == "วิทยาศาสตร์"
    assert "อนุมัติ" not in evidence.faculty
    assert evidence.institution == "สถาบันเทคโนโลยีตัวอย่าง"
    assert "อาจารย์ที่ปรึกษา" not in evidence.institution


def test_explicit_metadata_students_win_conflict_and_emit_nonfatal_warning() -> None:
    evidence = collect_wiki_source_evidence(
        _role_source(
            ("title page", "Students: Alice Example 63050001"),
            ("project metadata", "Students: Bob Sample 63050002"),
        )
    )

    assert evidence.students == ("Bob Sample",)
    assert evidence.student_ids == ("63050002",)
    assert evidence.warnings[0].code == "student_metadata_conflict"


def test_title_page_three_students_vs_explicit_two_emits_one_warning() -> None:
    evidence = collect_wiki_source_evidence(
        _role_source(
            (
                "title page",
                "การศึกษาน้ำยาขจัดคราบกระจกฉากกั้นห้องอาบน้ำ\n"
                "STUDY OF LIMESCALE REMOVER FOR SHOWER BOX\n"
                "ณัฐริกา เสาวคนธ์\n"
                "บัวชมพู ปาลกะวงศ์\n"
                "บุณยาพร ศรีอินทร์",
            ),
            (
                "project metadata",
                "ชื่อนักศึกษา นางสาวณัฐริกา เสาวคนธ์ รหัสนักศึกษา 61050365\n"
                "นางสาวบัวชมพู ปาลกะวงศ์ รหัสนักศึกษา 61050391",
            ),
        )
    )

    assert evidence.students == (
        "นางสาวณัฐริกา เสาวคนธ์",
        "นางสาวบัวชมพู ปาลกะวงศ์",
    )
    assert evidence.student_ids == ("61050365", "61050391")
    assert len(evidence.warnings) == 1
    assert evidence.warnings[0].code == "student_metadata_conflict"
    assert "title page lists 3 student(s)" in evidence.warnings[0].message
    assert "explicit metadata lists 2" in evidence.warnings[0].message
    assert "บุณยาพร ศรีอินทร์" in evidence.warnings[0].message


def test_matching_title_and_explicit_students_have_no_conflict_warning() -> None:
    evidence = collect_wiki_source_evidence(
        _role_source(
            (
                "title page",
                "การศึกษาน้ำยาขจัดคราบกระจกฉากกั้นห้องอาบน้ำ\nณัฐริกา เสาวคนธ์\nบัวชมพู ปาลกะวงศ์",
            ),
            (
                "project metadata",
                "ชื่อนักศึกษา นางสาวณัฐริกา เสาวคนธ์ รหัสนักศึกษา 61050365\n"
                "นางสาวบัวชมพู ปาลกะวงศ์ รหัสนักศึกษา 61050391",
            ),
        )
    )

    assert len(evidence.students) == 2
    assert evidence.warnings == ()


def test_title_page_only_extra_student_is_not_added_to_explicit_metadata() -> None:
    evidence = collect_wiki_source_evidence(
        _role_source(
            (
                "title page",
                "โครงงานระบบค้นคืนเอกสารภาษาไทยสำหรับสถาบันการศึกษา\nสมชาย ใจดี\nสมหญิง รักเรียน",
            ),
            (
                "project metadata",
                "ชื่อนักศึกษา นายสมชาย ใจดี รหัสนักศึกษา 63050001",
            ),
        )
    )

    assert evidence.students == ("นายสมชาย ใจดี",)
    assert evidence.student_ids == ("63050001",)
    assert len(evidence.warnings) == 1
    assert "สมหญิง รักเรียน" in evidence.warnings[0].message


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


def test_english_title_stops_before_all_caps_student_block() -> None:
    title = (
        "Effect of extracellular substance of lactic acid bacteria on growth and "
        "biofilm formation of Streptococcus mutans and Streptococcus sobrinus"
    )
    evidence = collect_wiki_source_evidence(
        _role_source(
            (
                "title page",
                "Effect of extracellular substance of lactic acid bacteria on growth and\n"
                "biofilm formation of Streptococcus mutans and Streptococcus sobrinus\n"
                "KHANITTHA JAMPRAJAK\n"
                "THITIMA SAYASEWEE\n"
                "PITCHAPORN PRAKOBSOOK\n"
                "A SPECIAL PROJECT SUBMITTED IN PARTIAL FULFILLMENT",
            ),
        )
    )

    assert evidence.title_en == title
    assert "KHANITTHA" not in evidence.title_en
    assert evidence.students == (
        "KHANITTHA JAMPRAJAK",
        "THITIMA SAYASEWEE",
        "PITCHAPORN PRAKOBSOOK",
    )


def test_bilingual_title_page_recovers_both_titles_without_students() -> None:
    title_th = "ผลของสารนอกเซลล์ต่อการเจริญและการสร้างไบโอฟิล์มของเชื้อแบคทีเรีย"
    title_en = "Effect of extracellular substance on bacterial growth and biofilm formation"
    evidence = collect_wiki_source_evidence(
        _role_source(
            (
                "title page",
                f"{title_th}\n{title_en}\nKHANITTHA JAMPRAJAK\nTHITIMA SAYASEWEE",
            ),
        )
    )

    assert evidence.title == title_th
    assert evidence.project_metadata()["title_th"] == title_th
    assert evidence.title_en == title_en
    assert "KHANITTHA" not in evidence.title_en


def test_explicit_metadata_title_is_preferred_and_title_page_is_fallback() -> None:
    title_page = "Title Page Project About Document Retrieval"
    explicit = "Explicit Metadata Project About Document Retrieval"
    evidence = collect_wiki_source_evidence(
        _role_source(
            ("title page", title_page),
            ("project metadata", f"Title: {explicit}\nStudents: Alice Example"),
        )
    )
    fallback = collect_wiki_source_evidence(_role_source(("title page", title_page)))

    assert evidence.title_en == explicit
    assert fallback.title_en == title_page


def test_short_explicit_english_title_stops_before_student_metadata() -> None:
    evidence = collect_wiki_source_evidence(
        _source(
            "Title VISUAL SEARCH\n"
            "Students Mister Naratcha Sathienpakiranakorn Student ID 62050181\n"
            "Mister Nirattisai Sathu Student ID 62050187\n"
            "Degree Bachelor of Science"
        )
    )

    assert evidence.title_en == "VISUAL SEARCH"
    assert "Mister" not in evidence.title_en
    assert "Student ID" not in evidence.title_en
    assert "Nirattisai" not in evidence.title_en
    assert evidence.student_ids == ("62050181", "62050187")


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


def test_document_064_english_title_excludes_student_names() -> None:
    pdf_path = Path(__file__).parent / "fixtures" / "sample" / "document_064.pdf"
    selection = prepare_wiki_source(extract_pdf(pdf_path))
    evidence = collect_wiki_source_evidence(selection.source_text)

    assert evidence.title_en == (
        "Effect of extracellular substance of lactic acid bacteria on growth and biofilm "
        "formation of Streptococcus mutans and Streptococcus sobrinus"
    )
    assert "KHANITTHA" not in evidence.title_en
    assert "THITIMA" not in evidence.title_en
    assert "PITCHAPORN" not in evidence.title_en


def test_document_339_metadata_regression() -> None:
    pdf_path = Path(__file__).parent / "fixtures" / "sample" / "document_339.pdf"
    selection = prepare_wiki_source(extract_pdf(pdf_path))
    evidence = collect_wiki_source_evidence(selection.source_text)
    metadata = evidence.project_metadata()

    assert metadata["title_th"] == "ระบบต้นแบบค้นหาสินค้าและระบุเส้นทางด้วยภาพ"
    assert metadata["title_en"] == "VISUAL SEARCH"
    assert "Mister" not in metadata["title_en"]
    assert "Student ID" not in metadata["title_en"]
    assert "Nirattisai" not in metadata["title_en"]
    assert metadata["students"] == [
        "นาย นรัชฌา เสถียรปกิรณกร",
        "นาย นิรัติศัย สาธุ",
    ]
    assert metadata["student_ids"] == ["62050181", "62050187"]
    assert metadata["academic_year"] == "2565"


def test_document_071_has_advisor_and_all_six_source_keywords() -> None:
    pdf_path = Path(__file__).parent / "fixtures" / "sample" / "document_071.pdf"
    selection = prepare_wiki_source(extract_pdf(pdf_path))
    evidence = collect_wiki_source_evidence(selection.source_text)

    assert evidence.advisor == "รศ.ดร.อิทธิพล แจ้งชัด"
    assert evidence.department == "เคมี"
    assert evidence.faculty == "วิทยาศาสตร์"
    assert evidence.institution == "สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง"
    assert len(evidence.warnings) == 1
    assert evidence.warnings[0].code == "student_metadata_conflict"
    assert "title page lists 3 student(s)" in evidence.warnings[0].message
    assert "explicit metadata lists 2" in evidence.warnings[0].message
    assert evidence.keywords[:6] == (
        "กรดไขมันของแอลกอฮอล์เอทอกซีเลเต็ดฟอสเฟตเอสเทอร์",
        "กระจกฉากกั้นห้องอาบน้ำ",
        "ค่าประสิทธิภาพในการทำความสะอาด",
        "โซเดียมลอริลซัลเฟต",
        "น้ำยาขจัดคราบ",
        "สารลดแรงตึงผิว",
    )
