"""Reusable ground-truth loading and focused Wiki-source selection tests."""

import json
from pathlib import Path

import pytest

from app.datasets.groundtruth import load_dataset_index, load_groundtruth
from app.services.pdf_extractor import PdfExtractionResult, PdfPageText
from app.services.wiki_source import WikiSourcePolicy, prepare_wiki_source

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def extraction_with_pages(*page_texts: str) -> PdfExtractionResult:
    return PdfExtractionResult(
        page_count=len(page_texts),
        full_text="FULL DOCUMENT BODY SHOULD NOT BE USED",
        pages=tuple(
            PdfPageText(page_number=index, text=text)
            for index, text in enumerate(page_texts, start=1)
        ),
        warnings=(),
    )


def test_dataset_schema_loads_all_records_and_index() -> None:
    index = load_dataset_index(DATA_ROOT / "index.json")
    schema = json.loads((DATA_ROOT / "groundtruth.schema.json").read_text(encoding="utf-8"))

    assert index.schema_version == 1
    assert len(index.documents) == 20
    assert "document_id" in schema["required"]
    assert {
        "document_id",
        "title_th",
        "title_en",
        "students",
        "advisor",
        "academic_year",
        "abstract_th",
        "abstract_en",
        "keywords",
    }.issubset(schema["properties"])
    for entry in index.documents:
        record = load_groundtruth(DATA_ROOT / entry.groundtruth)
        assert record.document_id == entry.document_id
        assert record.title_th
        assert record.abstract_th
        assert record.students
        assert record.advisor
        assert record.academic_year


def test_groundtruth_loader_preserves_thai_and_optional_fields(tmp_path) -> None:
    thai_title = "ระบบค้นคืน\u200bข้อมูลภาษาไทย"
    path = tmp_path / "document_999.json"
    path.write_text(
        json.dumps(
            {
                "document_id": "document_999",
                "title_th": thai_title,
                "abstract_th": "บทคัดย่อ\u200bภาษาไทย",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    record = load_groundtruth(path)

    assert record.title_th == thai_title
    assert record.abstract_th == "บทคัดย่อ\u200bภาษาไทย"
    assert record.title_en is None
    assert record.abstract_en is None
    assert record.advisor is None
    assert record.academic_year is None
    assert record.students == []
    assert record.keywords == []


def test_groundtruth_document_id_must_match_filename(tmp_path) -> None:
    path = tmp_path / "document_999.json"
    path.write_text('{"document_id": "document_998"}', encoding="utf-8")

    with pytest.raises(ValueError, match="does not match filename"):
        load_groundtruth(path)


def test_wiki_source_selects_front_matter_abstracts_and_keywords() -> None:
    thai_abstract = "บทคัดย่อ\nระบบค้นคืน\u200bข้อมูลภาษาไทย"
    extraction = extraction_with_pages(
        "ชื่อโครงการ: ระบบค้นคืนข้อมูล",
        "Project title: Thai search system",
        "ชื่อนักศึกษา: นักศึกษาตัวอย่าง\nปีการศึกษา: 2564",
        thai_abstract,
        "Abstract\nA Thai information retrieval system.",
        "กิตติกรรมประกาศ\nUnrelated acknowledgement text",
        "Keywords: search, Thai\nUnrelated chapter content",
        "บทที่ 1\nLong body chapter",
    )

    selection = prepare_wiki_source(extraction)

    assert selection.selected_pages == (1, 2, 3, 4, 5, 7)
    assert thai_abstract in selection.source_text
    assert "A Thai information retrieval system" in selection.source_text
    assert "Keywords: search, Thai" in selection.source_text
    assert "Unrelated chapter content" not in selection.source_text
    assert "Long body chapter" not in selection.source_text
    assert extraction.full_text not in selection.source_text


def test_english_abstract_suggests_previous_thai_page_when_heading_is_damaged() -> None:
    extraction = extraction_with_pages(
        "Thai project title",
        "English project title",
        "Project metadata",
        "เนื้อหา\u200bบทคัดย่อภาษาไทยที่หัวข้ออ่านไม่ชัด",
        "Abstract\nEnglish abstract text",
        "Body chapter that should be excluded",
    )

    selection = prepare_wiki_source(extraction)

    assert 4 in selection.selected_pages
    assert 5 in selection.selected_pages
    assert "เนื้อหา\u200bบทคัดย่อภาษาไทย" in selection.source_text
    assert "Body chapter" not in selection.source_text


def test_wiki_source_uses_bounded_fallback_without_full_document() -> None:
    extraction = extraction_with_pages(
        "Title page",
        "English title page",
        "Front matter",
        "Possible Thai abstract text",
        "Acknowledgements",
        "Chapter one body",
    )

    selection = prepare_wiki_source(extraction)

    assert selection.selected_pages == (1, 2, 4)
    assert "Possible Thai abstract text" in selection.source_text
    assert "Chapter one body" not in selection.source_text
    assert extraction.full_text not in selection.source_text


def test_wiki_source_policy_limits_pages() -> None:
    extraction = extraction_with_pages(
        "Title one",
        "Title two",
        "Metadata\nปีการศึกษา: 2564",
        "บทคัดย่อ\nThai abstract",
        "Abstract\nEnglish abstract",
    )

    selection = prepare_wiki_source(extraction, WikiSourcePolicy(max_source_pages=3))

    assert len(selection.selected_pages) == 3
    assert 4 in selection.selected_pages


def test_wiki_source_rejects_missing_front_matter_text() -> None:
    extraction = extraction_with_pages("", "", "")

    with pytest.raises(ValueError, match="No usable front-matter text"):
        prepare_wiki_source(extraction)
