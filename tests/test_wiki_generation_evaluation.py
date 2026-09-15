"""Batch and lexical evaluation tests that never contact Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.datasets.groundtruth import (
    DatasetEntry,
    DatasetIndex,
    GroundTruthRecord,
    load_dataset_index,
)
from app.evaluation.wiki_generation import aggregate_results, evaluate_markdown
from app.prompts import (
    MISSING_INFORMATION_MARKER,
    REQUIRED_WIKI_HEADINGS,
    WIKI_GENERATION_INSTRUCTIONS,
    build_wiki_generation_prompt,
)
from app.services.llm_service import (
    InvalidWikiOutputError,
    OllamaUnavailableError,
    WikiGenerationResult,
)
from app.services.pdf_extractor import PdfExtractionResult, PdfPageText
from app.services.wiki_evidence import collect_wiki_source_evidence
from scripts import evaluate_wiki_generation


def _record(document_id: str) -> GroundTruthRecord:
    return GroundTruthRecord(
        document_id=document_id,
        title_th="ระบบให้คำปรึกษาบน iOS",
        title_en="Consultation on iOS",
        students=["นางสาวมาลี ใจดี"],
        advisor="ดร.สมชาย",
        academic_year=2564,
        abstract_th="ผลการทดลองพบว่า EDTA 0.1% เพิ่มประสิทธิภาพ",
        keywords=["iOS", "EDTA"],
        raw_text="GROUNDTRUTH_ONLY_SENTINEL",
    )


def _markdown(*, extra_result: str = "") -> str:
    bodies = (
        "นักศึกษา นางสาวมาลี ใจดี ที่ปรึกษา ดร.สมชาย ปี 2564\nConsultation on iOS",
        MISSING_INFORMATION_MARKER,
        MISSING_INFORMATION_MARKER,
        "ใช้ iOS และ EDTA",
        "ทดลองโดยใช้ EDTA",
        f"พบว่า EDTA 0.1% เพิ่มประสิทธิภาพ {extra_result}",
        MISSING_INFORMATION_MARKER,
    )
    sections = "\n\n".join(
        f"{heading}\n{body}" for heading, body in zip(REQUIRED_WIKI_HEADINGS, bodies, strict=True)
    )
    return f"# ระบบให้คำปรึกษาบน iOS\n\n{sections}"


def _generation() -> WikiGenerationResult:
    markdown = _markdown()
    return WikiGenerationResult(markdown, markdown, ())


def _dataset(tmp_path: Path, *, missing_groundtruth: set[str] | None = None) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "sample").mkdir(parents=True)
    (data_dir / "groundtruth").mkdir()
    identifiers = ("document_001", "document_002", "document_003")
    index = DatasetIndex(
        documents=[
            DatasetEntry(
                document_id=identifier,
                pdf=f"sample/{identifier}.pdf",
                groundtruth=f"groundtruth/{identifier}.json",
            )
            for identifier in identifiers
        ]
    )
    (data_dir / "index.json").write_text(index.model_dump_json(), encoding="utf-8")
    for identifier in identifiers:
        (data_dir / "sample" / f"{identifier}.pdf").write_bytes(b"%PDF-1.7\n")
        if identifier not in (missing_groundtruth or set()):
            (data_dir / "groundtruth" / f"{identifier}.json").write_text(
                _record(identifier).model_dump_json(), encoding="utf-8"
            )
    return data_dir


def _extraction() -> PdfExtractionResult:
    pages = (
        PdfPageText(1, "ระบบให้คำปรึกษาบน iOS\nนางสาวมาลี ใจดี\nดร.สมชาย\n2564"),
        PdfPageText(4, "บทคัดย่อ\nใช้ iOS และ EDTA\nผลการทดลองพบว่า EDTA 0.1% เพิ่มประสิทธิภาพ"),
        PdfPageText(9, "บทที่ 1 ข้อมูลทั้งรายงานที่ต้องไม่ส่งให้ Ollama"),
    )
    return PdfExtractionResult(
        page_count=9, full_text="\n".join(page.text for page in pages), pages=pages, warnings=()
    )


def test_batch_pairs_indexed_pdfs_and_groundtruth_and_skips_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path, missing_groundtruth={"document_002"})
    original_gt = (data_dir / "groundtruth" / "document_001.json").read_bytes()
    received: list[str] = []
    monkeypatch.setattr(evaluate_wiki_generation, "extract_pdf", lambda pdf_file: _extraction())

    def generate(source_text: str) -> WikiGenerationResult:
        received.append(source_text)
        return _generation()

    monkeypatch.setattr(evaluate_wiki_generation, "generate_wiki_result", generate)

    report = evaluate_wiki_generation.evaluate_dataset(data_dir)

    assert [item["document_id"] for item in report["documents"]] == [
        "document_001",
        "document_002",
        "document_003",
    ]
    assert report["documents"][1]["failure_category"] == "missing_groundtruth"
    assert report["summary"]["generation_successes"] == 2
    assert report["summary"]["generation_failures"] == 1
    assert len(received) == 2
    assert all("ข้อมูลทั้งรายงาน" not in text for text in received)
    assert all("GROUNDTRUTH_ONLY_SENTINEL" not in text for text in received)
    assert all("บทคัดย่อ" in text for text in received)
    assert (data_dir / "groundtruth" / "document_001.json").read_bytes() == original_gt


def test_missing_groundtruth_does_not_call_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path, missing_groundtruth={"document_001"})

    def fail_if_called(source_text: str) -> WikiGenerationResult:
        raise AssertionError(f"Unexpected model call: {source_text}")

    monkeypatch.setattr(evaluate_wiki_generation, "generate_wiki_result", fail_if_called)
    entry = load_dataset_index(data_dir / "index.json").documents[0]
    result = evaluate_wiki_generation.evaluate_entry(data_dir, entry)
    assert result["failure_category"] == "missing_groundtruth"


def test_rejected_raw_model_output_remains_assessable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path)
    monkeypatch.setattr(evaluate_wiki_generation, "extract_pdf", lambda pdf_file: _extraction())
    raw = _markdown().replace(
        MISSING_INFORMATION_MARKER,
        f"{MISSING_INFORMATION_MARKER}\nextra unsupported prose",
        1,
    )

    def reject(source_text: str) -> WikiGenerationResult:
        raise InvalidWikiOutputError("Missing-information marker has extra section content", raw)

    monkeypatch.setattr(evaluate_wiki_generation, "generate_wiki_result", reject)
    entry = load_dataset_index(data_dir / "index.json").documents[0]
    result = evaluate_wiki_generation.evaluate_entry(data_dir, entry)

    assert result["failure_category"] == "invalid_wiki_output"
    assert result["markdown"] is None
    assert result["raw_model_markdown"] == raw
    assert result["raw_model_checks"]["structure"]["valid"]
    assert not result["raw_model_checks"]["missing_marker_usage"]["exact_format_valid"]


def test_prompt_tuning_requires_explicit_evidence_and_exact_missing_marker() -> None:
    instructions = WIKI_GENERATION_INSTRUCTIONS
    assert "only tools, materials, chemicals" in instructions
    assert "Do not infer" in instructions
    assert "is not a measured outcome" in instructions
    assert "no stronger than the reported findings" in instructions
    assert "percentages, chemical names, model names" in instructions
    assert "Avoid repeating the same" in instructions
    assert "entire section body" in instructions
    assert "non-empty body under every heading" in instructions
    assert "no repeated headings" in instructions
    assert "no extra punctuation" in instructions
    assert "Never invent, reorder, paraphrase, translate" in instructions
    assert "copy\n  that English title verbatim" in instructions
    assert "other scripts absent from the source" in instructions
    assert "A period, colon" in instructions
    assert "earlier\n  system failed" in instructions
    assert "student ID, advisor" in instructions
    assert "Do not summarize these values away" in instructions
    assert "FOCUSED_SOURCE_FACTS" in instructions
    assert MISSING_INFORMATION_MARKER in instructions


def test_aggregate_counts_structure_only_for_assessed_outputs() -> None:
    summary = aggregate_results(
        [
            {
                "document_id": "document_001",
                "status": "success",
                "structure_valid": True,
                "issue_categories": ["keywords_missing"],
                "notes": ["keyword missing"],
            },
            {
                "document_id": "document_002",
                "status": "failure",
                "structure_valid": False,
                "failure_category": "invalid_markdown",
                "notes": ["bad structure"],
            },
            {
                "document_id": "document_003",
                "status": "failure",
                "structure_valid": None,
                "failure_category": "ollama_unavailable",
                "notes": ["offline"],
            },
        ]
    )
    assert summary["total_documents"] == 3
    assert summary["generation_successes"] == 1
    assert summary["generation_failures"] == 2
    assert summary["structure_assessed"] == 2
    assert summary["structure_pass_rate"] == 0.5
    assert summary["common_failure_categories"] == {
        "keywords_missing": 1,
        "invalid_markdown": 1,
        "ollama_unavailable": 1,
    }


def test_ollama_failure_is_recorded_and_other_documents_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path)
    monkeypatch.setattr(evaluate_wiki_generation, "extract_pdf", lambda pdf_file: _extraction())
    calls = 0

    def generate(source_text: str) -> WikiGenerationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OllamaUnavailableError("offline")
        return _generation()

    monkeypatch.setattr(evaluate_wiki_generation, "generate_wiki_result", generate)
    report = evaluate_wiki_generation.evaluate_dataset(data_dir)
    assert [item["status"] for item in report["documents"]] == [
        "failure",
        "success",
        "success",
    ]
    assert report["documents"][0]["failure_category"] == "ollama_unavailable"
    assert calls == 3


def test_report_keeps_raw_model_marker_errors_and_finalized_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path)
    entry = load_dataset_index(data_dir / "index.json").documents[0]
    monkeypatch.setattr(evaluate_wiki_generation, "extract_pdf", lambda pdf_file: _extraction())
    finalized = _markdown()
    raw = finalized.replace(MISSING_INFORMATION_MARKER, MISSING_INFORMATION_MARKER + ".")
    monkeypatch.setattr(
        evaluate_wiki_generation,
        "generate_wiki_result",
        lambda source: WikiGenerationResult(finalized, raw, ("marker_punctuation_removed:3",)),
    )

    result = evaluate_wiki_generation.evaluate_entry(data_dir, entry)

    assert result["status"] == "success"
    assert result["markdown"] == finalized
    assert result["raw_model_markdown"] == raw
    assert result["refinement_changes"] == ["marker_punctuation_removed:3"]
    assert result["checks"]["missing_marker_usage"]["exact_format_valid"]
    assert not result["raw_model_checks"]["missing_marker_usage"]["exact_format_valid"]


def test_unexpected_extraction_error_does_not_stop_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _dataset(tmp_path)
    extraction_calls = 0

    def extract(pdf_file) -> PdfExtractionResult:
        nonlocal extraction_calls
        extraction_calls += 1
        if extraction_calls == 1:
            raise RuntimeError("unexpected extractor bug")
        return _extraction()

    monkeypatch.setattr(evaluate_wiki_generation, "extract_pdf", extract)
    monkeypatch.setattr(
        evaluate_wiki_generation, "generate_wiki_result", lambda source: _generation()
    )
    report = evaluate_wiki_generation.evaluate_dataset(data_dir)
    assert report["documents"][0]["failure_category"] == "unexpected_document_failure"
    assert [item["status"] for item in report["documents"][1:]] == ["success", "success"]


def test_thai_english_metadata_keywords_and_finding_anchors_are_preserved() -> None:
    source = (
        "Source page 1 (title page):\nระบบให้คำปรึกษาบน iOS\nConsultation on iOS\n"
        "นางสาวมาลี ใจดี\nดร.สมชาย\n2564\n\n"
        "Source page 4 (Thai abstract):\nใช้ iOS และ EDTA\n"
        "ผลการทดลองพบว่า EDTA 0.1% เพิ่มประสิทธิภาพ"
    )
    checks = evaluate_markdown(source, _markdown(), _record("document_001"))
    assert checks["structure"]["valid"]
    assert checks["title"]["status"] == "pass"
    assert checks["metadata"]["rate"] == 1.0
    assert checks["keywords"]["rate"] == 1.0
    assert checks["findings"]["rate"] == 1.0
    assert checks["hallucination_indicators"] == []


def test_possible_unsupported_claims_are_indicators_not_proof() -> None:
    source = "ระบบให้คำปรึกษาบน iOS\nผลการทดลองพบว่า EDTA 0.1% เพิ่มประสิทธิภาพ"
    checks = evaluate_markdown(
        source,
        _markdown(extra_result="HNSW และ NaOH 1% ไม่สามารถทำงานได้ ความแม่นยำ 97% และดีที่สุด"),
        _record("document_001"),
    )
    indicators = {(item["kind"], item["value"]) for item in checks["hallucination_indicators"]}
    assert ("technical_term", "HNSW") in indicators
    assert ("technical_term", "NaOH") in indicators
    assert ("number", "97%") in indicators
    assert ("strengthened_claim", "ดีที่สุด") in indicators
    assert ("strengthened_claim", "ไม่สามารถ") in indicators
    assert ("strengthened_claim", "ความแม่นยำ") in indicators
    assert "hallucination_indicator" in checks["issue_categories"]


def test_title_preservation_keeps_mixed_case_technical_terms() -> None:
    source = "ระบบให้คำปรึกษาบน iOS"
    markdown = _markdown().replace("# ระบบให้คำปรึกษาบน iOS", "# ระบบให้คำปรึกษาบน ios")
    checks = evaluate_markdown(source, markdown, _record("document_001"))
    assert checks["title"]["status"] == "fail"
    assert checks["title"]["grounding_status"] == "fail"


def test_english_title_is_copied_when_thai_title_text_is_damaged() -> None:
    source = (
        "Source page 1 (title page):\nการพัฒนาเว็บแอปพลิเคชันที่ให5คำปรึกษา\n\n"
        "WEB APPLICATION DEVELOPMENT OF\nSELF-ASSESSMENT OF LEARNING\n\n"
        "Source page 2 (title page):\n"
        "WEB APPLICATION DEVELOPMENT OF\nSELF-ASSESSMENT OF LEARNING\n\n"
        "CHAWIT SALA\nPERMPAT PINGKASAN\n\n"
        "ACADEMIC YEAR 2021\n\n"
        "Source page 3 (project metadata):\n"
        "รหัสนักศึกษา 61050033\nรหัสนักศึกษา 61050076\nปีการศึกษา 2564"
    )
    evidence = collect_wiki_source_evidence(source)
    assert evidence.title == "WEB APPLICATION DEVELOPMENT OF SELF-ASSESSMENT OF LEARNING"
    assert evidence.students == ("CHAWIT SALA", "PERMPAT PINGKASAN")
    assert evidence.student_ids == ("61050033", "61050076")
    assert evidence.academic_year == "2564"
    prompt = build_wiki_generation_prompt(source)
    assert "Project title: WEB APPLICATION DEVELOPMENT OF SELF-ASSESSMENT OF LEARNING" in prompt
    assert "Student ID: 61050033" in prompt
    assert "Academic year: 2564" in prompt
    assert f"<SOURCE_DOCUMENT>\n{source}\n</SOURCE_DOCUMENT>" in prompt


def test_missing_title_marker_is_allowed_without_source_match() -> None:
    source = "Source page 1 (title page):\nÖćøóĆçîćđüĘï"
    sections = "\n\n".join(
        f"{heading}\n{MISSING_INFORMATION_MARKER}" for heading in REQUIRED_WIKI_HEADINGS
    )
    checks = evaluate_markdown(
        source, f"# {MISSING_INFORMATION_MARKER}\n\n{sections}", _record("document_001")
    )
    assert checks["title"]["grounding_status"] == "missing_marker"
    assert checks["missing_marker_usage"]["exact_format_valid"]


@pytest.mark.parametrize("suffix", [".", ":", " เพราะไม่มีข้อมูล"])
def test_marker_requires_exact_line_without_punctuation_or_explanation(suffix: str) -> None:
    sections = "\n\n".join(
        f"{heading}\n{MISSING_INFORMATION_MARKER}{suffix}" for heading in REQUIRED_WIKI_HEADINGS
    )
    checks = evaluate_markdown(
        "Source page 1 (title page):\nระบบให้คำปรึกษาบน iOS",
        f"# ระบบให้คำปรึกษาบน iOS\n\n{sections}",
        _record("document_001"),
    )
    assert not checks["missing_marker_usage"]["exact_format_valid"]
    assert len(checks["missing_marker_usage"]["invalid_lines"]) == 7
    assert "invalid_marker_format" in checks["issue_categories"]


def test_source_only_ids_and_names_are_reported_when_omitted() -> None:
    source = (
        "Source page 1 (title page):\nWEB APPLICATION DEVELOPMENT OF\n"
        "SELF-ASSESSMENT OF LEARNING\n\n"
        "Source page 2 (title page):\nWEB APPLICATION DEVELOPMENT OF\n"
        "SELF-ASSESSMENT OF LEARNING\n\nCHAWIT SALA\n\n"
        "Source page 3 (project metadata):\nStudent ID 61050033\n2564"
    )
    sections = "\n\n".join(
        f"{heading}\n{MISSING_INFORMATION_MARKER}" for heading in REQUIRED_WIKI_HEADINGS
    )
    checks = evaluate_markdown(
        source,
        f"# WEB APPLICATION DEVELOPMENT OF SELF-ASSESSMENT OF LEARNING\n\n{sections}",
        _record("document_001"),
    )
    assert checks["title"]["grounding_status"] == "pass"
    assert checks["metadata"]["missing_by_field"]["student_id"] == ["61050033"]
    assert checks["metadata"]["missing_by_field"]["student"] == ["CHAWIT SALA"]
    assert checks["metadata"]["missing_by_field"]["academic_year"] == ["2564"]
    assert "Missing source-supported metadata" in " ".join(checks["notes"])


def test_empty_sections_foreign_script_and_marker_punctuation_are_flagged() -> None:
    source = "ระบบให้คำปรึกษาบน iOS"
    sections = "\n\n".join(REQUIRED_WIKI_HEADINGS)
    markdown = f"# 中文标题\n\n{sections}\n{MISSING_INFORMATION_MARKER}."
    checks = evaluate_markdown(source, markdown, _record("document_001"))
    assert "empty_section" in checks["issue_categories"]
    assert "title_not_grounded" in checks["issue_categories"]
    assert {item["kind"] for item in checks["hallucination_indicators"]} >= {"non_source_script"}
    assert REQUIRED_WIKI_HEADINGS[-1] in checks["missing_marker_usage"]["mixed_with_other_content"]


def test_result_frequency_range_is_one_finding_anchor() -> None:
    record = _record("document_001").model_copy(
        update={"abstract_th": "ผลการศึกษาพบว่า สั่งอาหาร 2-3 ครั้ง/สัปดาห์"}
    )
    source = "ระบบให้คำปรึกษาบน iOS\nผลการศึกษาพบว่า สั่งอาหาร 2-3 ครั้ง/สัปดาห์"
    checks = evaluate_markdown(source, _markdown(), record)
    assert "2-3" in checks["findings"]["available_in_source"]
    assert "2" not in checks["findings"]["available_in_source"]


def test_marker_usage_flags_unsupported_section_filler() -> None:
    source = "โครงการทดสอบเรื่องทั่วไป"
    sections = "\n\n".join(
        f"{heading}\n{MISSING_INFORMATION_MARKER}" for heading in REQUIRED_WIKI_HEADINGS
    )
    checks = evaluate_markdown(
        source, f"# โครงการทดสอบเรื่องทั่วไป\n\n{sections}", _record("document_001")
    )
    assert REQUIRED_WIKI_HEADINGS[3] in checks["missing_marker_usage"]["used_in_sections"]
    assert checks["missing_marker_usage"]["possible_omissions"] == []
