"""Source-backed finalization tests that do not require Ollama."""

import pytest

from app.prompts import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS, validate_wiki_markdown
from app.services import llm_service
from app.services.wiki_output import WikiOutputError, refine_wiki_markdown


def _source() -> str:
    return (
        "Source page 1 (title page):\nการพัฒนาเว็บแอปพลิเคชันที่ให5คำปรึกษา\n\n"
        "WEB APPLICATION DEVELOPMENT OF\nSELF-ASSESSMENT OF LEARNING\n\n2564\n\n"
        "Source page 2 (title page):\nWEB APPLICATION DEVELOPMENT OF\n"
        "SELF-ASSESSMENT OF LEARNING\n\nCHAWIT SALA\nPERMPAT PINGKASAN\n\n"
        "Source page 3 (project metadata):\nStudent ID 61050033\n"
        "Student ID 61050076\nAdvisor: Dr Smith"
    )


def _markdown(title: str, body: str = MISSING_INFORMATION_MARKER) -> str:
    sections = "\n\n".join(f"{heading}\n{body}" for heading in REQUIRED_WIKI_HEADINGS)
    return f"# {title}\n\n{sections}"


def _document_071_like_raw() -> str:
    return (
        "# DOCUMENT 071 TITLE\n\n"
        "## ภาพรวมโครงงาน\nเนื้อหาบทคัดย่อจากเอกสาร\n\n"
        "## วัตถุประสงค์\nศึกษาผลของโครงการ\n\n"
        "## ผลการศึกษา\nผลที่ระบุในบทคัดย่อ: 12.5%\n\n"
        "## สรุป\nสรุปตามข้อมูลต้นฉบับ"
    )


def _section_body(markdown: str, heading: str) -> str:
    after = markdown.split(heading, 1)[1]
    return after.split("\n## ", 1)[0].strip()


def test_document_071_alias_and_missing_sections_are_normalized_without_content_loss() -> None:
    raw = _document_071_like_raw()
    assert not validate_wiki_markdown(raw).is_valid

    result = refine_wiki_markdown("Generic focused source", raw)

    assert validate_wiki_markdown(result.markdown).is_valid
    assert "## ผลการศึกษา" not in result.markdown
    assert _section_body(result.markdown, "## ผลลัพธ์") == "ผลที่ระบุในบทคัดย่อ: 12.5%"
    for heading in (
        "## ปัญหาและที่มา",
        "## เครื่องมือและเทคโนโลยี",
        "## วิธีดำเนินงาน",
    ):
        assert _section_body(result.markdown, heading) == MISSING_INFORMATION_MARKER
    assert "เนื้อหาบทคัดย่อจากเอกสาร" in result.markdown
    assert "ศึกษาผลของโครงการ" in result.markdown
    assert "สรุปตามข้อมูลต้นฉบับ" in result.markdown
    assert "heading_alias_normalized:1" in result.changes
    assert "missing_sections_inserted:3" in result.changes


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("## ผลการศึกษา", "## ผลลัพธ์"),
        ("## ผลการทดลอง", "## ผลลัพธ์"),
        ("## ที่มาและปัญหา", "## ปัญหาและที่มา"),
        ("## เทคโนโลยีและเครื่องมือ", "## เครื่องมือและเทคโนโลยี"),
        ("## วิธีการ", "## วิธีดำเนินงาน"),
    ],
)
def test_only_known_heading_aliases_are_mapped(alias: str, canonical: str) -> None:
    raw = _markdown("Project", "ข้อมูลจากต้นฉบับ").replace(canonical, alias, 1)
    result = refine_wiki_markdown("Generic source", raw)
    assert alias not in result.markdown
    assert canonical in result.markdown
    assert validate_wiki_markdown(result.markdown).is_valid


def test_duplicate_canonical_and_alias_sections_merge_distinct_content() -> None:
    raw = _markdown("Project").replace(
        f"## ผลลัพธ์\n{MISSING_INFORMATION_MARKER}",
        "## ผลลัพธ์\nผลชุดแรก: 10%\n\n## ผลการทดลอง\nผลชุดที่สอง: 12%",
        1,
    )
    result = refine_wiki_markdown("Generic source", raw)
    body = _section_body(result.markdown, "## ผลลัพธ์")
    assert body == "ผลชุดแรก: 10%\n\nผลชุดที่สอง: 12%"
    assert result.markdown.count("## ผลลัพธ์") == 1
    assert "duplicate_sections_merged:1" in result.changes
    assert validate_wiki_markdown(result.markdown).is_valid


def test_duplicate_missing_marker_is_dropped_when_other_section_has_content() -> None:
    raw = _markdown("Project").replace(
        f"## ผลลัพธ์\n{MISSING_INFORMATION_MARKER}",
        f"## ผลลัพธ์\n{MISSING_INFORMATION_MARKER}.\n\n## ผลการศึกษา\nผลจากเอกสารต้นฉบับ",
        1,
    )
    result = refine_wiki_markdown("Generic source", raw)
    assert _section_body(result.markdown, "## ผลลัพธ์") == "ผลจากเอกสารต้นฉบับ"
    assert validate_wiki_markdown(result.markdown).is_valid


def test_unknown_heading_remains_for_strict_validator_and_keeps_content() -> None:
    raw = _markdown("Project") + "\n\n## หัวข้อที่ไม่รู้จัก\nข้อมูลที่ต้องไม่หาย"
    result = refine_wiki_markdown("Generic source", raw)
    assert "## หัวข้อที่ไม่รู้จัก\nข้อมูลที่ต้องไม่หาย" in result.markdown
    assert not validate_wiki_markdown(result.markdown).is_valid


def test_document_071_method_and_keywords_are_folded_into_canonical_structure() -> None:
    keyword_values = "ระบบผู้เชี่ยวชาญ, การวินิจฉัยโรค"
    raw = _markdown("Document 071", "ข้อมูลจากเอกสาร").replace(
        "## วิธีดำเนินงาน\nข้อมูลจากเอกสาร",
        "## วิธีการ\nขั้นตอนที่รองรับโดยต้นฉบับ",
        1,
    )
    raw = raw.replace(
        "## ผลลัพธ์",
        f"## คำสำคัญ\n- {keyword_values}\n\n## ผลลัพธ์",
        1,
    )

    result = refine_wiki_markdown("Generic source", raw)

    h2_headings = [line for line in result.markdown.splitlines() if line.startswith("## ")]
    assert h2_headings == list(REQUIRED_WIKI_HEADINGS)
    assert "## วิธีการ" not in result.markdown
    assert "## คำสำคัญ" not in result.markdown
    assert _section_body(result.markdown, "## วิธีดำเนินงาน") == "ขั้นตอนที่รองรับโดยต้นฉบับ"
    assert f"คำสำคัญ: {keyword_values}" in _section_body(result.markdown, "## ภาพรวมโครงงาน")
    assert f"คำสำคัญ: - {keyword_values}" not in result.markdown
    assert keyword_values in result.markdown
    assert validate_wiki_markdown(result.markdown).is_valid


def test_keyword_section_is_not_duplicated_when_overview_already_has_values() -> None:
    keyword_line = "คำสำคัญ: ภาษาไทย, ระบบสารสนเทศ"
    raw = _markdown("Project", "ข้อมูลจากเอกสาร").replace(
        "## ภาพรวมโครงงาน\nข้อมูลจากเอกสาร",
        f"## ภาพรวมโครงงาน\nข้อมูลจากเอกสาร\n{keyword_line}",
        1,
    )
    raw = raw.replace(
        "## ปัญหาและที่มา",
        "## คำสำคัญ\nภาษาไทย, ระบบสารสนเทศ\n\n## ปัญหาและที่มา",
        1,
    )

    result = refine_wiki_markdown("Generic source", raw)

    assert result.markdown.count(keyword_line) == 1
    assert "## คำสำคัญ" not in result.markdown
    assert validate_wiki_markdown(result.markdown).is_valid


def test_inserted_overview_stays_exact_marker_even_with_source_metadata() -> None:
    raw = _markdown("ชื่อที่แต่งขึ้น").replace(
        f"{REQUIRED_WIKI_HEADINGS[0]}\n{MISSING_INFORMATION_MARKER}\n\n", "", 1
    )
    result = refine_wiki_markdown(_source(), raw)
    assert _section_body(result.markdown, REQUIRED_WIKI_HEADINGS[0]) == MISSING_INFORMATION_MARKER
    assert validate_wiki_markdown(result.markdown).is_valid


def test_llm_service_finalizes_document_071_structure_before_strict_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _document_071_like_raw()
    monkeypatch.setattr(llm_service.OllamaClient, "generate", lambda self, prompt: raw)

    result = llm_service.generate_wiki_result("Generic source text")

    assert result.raw_markdown == raw
    assert validate_wiki_markdown(result.markdown).is_valid
    assert "## ผลลัพธ์" in result.markdown
    assert "missing_sections_inserted:3" in result.refinement_changes


def test_finalizer_copies_title_and_metadata_and_removes_invented_metadata() -> None:
    raw = _markdown("ระบบที่แต่งชื่อขึ้น", f"{MISSING_INFORMATION_MARKER}.")
    raw = raw.replace(
        REQUIRED_WIKI_HEADINGS[0],
        REQUIRED_WIKI_HEADINGS[0]
        + "\n- มหาวิทยาลัย: จุฬาลงกรณ์มหาวิทยาลัย"
        + "\n- อาจารย์ที่ปรึกษา: ชื่อที่แต่งขึ้น",
        1,
    )
    raw = raw.replace(
        "- อาจารย์ที่ปรึกษา: ชื่อที่แต่งขึ้น",
        "- อาจารย์ที่ปรึกษา: ชื่อที่แต่งขึ้น\n- ผู้ช่วยที่ปรึกษา: ชื่อที่แต่งขึ้นอีกชื่อ",
        1,
    )
    raw = raw.replace(
        "- ผู้ช่วยที่ปรึกษา: ชื่อที่แต่งขึ้นอีกชื่อ",
        "- ผู้ช่วยที่ปรึกษา: ชื่อที่แต่งขึ้นอีกชื่อ\n- อาจารย์ที่ปรึกษาช่วย: ชื่อที่แต่งขึ้นอีกชื่อ",
        1,
    )

    result = refine_wiki_markdown(_source(), raw)

    assert result.markdown.startswith(
        "# WEB APPLICATION DEVELOPMENT OF SELF-ASSESSMENT OF LEARNING"
    )
    assert "- นักศึกษา: CHAWIT SALA" in result.markdown
    assert "- นักศึกษา: PERMPAT PINGKASAN" in result.markdown
    assert "- รหัสนักศึกษา: 61050033" in result.markdown
    assert "- รหัสนักศึกษา: 61050076" in result.markdown
    assert "- อาจารย์ที่ปรึกษา: Dr Smith" in result.markdown
    assert "- ปีการศึกษา: 2564" in result.markdown
    assert "จุฬาลงกรณ์มหาวิทยาลัย" not in result.markdown
    assert "ชื่อที่แต่งขึ้น" not in result.markdown
    assert "ชื่อที่แต่งขึ้นอีกชื่อ" not in result.markdown
    assert f"{MISSING_INFORMATION_MARKER}." not in result.markdown
    assert validate_wiki_markdown(result.markdown).is_valid
    assert "title_copied_from_focused_source" in result.changes


def test_finalizer_uses_missing_marker_when_no_title_is_reliable() -> None:
    source = "Source page 1 (title page):\nÖćøóĆçîćđüĘï"
    result = refine_wiki_markdown(source, _markdown("ชื่อที่แต่งขึ้น"))
    assert result.markdown.startswith(f"# {MISSING_INFORMATION_MARKER}\n")


def test_extra_same_line_marker_explanation_is_rejected() -> None:
    raw = _markdown("ชื่อโครงงาน", f"{MISSING_INFORMATION_MARKER} เพราะไม่มีข้อมูล")
    with pytest.raises(WikiOutputError, match="extra same-line content"):
        refine_wiki_markdown(_source(), raw)


def test_extra_section_content_after_marker_is_rejected() -> None:
    raw = _markdown("ชื่อโครงงาน", f"{MISSING_INFORMATION_MARKER}\nคำอธิบายเพิ่มเติม")
    with pytest.raises(WikiOutputError, match="extra section content"):
        refine_wiki_markdown(_source(), raw)


def test_redundant_tools_marker_is_removed_only_for_source_backed_tools() -> None:
    source = _source() + "\nSwift\niOS"
    raw = _markdown("ชื่อที่แต่งขึ้น").replace(
        f"{REQUIRED_WIKI_HEADINGS[3]}\n{MISSING_INFORMATION_MARKER}",
        f"{REQUIRED_WIKI_HEADINGS[3]}\n- ภาษา: Swift\n- ระบบปฏิบัติการ: iOS\n"
        f"{MISSING_INFORMATION_MARKER}.",
        1,
    )
    result = refine_wiki_markdown(source, raw)
    assert "- ภาษา: Swift" in result.markdown
    assert "redundant_tools_marker_removed" in result.changes
    assert validate_wiki_markdown(result.markdown).is_valid

    unsupported = raw.replace("Swift", "InventedTool")
    with pytest.raises(WikiOutputError, match="extra section content"):
        refine_wiki_markdown(source, unsupported)


def test_generic_text_generation_is_unchanged() -> None:
    raw = _markdown("Generic project")
    result = refine_wiki_markdown("Generic source text", raw)
    assert result.markdown == raw
    assert result.changes == ()


def test_llm_service_maps_unsafe_marker_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_service.OllamaClient,
        "generate",
        lambda self, prompt: _markdown(
            "Generic project", f"{MISSING_INFORMATION_MARKER} explanation"
        ),
    )
    with pytest.raises(llm_service.InvalidWikiOutputError, match="extra same-line content"):
        llm_service.generate_wiki("Generic source text")


def test_llm_result_keeps_raw_model_reply_and_refinement_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _markdown("ชื่อที่แต่งขึ้น", f"{MISSING_INFORMATION_MARKER}.")
    monkeypatch.setattr(llm_service.OllamaClient, "generate", lambda self, prompt: raw)

    result = llm_service.generate_wiki_result(_source())

    assert result.raw_markdown == raw
    assert result.markdown.startswith(
        "# WEB APPLICATION DEVELOPMENT OF SELF-ASSESSMENT OF LEARNING"
    )
    assert f"{MISSING_INFORMATION_MARKER}." not in result.markdown
    assert "title_copied_from_focused_source" in result.refinement_changes
