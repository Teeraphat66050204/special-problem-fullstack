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
