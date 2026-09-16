"""Apply deterministic, source-backed title, metadata, and marker finalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.prompts.wiki_generation import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS
from app.services.wiki_evidence import WikiSourceEvidence, collect_wiki_source_evidence

_FOCUSED_PAGE = re.compile(r"(?m)^Source page \d+ \([^\n]*\):")
_HEADING_ALIASES = {
    "## ผลการศึกษา": REQUIRED_WIKI_HEADINGS[5],
    "## ผลการทดลอง": REQUIRED_WIKI_HEADINGS[5],
    "## ที่มาและปัญหา": REQUIRED_WIKI_HEADINGS[1],
    "## เทคโนโลยีและเครื่องมือ": REQUIRED_WIKI_HEADINGS[3],
    "## วิธีการ": REQUIRED_WIKI_HEADINGS[4],
}
_KEYWORDS_HEADING = "## คำสำคัญ"
_KEYWORD_LABEL = re.compile(r"^คำสำคัญ\s*[:：]\s*")
_LIST_PREFIX = re.compile(r"^[-*•]\s+")
_KEYWORD_OVERVIEW_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:คำสำคัญ|keywords?)\s*[:：]", re.IGNORECASE)
_MARKER_WITH_PUNCTUATION = re.compile(rf"^{re.escape(MISSING_INFORMATION_MARKER)}\s*[.:：。]+\s*$")
_METADATA_BULLET = re.compile(
    r"^\s*[-*]\s*(?:ชื่อโครงงาน|ชื่อโครงการภาษาอังกฤษ|ชื่อนักศึกษา|ชื่อผู้ศึกษา|"
    r"นักศึกษา|รหัสนักศึกษา|ปีการศึกษา|อาจารย์ที่ปรึกษา|ผู้ช่วยอาจารย์ที่ปรึกษา|"
    r"ผู้ช่วยที่ปรึกษา|ที่ปรึกษา|อาจารย์ที่ปรึกษาร่วม|[^:\n]*ที่ปรึกษา[^:\n]*|"
    r"ปริญญา|ภาควิชา|สาขา|คณะ|มหาวิทยาลัย|Project title|English title|"
    r"Student names?|Student IDs?|Advisor|Academic year)\s*[:：]",
    re.IGNORECASE,
)


class WikiOutputError(ValueError):
    """A marker has extra content that cannot safely be removed."""


@dataclass(frozen=True, slots=True)
class WikiOutputRefinement:
    markdown: str
    changes: tuple[str, ...]


def _trim_blank_lines(lines: list[str]) -> list[str]:
    """Discard only boundary whitespace, keeping the original section prose."""

    start = next((index for index, line in enumerate(lines) if line.strip()), len(lines))
    end = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()), -1)
    return lines[start : end + 1]


def _keyword_value(lines: list[str]) -> str:
    """Remove only section-label and list syntax while preserving keyword values."""

    values: list[str] = []
    for index, line in enumerate(lines):
        value = line.strip()
        if index == 0:
            value = _KEYWORD_LABEL.sub("", value)
        value = _LIST_PREFIX.sub("", value)
        if value:
            values.append(value)
    return ", ".join(values)


def _keyword_comparable(text: str) -> str:
    return re.sub(r"[\s,;:：•*-]+", "", _KEYWORD_LABEL.sub("", text))


def _normalize_wiki_structure(
    lines: list[str],
) -> tuple[list[str], tuple[str, ...], frozenset[str]]:
    """Repair only known section labels, order, omissions, and safe duplicates."""

    positions = [index for index, line in enumerate(lines) if line.startswith("## ")]
    sections = [
        (
            lines[position],
            lines[position + 1 : positions[index + 1] if index + 1 < len(positions) else None],
        )
        for index, position in enumerate(positions)
    ]
    bodies: dict[str, list[list[str]]] = {heading: [] for heading in REQUIRED_WIKI_HEADINGS}
    unknown: list[tuple[str, list[str]]] = []
    keyword_blocks: list[list[str]] = []
    aliases = 0
    known_order: list[str] = []
    for heading, body in sections:
        stripped_heading = heading.strip()
        if stripped_heading == _KEYWORDS_HEADING:
            keyword_blocks.append(_trim_blank_lines(body))
            continue
        canonical = _HEADING_ALIASES.get(stripped_heading, heading)
        if canonical != heading:
            aliases += 1
        if canonical in bodies:
            bodies[canonical].append(_trim_blank_lines(body))
            known_order.append(canonical)
        else:
            unknown.append((heading, body))

    moved_keywords = 0
    overview_blocks = bodies[REQUIRED_WIKI_HEADINGS[0]]
    for block in keyword_blocks:
        content = [line.strip() for line in block if line.strip()]
        if not content:
            continue
        value = _keyword_value(content)
        if not value:
            continue
        overview_text = " ".join(
            line.strip() for overview in overview_blocks for line in overview if line.strip()
        )
        if _keyword_comparable(value) in _keyword_comparable(overview_text):
            continue
        keyword_lines = [f"คำสำคัญ: {value}"]
        if not overview_blocks:
            overview_blocks.append(keyword_lines)
        else:
            target = overview_blocks[0]
            if target == [MISSING_INFORMATION_MARKER]:
                target.clear()
            if target:
                target.append("")
            target.extend(keyword_lines)
        moved_keywords += 1

    missing = [heading for heading, blocks in bodies.items() if not blocks]
    duplicates = sum(max(len(blocks) - 1, 0) for blocks in bodies.values())
    canonical_indices = [REQUIRED_WIKI_HEADINGS.index(heading) for heading in known_order]
    reordered = canonical_indices != sorted(canonical_indices)
    if not (aliases or keyword_blocks or missing or duplicates or reordered):
        return lines, (), frozenset()

    changes: list[str] = []
    if aliases:
        changes.append(f"heading_alias_normalized:{aliases}")
    if keyword_blocks:
        changes.append(f"keyword_sections_removed:{len(keyword_blocks)}")
    if moved_keywords:
        changes.append(f"keyword_sections_moved:{moved_keywords}")
    if missing:
        changes.append(f"missing_sections_inserted:{len(missing)}")
    if duplicates:
        changes.append(f"duplicate_sections_merged:{duplicates}")
    if reordered:
        changes.append("sections_reordered")

    normalized = lines[: positions[0]] if positions else lines.copy()
    for heading in REQUIRED_WIKI_HEADINGS:
        blocks = bodies[heading]
        if not blocks:
            content = [MISSING_INFORMATION_MARKER]
        else:
            substantive = any(block and block != [MISSING_INFORMATION_MARKER] for block in blocks)
            unique: list[list[str]] = []
            for block in blocks:
                if substantive and block == [MISSING_INFORMATION_MARKER]:
                    continue
                if block not in unique:
                    unique.append(block)
            content = [
                line
                for index, block in enumerate(unique)
                for line in ([""] if index else []) + block
            ]
        normalized.extend(["", heading, "", *content])
    for heading, body in unknown:
        normalized.extend(["", heading, *body])
    return normalized, tuple(changes), frozenset(missing)


def _literal_source_match(source_text: str, title: str) -> bool:
    normalized_source = " ".join(unicodedata.normalize("NFC", source_text).split())
    normalized_title = " ".join(unicodedata.normalize("NFC", title).split())
    return bool(normalized_title) and normalized_title in normalized_source


def _canonical_metadata_lines(evidence: WikiSourceEvidence, title: str) -> list[str]:
    lines: list[str] = []
    if evidence.title_en and evidence.title_en != title:
        lines.append(f"- ชื่อโครงการภาษาอังกฤษ: {evidence.title_en}")
    lines.extend(f"- นักศึกษา: {name}" for name in evidence.students)
    lines.extend(f"- รหัสนักศึกษา: {identifier}" for identifier in evidence.student_ids)
    if evidence.advisor:
        lines.append(f"- อาจารย์ที่ปรึกษา: {evidence.advisor}")
    if evidence.academic_year:
        lines.append(f"- ปีการศึกษา: {evidence.academic_year}")
    if evidence.keywords:
        lines.append(f"คำสำคัญ: {', '.join(evidence.keywords)}")
    return lines


def refine_wiki_markdown(source_text: str, markdown: str) -> WikiOutputRefinement:
    """Use exact selected-page facts; reject marker lines with nontrivial extra text."""

    lines = markdown.splitlines()
    changes: list[str] = []
    punctuation_fixes = 0
    for index, line in enumerate(lines):
        if MISSING_INFORMATION_MARKER not in line or line == f"# {MISSING_INFORMATION_MARKER}":
            continue
        if line == MISSING_INFORMATION_MARKER:
            continue
        if _MARKER_WITH_PUNCTUATION.fullmatch(line):
            lines[index] = MISSING_INFORMATION_MARKER
            punctuation_fixes += 1
        else:
            raise WikiOutputError("Missing-information marker has extra same-line content")
    if punctuation_fixes:
        changes.append(f"marker_punctuation_removed:{punctuation_fixes}")

    lines, structure_changes, inserted = _normalize_wiki_structure(lines)
    changes.extend(structure_changes)
    focused = bool(_FOCUSED_PAGE.search(source_text))
    evidence = collect_wiki_source_evidence(source_text) if focused else None
    title_positions = [index for index, line in enumerate(lines) if line.startswith("# ")]
    if focused and len(title_positions) == 1:
        generated_title = lines[title_positions[0]][2:].strip()
        preferred_title = evidence.title or (
            generated_title
            if _literal_source_match(source_text, generated_title)
            else MISSING_INFORMATION_MARKER
        )
        preferred_heading = f"# {preferred_title}"
        if lines[title_positions[0]] != preferred_heading:
            lines[title_positions[0]] = preferred_heading
            changes.append("title_copied_from_focused_source")

    if focused and REQUIRED_WIKI_HEADINGS[0] in lines and REQUIRED_WIKI_HEADINGS[0] not in inserted:
        overview_start = lines.index(REQUIRED_WIKI_HEADINGS[0]) + 1
        overview_end = next(
            (
                index
                for index in range(overview_start, len(lines))
                if lines[index].startswith("## ")
            ),
            len(lines),
        )
        original_body = lines[overview_start:overview_end]
        cleaned_body = [line for line in original_body if not _METADATA_BULLET.match(line)]
        if evidence.keywords:
            cleaned_body = [line for line in cleaned_body if not _KEYWORD_OVERVIEW_LINE.match(line)]
        while cleaned_body and not cleaned_body[0].strip():
            cleaned_body.pop(0)
        title = lines[title_positions[0]][2:].strip() if len(title_positions) == 1 else ""
        metadata_lines = _canonical_metadata_lines(evidence, title)
        if metadata_lines:
            cleaned_body = [line for line in cleaned_body if line != MISSING_INFORMATION_MARKER]
            new_body = ["", *metadata_lines, "", *cleaned_body]
        else:
            new_body = ["", *cleaned_body]
        if new_body != original_body:
            lines[overview_start:overview_end] = new_body
            changes.append("overview_metadata_rebuilt_from_focused_source")

    tools_heading = REQUIRED_WIKI_HEADINGS[3]
    if focused and tools_heading in lines:
        tools_start = lines.index(tools_heading) + 1
        tools_end = next(
            (index for index in range(tools_start, len(lines)) if lines[index].startswith("## ")),
            len(lines),
        )
        tools_body = [
            (index, lines[index].strip())
            for index in range(tools_start, tools_end)
            if lines[index].strip()
        ]
        redundant_markers = [
            index for index, line in tools_body if line == MISSING_INFORMATION_MARKER
        ]
        tool_bullets = [line for _, line in tools_body if line != MISSING_INFORMATION_MARKER]
        if redundant_markers and tool_bullets:
            values = [re.match(r"^[-*]\s*[^:\n]+[:：]\s*(.+)$", line) for line in tool_bullets]
            if all(
                match and _literal_source_match(source_text, match.group(1)) for match in values
            ):
                for index in reversed(redundant_markers):
                    del lines[index]
                changes.append("redundant_tools_marker_removed")

    for heading in REQUIRED_WIKI_HEADINGS:
        if heading not in lines:
            continue
        start = lines.index(heading) + 1
        end = next(
            (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
            len(lines),
        )
        body = [line.strip() for line in lines[start:end] if line.strip()]
        if MISSING_INFORMATION_MARKER in body and body != [MISSING_INFORMATION_MARKER]:
            raise WikiOutputError("Missing-information marker has extra section content")

    if not changes:
        return WikiOutputRefinement(markdown, ())
    return WikiOutputRefinement("\n".join(lines).strip() + "\n", tuple(changes))
