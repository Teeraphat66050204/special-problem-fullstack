"""Reference and source-evidence checks for generated Wiki Markdown.

These lexical checks are review aids, not a semantic proof of factual accuracy.
GroundTruth is a reference only; claims must also be supported by focused PDF text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from app.datasets.groundtruth import GroundTruthRecord
from app.prompts import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS, validate_wiki_markdown
from app.services.wiki_evidence import collect_wiki_source_evidence

_SOURCE_LABEL = re.compile(r"(?m)^Source page \d+ \([^\n]*\):\n")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:\s*[-–]\s*\d+(?:[.,]\d+)*)?(?:\s*%|\s*°C)?(?!\w)")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_TECHNICAL_TERM = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9-]*|"
    r"[a-z]+[A-Z][A-Za-z0-9-]*|[A-Za-z]+\d+(?:\.\d+)*)\b"
)
_FINDING_CUE = re.compile(
    r"พบว่า|ผลการศึกษา|ผลการทดลอง|ผลการทดสอบ|สรุปว่า|"
    r"results? (?:show|indicat)|found that|concluded that",
    re.IGNORECASE,
)
_STRONG_CLAIMS = (
    "ดีที่สุด",
    "พิสูจน์",
    "รับรอง",
    "ปลอดภัย",
    "สำเร็จ",
    "ไม่สามารถ",
    "ไม่เป็นระบบ",
    "ความแม่นยำ",
    "best",
    "proven",
    "guaranteed",
)
_SECTION_CUES = {
    REQUIRED_WIKI_HEADINGS[1]: ("ปัญหา", "เนื่องจาก", "problem", "challenge"),
    REQUIRED_WIKI_HEADINGS[2]: ("วัตถุประสงค์", "จุดประสงค์", "มีเป้าหมาย", "objective", "aim"),
    REQUIRED_WIKI_HEADINGS[3]: (
        "ใช้",
        "วัสดุ",
        "เครื่องมือ",
        "เทคโนโลยี",
        "สาร",
        "ภาษา",
        "platform",
        "software",
        "using",
    ),
    REQUIRED_WIKI_HEADINGS[4]: ("วิธี", "ขั้นตอน", "ทดลอง", "ดำเนิน", "พัฒนา", "method", "tested"),
    REQUIRED_WIKI_HEADINGS[5]: ("พบว่า", "ผลการ", "ผลลัพธ์", "เพิ่มขึ้น", "ลดลง", "results", "found"),
}


def _plain_source(source_text: str) -> str:
    return _SOURCE_LABEL.sub("", source_text)


def _comparable(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", normalized).casefold()


def _contains(text: str, phrase: str) -> bool:
    return _comparable(phrase) in _comparable(text)


def _literal_layout(text: str) -> str:
    """Join layout whitespace while preserving every letter, digit, and case."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def _literal_contains(text: str, phrase: str) -> bool:
    return bool(phrase) and _literal_layout(phrase) in _literal_layout(text)


def _same_spelling(left: str, right: str) -> bool:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", left)) == re.sub(
        r"\s+", "", unicodedata.normalize("NFC", right)
    )


def _unique_tokens(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group().strip() for match in pattern.finditer(text)))


def _coverage(reference: list[str], source: str, markdown: str) -> dict[str, Any]:
    available = [value for value in reference if _contains(source, value)]
    unavailable = [value for value in reference if value not in available]
    covered = [value for value in available if _contains(markdown, value)]
    missing = [value for value in available if value not in covered]
    return {
        "reference": reference,
        "available_in_source": available,
        "unavailable_in_source": unavailable,
        "covered": covered,
        "missing": missing,
        "rate": len(covered) / len(available) if available else None,
    }


def _metadata_coverage(
    focused_source_text: str, source: str, markdown: str, groundtruth: GroundTruthRecord
) -> dict[str, Any]:
    """Include GroundTruth references and extra legible metadata copied from PDF."""

    evidence = collect_wiki_source_evidence(focused_source_text)
    entries: list[tuple[str, str]] = [("student", student) for student in groundtruth.students]
    entries.extend(
        (field, str(value))
        for field, value in (
            ("english_title", groundtruth.title_en),
            ("advisor", groundtruth.advisor),
            ("academic_year", groundtruth.academic_year),
        )
        if value is not None
    )
    if evidence.title:
        entries.append(("project_title", evidence.title))
    if evidence.title_en:
        entries.append(("english_title", evidence.title_en))
    entries.extend(("student", name) for name in evidence.students)
    entries.extend(("student_id", identifier) for identifier in evidence.student_ids)
    if evidence.advisor:
        entries.append(("advisor", evidence.advisor))
    if evidence.academic_year:
        entries.append(("academic_year", evidence.academic_year))

    unique_entries: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for field, value in entries:
        if value not in seen_values:
            unique_entries.append((field, value))
            seen_values.add(value)
    items = [
        {
            "field": field,
            "value": value,
            "source_supported": _literal_contains(source, value),
            "generated_present": _literal_contains(markdown, value),
        }
        for field, value in unique_entries
    ]
    for item in items:
        item["status"] = (
            "unavailable_in_source"
            if not item["source_supported"]
            else "covered"
            if item["generated_present"]
            else "missing"
        )
    available = [item for item in items if item["source_supported"]]
    missing = [item for item in available if not item["generated_present"]]
    missing_by_field: dict[str, list[str]] = {}
    for item in missing:
        missing_by_field.setdefault(item["field"], []).append(item["value"])
    return {
        "items": items,
        "reference": [item["value"] for item in items],
        "fields": [item["field"] for item in items],
        "available_in_source": [item["value"] for item in available],
        "unavailable_in_source": [item["value"] for item in items if not item["source_supported"]],
        "covered": [item["value"] for item in available if item["generated_present"]],
        "missing": [item["value"] for item in missing],
        "missing_by_field": missing_by_field,
        "rate": (len(available) - len(missing)) / len(available) if available else None,
    }


def _reference_findings(record: GroundTruthRecord) -> list[str]:
    excerpts: list[str] = []
    for abstract in (record.abstract_th, record.abstract_en):
        if not abstract:
            continue
        for match in _FINDING_CUE.finditer(abstract):
            tail = abstract[match.start() : match.start() + 240]
            excerpt = re.split(r"(?<!\d)[.!?](?!\d)|\n", tail, maxsplit=1)[0].strip()
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)
            if len(excerpts) >= 3:
                return excerpts
    return excerpts


def _section_bodies(markdown: str) -> dict[str, str]:
    bodies: dict[str, list[str]] = {}
    active: str | None = None
    for line in markdown.splitlines():
        if line in REQUIRED_WIKI_HEADINGS:
            active = line
            bodies.setdefault(line, [])
        elif active is not None and line.startswith("## "):
            active = None
        elif active is not None:
            bodies[active].append(line)
    return {heading: "\n".join(lines).strip() for heading, lines in bodies.items()}


def _hallucination_indicators(source: str, markdown: str) -> list[dict[str, str]]:
    indicators: list[dict[str, str]] = []
    for kind, pattern in (("number", _NUMBER), ("technical_term", _TECHNICAL_TERM)):
        for value in _unique_tokens(pattern, markdown):
            if not _contains(source, value):
                indicators.append({"kind": kind, "value": value})
    for value in _STRONG_CLAIMS:
        if _contains(markdown, value) and not _contains(source, value):
            indicators.append({"kind": "strengthened_claim", "value": value})
    return indicators


def evaluate_markdown(
    source_text: str, markdown: str, groundtruth: GroundTruthRecord
) -> dict[str, Any]:
    """Compare output to GroundTruth while requiring source evidence for coverage."""

    source = _plain_source(source_text)
    validation = validate_wiki_markdown(markdown)
    notes: list[str] = []
    issues: list[str] = []
    title = groundtruth.title_th or groundtruth.title_en
    evidence = collect_wiki_source_evidence(source_text)
    heading = next(
        (line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), ""
    )
    title_available = bool(title and _contains(source, title))
    title_match = bool(title and _same_spelling(heading, title))
    title_is_marker = heading == MISSING_INFORMATION_MARKER
    title_grounded = title_is_marker or _literal_contains(source, heading)
    if heading and not title_grounded:
        notes.append("Generated project title is not intact in focused PDF text")
        issues.append("title_not_grounded")
    if title and not title_available:
        notes.append(
            "GroundTruth title is not intact in focused PDF text; title grounding needs review"
        )
        issues.append("source_reference_gap")
    elif title and not title_match:
        notes.append("Generated title does not preserve the source-supported GroundTruth title")
        issues.append("title_mismatch")

    metadata = _metadata_coverage(source_text, source, markdown, groundtruth)
    keywords = _coverage(groundtruth.keywords, source, markdown)
    if metadata["missing"]:
        detail = ", ".join(
            f"{field}: {', '.join(values)}"
            for field, values in metadata["missing_by_field"].items()
        )
        notes.append(f"Missing source-supported metadata: {detail}")
        issues.append("metadata_missing")
    if keywords["missing"]:
        notes.append(f"Missing {len(keywords['missing'])} source-supported keyword(s)")
        issues.append("keywords_missing")
    if metadata["unavailable_in_source"] or keywords["unavailable_in_source"]:
        notes.append("Some GroundTruth metadata/keywords are not intact in focused PDF text")
        issues.append("source_reference_gap")

    finding_excerpts = _reference_findings(groundtruth)
    anchors = list(
        dict.fromkeys(
            token
            for excerpt in finding_excerpts
            for pattern in (_NUMBER, _TECHNICAL_TERM)
            for token in _unique_tokens(pattern, excerpt)
        )
    )
    findings = _coverage(anchors, source, markdown)
    findings["reference_excerpts"] = finding_excerpts
    findings["needs_manual_review"] = not bool(findings["available_in_source"])
    if findings["missing"]:
        notes.append(f"Missing {len(findings['missing'])} source-supported finding anchor(s)")
        issues.append("finding_anchors_missing")
    if not finding_excerpts:
        notes.append("No explicit finding cue in GroundTruth abstract; results need manual review")
    elif findings["needs_manual_review"]:
        notes.append("GroundTruth finding has no exact numeric/technical anchor in focused source")

    indicators = _hallucination_indicators(source, markdown)
    if _HAN.search(markdown) and not _HAN.search(source):
        indicators.append({"kind": "non_source_script", "value": "Han characters"})
    if indicators:
        notes.append(
            f"Review {len(indicators)} possible unsupported number/term/strong-claim indicator(s)"
        )
        issues.append("hallucination_indicator")

    section_bodies = _section_bodies(markdown)
    empty_sections = [
        heading for heading in REQUIRED_WIKI_HEADINGS if not section_bodies.get(heading, "")
    ]
    if empty_sections:
        notes.append(f"{len(empty_sections)} required section(s) have empty bodies")
        issues.append("empty_section")
    marker_sections = [
        heading for heading, body in section_bodies.items() if MISSING_INFORMATION_MARKER in body
    ]
    likely_unsupported = [
        heading
        for heading, cues in _SECTION_CUES.items()
        if not any(_contains(source, cue) for cue in cues)
    ]
    possible_marker_omissions = [
        heading
        for heading in likely_unsupported
        if section_bodies.get(heading, "") != MISSING_INFORMATION_MARKER
    ]
    mixed_marker_sections = [
        heading
        for heading in marker_sections
        if section_bodies.get(heading) != MISSING_INFORMATION_MARKER
    ]
    marker_lines = [
        line
        for line in markdown.splitlines()
        if MISSING_INFORMATION_MARKER in line and not line.startswith("# ")
    ]
    invalid_marker_lines = [line for line in marker_lines if line != MISSING_INFORMATION_MARKER]
    marker_correct = not invalid_marker_lines and not mixed_marker_sections
    marker_usage = {
        "used_in_sections": marker_sections,
        "likely_unsupported_sections": likely_unsupported,
        "possible_omissions": possible_marker_omissions,
        "mixed_with_other_content": mixed_marker_sections,
        "empty_sections": empty_sections,
        "exact_format_valid": marker_correct,
        "exact_line_count": len(marker_lines) - len(invalid_marker_lines),
        "invalid_lines": invalid_marker_lines,
    }
    if not marker_correct:
        notes.append(
            f"Missing-information marker is not exact in {len(invalid_marker_lines)} line(s)"
        )
        issues.append("invalid_marker_format")
    if possible_marker_omissions:
        notes.append(
            "Review missing-information marker usage in sections without clear source evidence"
        )
        issues.append("marker_issue")
    if not validation.is_valid:
        issues.append("invalid_structure")

    return {
        "structure": {
            "valid": validation.is_valid,
            "issues": [issue.code for issue in validation.issues],
        },
        "title": {
            "groundtruth": title,
            "focused_source_candidate": evidence.title or MISSING_INFORMATION_MARKER,
            "generated": heading,
            "generated_is_missing_marker": title_is_marker,
            "grounding_status": "missing_marker"
            if title_is_marker
            else "pass"
            if title_grounded
            else "fail",
            "available_in_source": title_available,
            "generated_grounded_in_source": title_grounded,
            "generated_matches_groundtruth": title_match,
            "status": "not_applicable"
            if not title
            else "source_gap"
            if not title_available
            else "pass"
            if title_match
            else "fail",
        },
        "metadata": metadata,
        "keywords": keywords,
        "findings": findings,
        "hallucination_indicators": indicators,
        "missing_marker_usage": marker_usage,
        "notes": list(dict.fromkeys(notes)),
        "issue_categories": list(dict.fromkeys(issues)),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize generation and quality without counting outages as structure failures."""

    successful = [result for result in results if result["status"] == "success"]
    assessed = [result for result in results if result.get("structure_valid") is not None]
    structure_passes = sum(result["structure_valid"] is True for result in assessed)
    categories: Counter[str] = Counter()
    raw_categories: Counter[str] = Counter()
    for result in results:
        if result.get("failure_category"):
            categories.update([result["failure_category"]])
        categories.update(result.get("issue_categories", []))
        raw_categories.update(result.get("raw_model_issue_categories", []))
    return {
        "total_documents": len(results),
        "generation_successes": len(successful),
        "generation_failures": len(results) - len(successful),
        "structure_assessed": len(assessed),
        "structure_passes": structure_passes,
        "structure_pass_rate": structure_passes / len(assessed) if assessed else None,
        "common_failure_categories": dict(categories.most_common()),
        "raw_model_quality_categories": dict(raw_categories.most_common()),
        "per_document_notes": [
            {
                "document_id": result["document_id"],
                "status": result["status"],
                "notes": result.get("notes", []),
                "refinement_changes": result.get("refinement_changes", []),
                "raw_model_notes": result.get("raw_model_notes", []),
            }
            for result in results
        ],
    }
