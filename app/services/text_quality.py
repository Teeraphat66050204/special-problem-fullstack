"""Deterministic quality checks for native PDF front-matter text."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.pdf_extractor import PdfPageText

IMPORTANT_THAI_LABELS: tuple[str, ...] = (
    "ชื่อนักศึกษา",
    "อาจารย์ที่ปรึกษา",
    "ปีการศึกษา",
    "บทคัดย่อ",
    "คำสำคัญ",
)

_THAI_CHARACTER = re.compile(r"[\u0e00-\u0e7f]")
_ASCII_SUBSTITUTION_IN_THAI = re.compile(
    r"(?:[\u0e00-\u0e7f][0-9%<>=|][\u0e00-\u0e7f]|"
    r"[0-9%<>=|][\u0e00-\u0e7f]{2})"
)
_KNOWN_CORRUPTION_CHARACTER = re.compile(r"[\u0100-\u02ff\ufffd]")
_KNOWN_DATASET_GLYPHS = frozenset("ÿĂćĉĊüđøÖĒþåðĕŤŠęúĆǰ")
_TRAILING_TITLE_START = re.compile(r"^(?:และ|หรือ|and\b|or\b)", re.IGNORECASE)
_TITLE_METADATA = re.compile(
    r"(?i)(?:ชื่อนักศึกษา|รหัสนักศึกษา|student\s*ids?|advisor|academic\s+year|"
    r"degree|department|university|abstract|keywords?)"
)
_EXPECTED_THAI_PAGE_NUMBERS = frozenset((1, 3, 4))


@dataclass(frozen=True, slots=True)
class TextQualityAssessment:
    """A bounded score, degradation decision, and explainable diagnostics."""

    score: float
    degraded: bool
    reasons: tuple[str, ...]
    degraded_page_numbers: tuple[int, ...] = ()


def _visible_characters(text: str) -> list[str]:
    return [character for character in text if not character.isspace()]


def _title_fragment_reason(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidates = [line for line in lines[:8] if not _TITLE_METADATA.search(line)]
    if not candidates:
        return "no plausible title candidate near the page top"
    first = candidates[0]
    if _TRAILING_TITLE_START.match(first):
        return "top title candidate starts with a continuation word"
    if len(first) < 12:
        return "top title candidate is suspiciously short"
    return None


def assess_text_quality(
    text: str,
    *,
    expected_thai_metadata: bool = False,
) -> TextQualityAssessment:
    """Score one page without translating, correcting, or inferring its text."""

    visible = _visible_characters(text)
    visible_count = len(visible)
    thai_count = len(_THAI_CHARACTER.findall(text))
    thai_ratio = thai_count / visible_count if visible_count else 0.0
    corruption_count = len(_KNOWN_CORRUPTION_CHARACTER.findall(text))
    dataset_glyph_count = sum(character in _KNOWN_DATASET_GLYPHS for character in text)
    substitution_count = len(_ASCII_SUBSTITUTION_IN_THAI.findall(text))

    score = 1.0
    reasons: list[str] = []
    severe = False
    if not text.strip():
        reasons.append("page has no native text")
        score -= 0.9
        severe = True
    elif visible_count < 20:
        reasons.append("page text is suspiciously short")
        score -= 0.3

    if corruption_count >= 3 and corruption_count / max(visible_count, 1) >= 0.02:
        reasons.append("known PDF glyph-corruption characters are frequent")
        score -= 0.65
        severe = True
    elif dataset_glyph_count >= 3:
        reasons.append("known dataset glyph-corruption pattern detected")
        score -= 0.55
        severe = True

    if substitution_count >= 2:
        reasons.append("abnormal ASCII or symbol substitutions occur inside Thai text")
        score -= 0.4

    if expected_thai_metadata and visible_count >= 40 and thai_ratio < 0.12:
        reasons.append("Thai-character ratio is very low for an expected Thai metadata page")
        score -= 0.3

    fragment_reason = _title_fragment_reason(text)
    if fragment_reason:
        reasons.append(fragment_reason)
        score -= 0.5 if "continuation word" in fragment_reason else 0.2

    bounded_score = round(max(0.0, min(score, 1.0)), 3)
    return TextQualityAssessment(
        score=bounded_score,
        degraded=severe or bounded_score < 0.55,
        reasons=tuple(reasons),
    )


def assess_front_matter_quality(
    pages: Sequence[PdfPageText],
) -> TextQualityAssessment:
    """Assess only the supplied front-matter pages and identify OCR candidates."""

    if not pages:
        return TextQualityAssessment(
            score=0.0,
            degraded=True,
            reasons=("no front-matter pages were available",),
            degraded_page_numbers=(),
        )

    page_assessments = {
        page.page_number: assess_text_quality(
            page.text,
            expected_thai_metadata=page.page_number in _EXPECTED_THAI_PAGE_NUMBERS,
        )
        for page in pages
    }
    combined = "\n\n".join(page.text for page in pages)
    missing_labels = [label for label in IMPORTANT_THAI_LABELS if label not in combined]
    thai_count = len(_THAI_CHARACTER.findall(combined))
    material_page_reasons = (
        "known PDF glyph-corruption",
        "known dataset glyph-corruption",
        "abnormal ASCII or symbol substitutions",
        "Thai-character ratio is very low",
        "page has no native text",
        "top title candidate starts with a continuation word",
    )
    suspicious_pages = [
        number
        for number, assessment in page_assessments.items()
        if assessment.degraded
        or any(reason.startswith(material_page_reasons) for reason in assessment.reasons)
    ]
    reasons = [
        f"page {number}: {reason}"
        for number, assessment in page_assessments.items()
        for reason in assessment.reasons
    ]

    score = sum(assessment.score for assessment in page_assessments.values()) / len(pages)
    missing_labels_are_material = len(pages) >= 3 and (thai_count >= 30 or bool(suspicious_pages))
    if missing_labels:
        reasons.append("important Thai labels missing: " + ", ".join(missing_labels))
    if missing_labels and missing_labels_are_material:
        score -= min(0.5, 0.1 * len(missing_labels))

    degraded = any(assessment.degraded for assessment in page_assessments.values()) or score < 0.55
    if degraded and not suspicious_pages:
        suspicious_pages = [
            page.page_number for page in pages if page.page_number in _EXPECTED_THAI_PAGE_NUMBERS
        ]
    if degraded and not suspicious_pages:
        suspicious_pages = [page.page_number for page in pages]

    return TextQualityAssessment(
        score=round(max(0.0, min(score, 1.0)), 3),
        degraded=degraded,
        reasons=tuple(dict.fromkeys(reasons)),
        degraded_page_numbers=tuple(suspicious_pages if degraded else ()),
    )
