"""Read conservative title and metadata candidates from focused PDF text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_PAGE_LABEL = re.compile(r"(?m)^Source page (?P<number>\d+) \((?P<role>[^)]*)\):\n")
_EMBEDDED_DIGIT = re.compile(r"(?<=[\u0e00-\u0e7f])\d(?=[\u0e00-\u0e7f])")
_STUDENT_ID = re.compile(r"(?i)(?:student\s*id|รหัสนักศึกษา)[^\d\n]{0,16}(\d{6,12})")
_ACADEMIC_YEAR = re.compile(r"(?<!\d)25\d{2}(?!\d)")
_THAI_TITLE_LABEL = re.compile(r"(?im)^\s*(?:หัวข้อสหกิจศึกษา|หัวข้อโครงงาน)\s*[:：]?\s*(.+?)\s*$")
_THAI_YEAR_LABEL = re.compile(r"(?im)^\s*ปีการศึกษา\s*[:：]?\s*(25\d{2})\s*$")
_ENGLISH_YEAR_LABEL = re.compile(r"(?im)^\s*academic\s+year\s*[:：]?\s*(\d{4})\s*$")
_THAI_ADVISOR = re.compile(r"(?im)^\s*อาจารย์ที่ปรึกษา\s*[:：]?\s*(\S.+?)\s*$")
_ENGLISH_ADVISOR = re.compile(r"(?im)^\s*advisor\s*[:：]?\s*(\S.+?)\s*$")
_KEYWORD_START = re.compile(r"(?i)^\s*(?:keywords?|คำ\s*สำคัญ)\s*[:：]\s*(.*)$")
_KEYWORD_END = re.compile(
    r"(?i)^\s*(?:abstract|บทคัดย่อ|title|students?|student\s*id|advisor|academic\s*year|"
    r"หัวข้อ(?:สหกิจศึกษา|โครงงาน)|ชื่อนักศึกษา|รหัสนักศึกษา|อาจารย์ที่ปรึกษา|ปีการศึกษา)\b"
)
_PAGE_MARKER = re.compile(r"^(?:[ก-ฮ]|\d+)$")
_NAME_LINE = re.compile(r"^[A-Za-z][A-Za-z .'-]{5,59}$")
_NON_TITLE_WORDS = ("ACADEMIC YEAR", "SUBMITTED", "DEGREE", "DEPARTMENT", "UNIVERSITY")


@dataclass(frozen=True, slots=True)
class WikiSourceEvidence:
    """Candidates copied from the selected pages, never from GroundTruth."""

    title: str | None
    title_en: str | None
    students: tuple[str, ...]
    student_ids: tuple[str, ...]
    advisor: str | None
    academic_year: str | None
    keywords: tuple[str, ...]


def _pages(source_text: str) -> dict[int, str]:
    matches = list(_PAGE_LABEL.finditer(source_text))
    return {
        int(match.group("number")): source_text[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else None
        ].strip()
        for index, match in enumerate(matches)
    }


def _candidate_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _prefer_repeated_candidate(candidates: list[str]) -> str | None:
    """Prefer exact repeated evidence, preserving the first source form on ties."""

    cleaned = [" ".join(value.split()) for value in candidates if value.strip()]
    if not cleaned:
        return None
    counts = Counter(_candidate_key(value) for value in cleaned)
    highest = max(counts.values())
    return next(value for value in cleaned if counts[_candidate_key(value)] == highest)


def _join_keyword_fragments(fragments: list[str]) -> str:
    joined = ""
    for fragment in fragments:
        value = fragment.strip()
        if not value:
            continue
        if not joined:
            joined = value
        elif joined.rstrip().endswith((",", ";")):
            joined = f"{joined} {value}"
        elif "\u0e00" <= joined[-1] <= "\u0e7f" and "\u0e00" <= value[0] <= "\u0e7f":
            joined += value
        else:
            joined = f"{joined} {value}"
    return joined


def _extract_keywords(pages: dict[int, str]) -> tuple[str, ...]:
    """Read comma-separated keyword sections, including wrapped page continuations."""

    sections: list[list[str]] = []
    active: list[str] | None = None
    carry_to_next_page = False
    for page_number in sorted(pages):
        lines = pages[page_number].splitlines()
        continuing = carry_to_next_page and active is not None
        carry_to_next_page = False
        for line in lines:
            stripped = line.strip()
            if active is None:
                match = _KEYWORD_START.match(stripped)
                if match:
                    active = []
                    sections.append(active)
                    if match.group(1).strip():
                        active.append(match.group(1).strip())
                continue
            if continuing and (not stripped or _PAGE_MARKER.fullmatch(stripped)):
                continue
            continuing = False
            if not stripped or _KEYWORD_END.match(stripped):
                active = None
                continue
            match = _KEYWORD_START.match(stripped)
            if match:
                active = []
                sections.append(active)
                if match.group(1).strip():
                    active.append(match.group(1).strip())
                continue
            active.append(stripped)
        if active is not None:
            carry_to_next_page = bool(active and active[-1].rstrip().endswith((",", ";")))
            if not carry_to_next_page:
                active = None

    keywords: list[str] = []
    seen: set[str] = set()
    for section in sections:
        joined = _join_keyword_fragments(section)
        for value in re.split(r"[,;]", joined):
            keyword = value.strip()
            key = _candidate_key(keyword)
            if keyword and key not in seen:
                keywords.append(keyword)
                seen.add(key)
    return tuple(keywords)


def _english_title_line(line: str) -> bool:
    letters = [character for character in line if character.isascii() and character.isalpha()]
    if len(letters) < 8 or len(line.split()) < 2:
        return False
    if sum(character.isupper() for character in letters) / len(letters) < 0.7:
        return False
    return not any(word in line.upper() for word in _NON_TITLE_WORDS)


def _english_title(page_text: str) -> tuple[str | None, int | None]:
    lines = page_text.splitlines()
    for index, line in enumerate(lines):
        if not _english_title_line(line.strip()):
            continue
        group = [line.strip()]
        end = index + 1
        while end < len(lines) and _english_title_line(lines[end].strip()):
            group.append(lines[end].strip())
            end += 1
        candidate = " ".join(group)
        if len(candidate.split()) >= 4:
            return candidate, end
    return None, None


def _readable_thai_line(line: str) -> bool:
    thai_count = sum("\u0e00" <= character <= "\u0e7f" for character in line)
    if thai_count < 10 or thai_count / max(len(line.strip()), 1) < 0.55:
        return False
    if _EMBEDDED_DIGIT.search(line):
        return False
    return not any("\u0100" <= character <= "\u02ff" for character in line)


def _thai_title(page_text: str) -> str | None:
    lines = [line.strip() for line in page_text.splitlines()]
    first = next((index for index, line in enumerate(lines) if line), None)
    if first is None or not _readable_thai_line(lines[first]):
        return None
    title_lines = [lines[first]]
    for line in lines[first + 1 : first + 3]:
        if not line or not _readable_thai_line(line):
            break
        title_lines.append(line)
    return " ".join(title_lines)


def _english_student_names(page_text: str, title_end: int | None) -> tuple[str, ...]:
    if title_end is None:
        return ()
    lines = page_text.splitlines()
    following = lines[title_end:]
    start = next((index for index, line in enumerate(following) if line.strip()), None)
    if start is None:
        return ()
    names: list[str] = []
    for line in following[start : start + 5]:
        name = line.strip()
        if not name:
            break
        if not _NAME_LINE.fullmatch(name) or len(name.split()) not in (2, 3, 4):
            break
        if any(word in name.upper() for word in _NON_TITLE_WORDS):
            break
        names.append(name)
    return tuple(names)


def collect_wiki_source_evidence(source_text: str) -> WikiSourceEvidence:
    """Select exact readable title text and explicit source metadata when possible."""

    pages = _pages(source_text)
    if not pages:
        return WikiSourceEvidence(None, None, (), (), None, None, ())

    first_page = pages.get(1, "")
    english_title_candidates: list[str] = []
    student_names: tuple[str, ...] = ()
    for page_number in (1, 2, 3, 4, 5, 6):
        page = pages.get(page_number, "")
        candidate, end = _english_title(page)
        if candidate:
            english_title_candidates.append(candidate)
        if page_number == 2 and candidate:
            student_names = _english_student_names(page, end)
    title_en = _prefer_repeated_candidate(english_title_candidates)
    plain_source = _PAGE_LABEL.sub("", source_text)
    thai_title_candidates = [candidate for candidate in [_thai_title(first_page)] if candidate]
    thai_title_candidates.extend(_THAI_TITLE_LABEL.findall(plain_source))
    title_th = _prefer_repeated_candidate(thai_title_candidates)
    title = title_th or title_en
    student_ids = tuple(dict.fromkeys(_STUDENT_ID.findall(plain_source)))
    thai_years = _THAI_YEAR_LABEL.findall(plain_source)
    english_years = _ENGLISH_YEAR_LABEL.findall(plain_source)
    years = thai_years or english_years or _ACADEMIC_YEAR.findall(plain_source)
    advisor_candidates = _THAI_ADVISOR.findall(plain_source)
    if not advisor_candidates:
        advisor_candidates = _ENGLISH_ADVISOR.findall(plain_source)
    return WikiSourceEvidence(
        title=title,
        title_en=title_en,
        students=student_names,
        student_ids=student_ids,
        advisor=_prefer_repeated_candidate(advisor_candidates),
        academic_year=_prefer_repeated_candidate(years),
        keywords=_extract_keywords(pages),
    )
