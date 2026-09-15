"""Read conservative title and metadata candidates from focused PDF text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PAGE_LABEL = re.compile(r"(?m)^Source page (?P<number>\d+) \((?P<role>[^)]*)\):\n")
_EMBEDDED_DIGIT = re.compile(r"(?<=[\u0e00-\u0e7f])\d(?=[\u0e00-\u0e7f])")
_STUDENT_ID = re.compile(r"(?i)(?:student\s*id|รหัสนักศึกษา)[^\d\n]{0,16}(\d{6,12})")
_ACADEMIC_YEAR = re.compile(r"(?<!\d)25\d{2}(?!\d)")
_ADVISOR = re.compile(r"(?im)^advisor\s*[:：]\s*(.+)$")
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


def _pages(source_text: str) -> dict[int, str]:
    matches = list(_PAGE_LABEL.finditer(source_text))
    return {
        int(match.group("number")): source_text[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else None
        ].strip()
        for index, match in enumerate(matches)
    }


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
        return WikiSourceEvidence(None, None, (), (), None, None)

    first_page = pages.get(1, "")
    title_en, _ = _english_title(first_page)
    student_names: tuple[str, ...] = ()
    for page_number in (2, 1, 3, 4):
        page = pages.get(page_number, "")
        candidate, end = _english_title(page)
        if title_en is None and candidate:
            title_en = candidate
        if page_number == 2 and candidate:
            student_names = _english_student_names(page, end)
    title_th = _thai_title(first_page)
    title = title_th or title_en
    plain_source = _PAGE_LABEL.sub("", source_text)
    student_ids = tuple(dict.fromkeys(_STUDENT_ID.findall(plain_source)))
    year_match = _ACADEMIC_YEAR.search(first_page) or _ACADEMIC_YEAR.search(plain_source)
    advisor_match = _ADVISOR.search(plain_source)
    return WikiSourceEvidence(
        title=title,
        title_en=title_en,
        students=student_names,
        student_ids=student_ids,
        advisor=advisor_match.group(1).strip() if advisor_match else None,
        academic_year=year_match.group() if year_match else None,
    )
