"""Read conservative title and metadata candidates from focused PDF text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_PAGE_LABEL = re.compile(r"(?m)^Source page (?P<number>\d+) \((?P<role>[^)]*)\):\n")
_EMBEDDED_DIGIT = re.compile(r"(?<=[\u0e00-\u0e7f])\d(?=[\u0e00-\u0e7f])")
_STUDENT_ID = re.compile(r"(?i)(?:student\s*ids?|รหัสนักศึกษา)[^\d\n]{0,24}(\d{6,12})")
_ANY_STUDENT_ID = re.compile(r"(?<!\d)(\d{6,12})(?!\d)")
_ACADEMIC_YEAR = re.compile(r"(?<!\d)25\d{2}(?!\d)")
_THAI_TITLE_LABEL = re.compile(
    r"^\s*(?:หัวข้อสหกิจศึกษา|หัวข้อโครงงาน(?:พิเศษ)?|หัวข้อปัญหาพิเศษ)"
    r"\s*[:：]?\s+(.*?)\s*$"
)
_ENGLISH_TITLE_LABEL = re.compile(
    r"(?i)^\s*(?:english\s+title|project\s+title|title)\s*[:：]?\s*(.*?)\s*$"
)
_STUDENT_LABEL = re.compile(
    r"(?i)^\s*(?:ชื่อนักศึกษา|นักศึกษา|student\s+names?|students?)\s*[:：]?\s*(.*?)\s*$"
)
_STUDENT_ID_LABEL = re.compile(r"(?i)^\s*(?:student\s*ids?|studentid|รหัสนักศึกษา)\b\s*[:：]?\s*(.*)$")
_ATTACHED_STUDENT_LABEL = re.compile(
    r"(?i)\s+(?:รหัสนักศึกษา|student(?:\s*ids?)?|studentid)\s*[:：]?\s*$"
)
_THAI_YEAR_LABEL = re.compile(r"(?im)^\s*(?:ปี|ป\])การศึกษา\s*[:：]?\s*(25\d{2})\s*$")
_ENGLISH_YEAR_LABEL = re.compile(r"(?im)^\s*academic\s+year\s*[:：]?\s*(\d{4})\s*$")
_THAI_ADVISOR = re.compile(r"(?im)^\s*อาจารย์ที่ปรึกษา\s*[:：]?\s*(\S.+?)\s*$")
_ENGLISH_ADVISOR = re.compile(r"(?im)^\s*advisor\s*[:：]?\s*(\S.+?)\s*$")
_DEPARTMENT = re.compile(r"(?i)^\s*(?:ภาควิชา|สาขาวิชา|department(?:\s+of)?)\s*[:：]?\s*(\S.+?)\s*$")
_FACULTY = re.compile(r"(?i)^\s*(?:คณะ|faculty(?:\s+of)?|school(?:\s+of)?)\s*[:：]?\s*(\S.+?)\s*$")
_INSTITUTION = re.compile(
    r"(?i)^\s*(?:มหาวิทยาลัย|สถาบัน(?=\s|[:：])(?:การศึกษา)?|ชื่อสถาบัน|"
    r"institution|university)\s*[:：]?\s*(\S.+?)\s*$"
)
_PROJECT_TYPE = re.compile(
    r"(?i)^\s*(?:ประเภทโครงงาน|ประเภทโครงการ|project\s+type)\s*[:：]?\s*(\S.+?)\s*$"
)
_METADATA_BOUNDARY = re.compile(
    r"(?i)\s+(?=(?:หัวข้อ(?:สหกิจศึกษา|โครงงาน(?:พิเศษ)?|ปัญหาพิเศษ)|"
    r"ชื่อนักศึกษา|นักศึกษา|รหัสนักศึกษา|ภาควิชา|สาขาวิชา|คณะ|ชื่อสถาบัน|"
    r"ปีการศึกษา|อาจารย์ที่ปรึกษา|ประเภทโครงงาน|"
    r"ประเภทโครงการ|title|students?|student\s+ids?|department|faculty|school|"
    r"academic\s+year|advisor|project\s+type)\s*[:：]?)"
)
_METADATA_TRAILER = re.compile(
    r"(?i)\s+(?:อนุมัติ|คณะกรรมการสอบ|ลายมือชื่อ|approved|exam(?:ination)?\s+committee)\b.*$"
)
_KEYWORD_START = re.compile(r"(?i)^\s*(?:keywords?|คำ\s*สำคัญ)\s*(?:(?:[:：])\s*(.*))?$")
_KEYWORD_END = re.compile(
    r"(?i)^\s*(?:abstract|บทคัดย่อ|title|students?|student\s*id|advisor|academic\s*year|"
    r"หัวข้อ(?:สหกิจศึกษา|โครงงาน)|ชื่อนักศึกษา|รหัสนักศึกษา|อาจารย์ที่ปรึกษา|ปีการศึกษา)\b"
)
_PAGE_MARKER = re.compile(r"^(?:[ก-ฮ]|\d+)$")
_NAME_LINE = re.compile(r"^[A-Za-z][A-Za-z .'-]{5,59}$")
_ENGLISH_BOILERPLATE = re.compile(
    r"(?i)\b(?:"
    r"FULFILLMENT\s+OF\s+THE\s+REQUIREMENT|"
    r"THE\s+REQUIREMENT\s+FOR|"
    r"THE\s+DEGREE\s+OF|DEGREE\s+OF|DEPARTMENT\s+OF|FACULTY\s+OF|"
    r"A\s+COOPERATIVE\s+EDUCATION|A\s+SPECIAL\s+PROJECT|"
    r"SUBMITTED\s+IN|ACADEMIC\s+YEAR|BACHELOR\s+OF|MASTER\s+OF|"
    r"INSTITUTE\s+OF\s+TECHNOLOGY|UNIVERSITY"
    r")\b"
)
_ENGLISH_METADATA = re.compile(
    r"(?i)^\s*(?:students?|student\s*ids?|advisor|degree|department|faculty|school|"
    r"institution|university|project\s+type|academic\s+year|abstract|keywords?)\b"
)
_THAI_METADATA = re.compile(
    r"^\s*(?:ชื่อนักศึกษา|นักศึกษา|รหัสนักศึกษา|ปริญญา|ภาควิชา|คณะ|มหาวิทยาลัย|"
    r"ปีการศึกษา|อาจารย์ที่ปรึกษา|บทคัดย่อ|คำ\s*สำคัญ)\b"
)
_THAI_NON_TITLE = re.compile(
    r"(?:สหกิจศึกษานี้เป็นส่วนหนึ่ง|ส่วนหนึ่งของการศึกษา|คณะกรรมการสอบ|ลิขสิทธิ์ของ|"
    r"หลักสูตรปริญญา)"
)
_THAI_NAME = re.compile(r"^(?:นาย|นางสาว|นาง)\s*\S+(?:\s+\S+){1,3}$")
_THAI_CONTINUATION_WORDS = ("และ", "หรือ", "เพื่อ", "โดย", "ด้วย", "ของ")
_PDF_LAYOUT_FILLER = re.compile(r"ǰ+")
_THAI_TITLE_GLYPH = re.compile(r"(?<=[\u0e00-\u0e7f])[%&](?=[\u0e00-\u0e7f])")
_MARKDOWN_LINE_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|[-+*]\s+)")


@dataclass(frozen=True, slots=True)
class MetadataEvidenceWarning:
    code: str
    message: str


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
    department: str | None = None
    faculty: str | None = None
    institution: str | None = None
    project_type: str | None = None
    warnings: tuple[MetadataEvidenceWarning, ...] = ()

    def project_metadata(self) -> dict[str, str | list[str] | None]:
        """Return the production metadata schema using deterministic evidence only."""
        title_is_thai = bool(
            self.title and any("ก" <= character <= "๛" for character in self.title)
        )
        return {
            "title_th": self.title if title_is_thai else None,
            "title_en": self.title_en or (self.title if self.title and not title_is_thai else None),
            "students": list(self.students),
            "student_ids": list(self.student_ids),
            "academic_year": self.academic_year,
            "advisor": self.advisor,
            "department": self.department,
            "faculty": self.faculty,
            "institution": self.institution,
            "project_type": self.project_type,
        }


@dataclass(frozen=True, slots=True)
class _SourcePage:
    number: int
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    value: str
    page_number: int
    line_number: int
    labeled: bool
    title_page: bool
    end_line: int
    metadata_page: bool = False


@dataclass(frozen=True, slots=True)
class _StudentGroup:
    names: tuple[str, ...]
    student_ids: tuple[str, ...]
    page_number: int
    line_number: int
    labeled: bool
    metadata_page: bool = False
    title_page: bool = False


@dataclass(frozen=True, slots=True)
class _MetadataCandidate:
    value: str
    page_number: int
    metadata_page: bool
    clean_boundary: bool


def _source_pages(source_text: str) -> tuple[_SourcePage, ...]:
    matches = list(_PAGE_LABEL.finditer(source_text))
    return tuple(
        _SourcePage(
            number=int(match.group("number")),
            role=match.group("role"),
            text="\n".join(
                _clean_markdown_layout(line)
                for line in source_text[
                    match.end() : matches[index + 1].start() if index + 1 < len(matches) else None
                ]
                .strip()
                .splitlines()
            ),
        )
        for index, match in enumerate(matches)
    )


def _candidate_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _clean_markdown_layout(line: str) -> str:
    """Remove OCR Markdown decoration without rewriting source wording."""

    value = _MARKDOWN_LINE_PREFIX.sub("", line).strip()
    if value.startswith(("**", "__")):
        marker = value[:2]
        value = value[2:]
        closing = value.find(marker)
        if closing >= 0:
            value = value[:closing] + value[closing + 2 :]
    return value.strip()


def _clean_title_layout(line: str) -> str:
    """Treat known PDF positioning glyphs as spacing, without repairing wording."""

    return " ".join(_PDF_LAYOUT_FILLER.sub(" ", line).split())


def _clean_thai_title_layout(line: str) -> str:
    """Repair a known embedded Thai mai-tho glyph loss in title evidence only."""

    return _THAI_TITLE_GLYPH.sub("้", line.strip())


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
                    inline = (match.group(1) or "").strip()
                    if inline:
                        active.append(inline)
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
                inline = (match.group(1) or "").strip()
                if inline:
                    active.append(inline)
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


def _english_title_line(
    line: str, *, labeled: bool = False, allow_mixed_case: bool = False
) -> bool:
    letters = [character for character in line if character.isascii() and character.isalpha()]
    if len(letters) < 8 or len(line.split()) < 2:
        return False
    if (
        _ENGLISH_BOILERPLATE.search(line)
        or _ENGLISH_METADATA.match(line)
        or _STUDENT_ID.search(line)
    ):
        return False
    if any("\u0e00" <= character <= "\u0e7f" for character in line):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
    capitalized = sum(word[0].isupper() for word in words)
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    sentence_like_title = (
        len(words) >= 5 and capitalized >= 1 and not line.rstrip().endswith((".", ":"))
    )
    return (
        labeled
        or uppercase_ratio >= 0.65
        or capitalized >= 2
        or allow_mixed_case
        and sentence_like_title
    )


def _english_title_continuation_line(
    line: str, *, labeled: bool = False, allow_mixed_case: bool = False
) -> bool:
    if _english_title_line(line, labeled=labeled, allow_mixed_case=allow_mixed_case):
        return True
    word = line.strip()
    return (
        bool(word)
        and " " not in word
        and len(word) >= 4
        and word.isascii()
        and word.isalpha()
        and word.isupper()
        and not _ENGLISH_BOILERPLATE.search(word)
    )


def _title_page(role: str) -> bool:
    return "title page" in role.casefold()


def _metadata_page(role: str) -> bool:
    return "project metadata" in role.casefold()


def _bounded_metadata_value(value: str) -> tuple[str, bool]:
    """Stop one labeled field before the next field or approval boilerplate."""

    original = " ".join(value.split())
    bounded = _METADATA_BOUNDARY.split(original, maxsplit=1)[0]
    bounded = _METADATA_TRAILER.sub("", bounded).strip(" :-–—\t")
    return bounded, bounded == original


def _metadata_candidates(
    pages: tuple[_SourcePage, ...], pattern: re.Pattern[str]
) -> list[_MetadataCandidate]:
    candidates: list[_MetadataCandidate] = []
    for page in pages:
        for line in page.text.splitlines():
            for segment in _METADATA_BOUNDARY.split(line):
                match = pattern.match(segment)
                if not match:
                    continue
                value, clean_boundary = _bounded_metadata_value(match.group(1))
                if pattern in (_DEPARTMENT, _FACULTY):
                    before_institution = re.split(
                        r"(?i)\s+(?=(?:มหาวิทยาลัย|สถาบัน|university\b|institution\b))",
                        value,
                        maxsplit=1,
                    )[0]
                    if before_institution != value:
                        value = before_institution
                        clean_boundary = False
                if value:
                    candidates.append(
                        _MetadataCandidate(
                            value=value,
                            page_number=page.number,
                            metadata_page=_metadata_page(page.role),
                            clean_boundary=clean_boundary,
                        )
                    )
    return candidates


def _best_metadata_value(candidates: list[_MetadataCandidate]) -> str | None:
    if not candidates:
        return None
    counts = Counter(_candidate_key(candidate.value) for candidate in candidates)
    return max(
        candidates,
        key=lambda candidate: (
            candidate.metadata_page,
            candidate.clean_boundary,
            counts[_candidate_key(candidate.value)],
            -candidate.page_number,
        ),
    ).value


def _probable_all_caps_student_line(line: str) -> bool:
    words = re.findall(r"[A-Z][A-Z'-]*", line)
    if not 2 <= len(words) <= 8 or " ".join(words) != " ".join(line.split()):
        return False
    title_connectors = {"AND", "FOR", "FROM", "IN", "OF", "ON", "THE", "TO", "USING", "WITH"}
    return not title_connectors.intersection(words)


def _joined_title_candidates(
    lines: list[str],
    start: int,
    *,
    max_lines: int = 4,
    labeled: bool = False,
    allow_mixed_case: bool = False,
) -> list[tuple[str, int]]:
    """Return each valid wrapped prefix so repetition can disambiguate its boundary."""

    joined: list[str] = []
    candidates: list[tuple[str, int]] = []
    for index in range(start, min(len(lines), start + max_lines)):
        line = _clean_title_layout(lines[index])
        if joined and _probable_all_caps_student_line(line):
            following = _clean_title_layout(lines[index + 1]) if index + 1 < len(lines) else ""
            joined_has_lowercase = any(character.islower() for character in " ".join(joined))
            if joined_has_lowercase or _probable_all_caps_student_line(following):
                break
        if not line or not _english_title_continuation_line(
            line, labeled=labeled, allow_mixed_case=allow_mixed_case
        ):
            break
        joined.append(line)
        candidate = " ".join(joined)
        if len(candidate.split()) >= (1 if labeled else 3):
            candidates.append((candidate, index + 1))
    return candidates


def _english_title_candidates(page: _SourcePage) -> list[_Candidate]:
    lines = [_clean_title_layout(line) for line in page.text.splitlines()]
    candidates: list[_Candidate] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        label = _ENGLISH_TITLE_LABEL.match(line)
        if (
            label
            and _title_page(page.role)
            and not re.match(r"(?i)^\s*(?:english\s+title|project\s+title|title)\s*[:：]", line)
        ):
            label = None
        if label:
            payload, _ = _bounded_metadata_value(label.group(1))
            virtual_lines = ([payload] if payload else []) + lines[index + 1 :]
            if not virtual_lines:
                continue
            for value, consumed in _joined_title_candidates(virtual_lines, 0, labeled=True):
                end_line = index + consumed + (0 if payload else 1)
                candidates.append(
                    _Candidate(
                        value=value,
                        page_number=page.number,
                        line_number=index,
                        labeled=True,
                        title_page=_title_page(page.role),
                        end_line=end_line,
                        metadata_page=_metadata_page(page.role),
                    )
                )
            continue
        allow_mixed_case = _title_page(page.role)
        if index > 12 or not _english_title_line(line, allow_mixed_case=allow_mixed_case):
            continue
        mixed_case_only = not _english_title_line(line)
        if (
            index
            and lines[index - 1]
            and _english_title_continuation_line(
                lines[index - 1], allow_mixed_case=allow_mixed_case
            )
        ):
            continue
        for value, end_line in _joined_title_candidates(
            lines, index, allow_mixed_case=allow_mixed_case
        ):
            bilingual_context = any(_readable_thai_line(value) for value in lines[:index])
            if mixed_case_only and end_line == index + 1 and not bilingual_context:
                continue
            candidates.append(
                _Candidate(
                    value=value,
                    page_number=page.number,
                    line_number=index,
                    labeled=False,
                    title_page=_title_page(page.role),
                    end_line=end_line,
                    metadata_page=_metadata_page(page.role),
                )
            )
    return candidates


def _readable_thai_line(line: str) -> bool:
    thai_count = sum("\u0e00" <= character <= "\u0e7f" for character in line)
    if thai_count < 10 or thai_count / max(len(line.strip()), 1) < 0.55:
        return False
    if (
        _EMBEDDED_DIGIT.search(line)
        or _THAI_METADATA.match(line)
        or _THAI_NON_TITLE.search(line)
        or _THAI_NAME.fullmatch(line)
    ):
        return False
    return not any("\u0100" <= character <= "\u02ff" for character in line)


def _join_thai_title_lines(lines: list[str]) -> str:
    joined = lines[0]
    for line in lines[1:]:
        separator = ""
        if not ("\u0e00" <= joined[-1] <= "\u0e7f" and "\u0e00" <= line[0] <= "\u0e7f"):
            separator = " "
        joined = f"{joined}{separator}{line}"
    return joined


def _thai_title_candidates(page: _SourcePage) -> list[_Candidate]:
    lines = [_clean_thai_title_layout(line) for line in page.text.splitlines()]
    candidates: list[_Candidate] = []
    for index, line in enumerate(lines):
        label = _THAI_TITLE_LABEL.match(line)
        if label:
            first, _ = _bounded_metadata_value(label.group(1))
            if not first or not _readable_thai_line(first):
                continue
            title_lines = [first]
            end_line = index + 1
            for continuation in lines[index + 1 : index + 4]:
                if (
                    not continuation
                    or _unprefixed_thai_student_name(continuation)
                    or not _readable_thai_line(continuation)
                ):
                    break
                previous = title_lines[-1].rstrip()
                if not (
                    previous.endswith(_THAI_CONTINUATION_WORDS)
                    or continuation.startswith(_THAI_CONTINUATION_WORDS)
                    or len(previous) >= 35
                ):
                    break
                title_lines.append(continuation)
                end_line += 1
            candidates.append(
                _Candidate(
                    value=_join_thai_title_lines(title_lines),
                    page_number=page.number,
                    line_number=index,
                    labeled=True,
                    title_page=_title_page(page.role),
                    end_line=end_line,
                    metadata_page=_metadata_page(page.role),
                )
            )
            continue

        if (
            not _title_page(page.role)
            or index > 8
            or not _readable_thai_line(line)
            or line.startswith(_THAI_CONTINUATION_WORDS)
            or (
                len(line) < 40
                and len(line.split()) in (2, 3)
                and not any(character.isascii() and character.isalpha() for character in line)
            )
        ):
            continue
        title_lines = [line]
        end_line = index + 1
        for continuation in lines[index + 1 : index + 4]:
            if (
                not continuation
                or _unprefixed_thai_student_name(continuation)
                or not _readable_thai_line(continuation)
            ):
                break
            previous = title_lines[-1].rstrip()
            if not (
                previous.endswith(_THAI_CONTINUATION_WORDS)
                or continuation.startswith(_THAI_CONTINUATION_WORDS)
                or len(previous) >= 35
            ):
                break
            title_lines.append(continuation)
            end_line += 1
        candidates.append(
            _Candidate(
                value=_join_thai_title_lines(title_lines),
                page_number=page.number,
                line_number=index,
                labeled=False,
                title_page=True,
                end_line=end_line,
                metadata_page=_metadata_page(page.role),
            )
        )
    return candidates


def _best_candidate(candidates: list[_Candidate]) -> _Candidate | None:
    """Rank repeated, labeled, early-page evidence while preserving source text."""

    if not candidates:
        return None
    counts = Counter(_candidate_key(candidate.value) for candidate in candidates)
    return max(
        candidates,
        key=lambda candidate: (
            candidate.metadata_page,
            candidate.labeled,
            counts[_candidate_key(candidate.value)],
            candidate.title_page,
            -candidate.page_number,
            -candidate.line_number,
            len(candidate.value.split()),
        ),
    )


def find_reliable_repeated_english_title(source_text: str) -> str | None:
    """Return one complete English title repeated on distinct focused pages."""

    candidates = [
        candidate
        for page in _source_pages(source_text)
        for candidate in _english_title_candidates(page)
    ]
    pages_by_key: dict[str, set[int]] = {}
    for candidate in candidates:
        pages_by_key.setdefault(_candidate_key(candidate.value), set()).add(candidate.page_number)
    repeated = {key for key, pages in pages_by_key.items() if len(pages) >= 2}
    maximal = {
        key for key in repeated if not any(key != other and key in other for other in repeated)
    }
    if len(maximal) != 1:
        return None
    selected_key = maximal.pop()
    return next(
        candidate.value
        for candidate in candidates
        if _candidate_key(candidate.value) == selected_key
    )


def _clean_student_name(line: str) -> tuple[str | None, tuple[str, ...]]:
    value = line.strip().strip("-•")
    label = _STUDENT_LABEL.match(value)
    if label:
        value = label.group(1).strip()
    identifier_label = _STUDENT_ID_LABEL.match(value)
    if identifier_label:
        return None, tuple(_ANY_STUDENT_ID.findall(identifier_label.group(1)))

    identifiers = tuple(_ANY_STUDENT_ID.findall(value))
    value = _ANY_STUDENT_ID.sub("", value).strip(" :-–—\t")
    value = _ATTACHED_STUDENT_LABEL.sub("", value).strip(" :-–—\t")
    if not value:
        return None, identifiers
    if _THAI_NAME.fullmatch(value):
        return " ".join(value.split()), identifiers
    if not _NAME_LINE.fullmatch(value) or len(value.split()) not in (2, 3, 4):
        return None, identifiers
    if _ENGLISH_BOILERPLATE.search(value) or _ENGLISH_METADATA.match(value):
        return None, identifiers
    return " ".join(value.split()), identifiers


def _student_block_end(lines: list[str], start: int) -> int:
    end = start + 1
    while end < len(lines) and end <= start + 8:
        line = lines[end].strip()
        if line and (
            _THAI_TITLE_LABEL.match(line)
            or _ENGLISH_TITLE_LABEL.match(line)
            or _THAI_ADVISOR.match(line)
            or _ENGLISH_ADVISOR.match(line)
            or _THAI_YEAR_LABEL.match(line)
            or _ENGLISH_YEAR_LABEL.match(line)
            or _THAI_METADATA.match(line)
            and not (_STUDENT_LABEL.match(line) or _STUDENT_ID_LABEL.match(line))
            or _ENGLISH_METADATA.match(line)
            and not (_STUDENT_LABEL.match(line) or _STUDENT_ID_LABEL.match(line))
        ):
            break
        end += 1
    return end


def _labeled_student_groups(page: _SourcePage) -> list[_StudentGroup]:
    lines = page.text.splitlines()
    groups: list[_StudentGroup] = []
    for index, line in enumerate(lines):
        if not _STUDENT_LABEL.match(line.strip()):
            continue
        names: list[str] = []
        identifiers: list[str] = []
        for student_line in lines[index : _student_block_end(lines, index)]:
            name, found_ids = _clean_student_name(student_line)
            if name and name not in names:
                names.append(name)
            identifiers.extend(value for value in found_ids if value not in identifiers)
        if names:
            groups.append(
                _StudentGroup(
                    names=tuple(names),
                    student_ids=tuple(identifiers),
                    page_number=page.number,
                    line_number=index,
                    labeled=True,
                    metadata_page=_metadata_page(page.role),
                    title_page=_title_page(page.role),
                )
            )
    return groups


def _unlabeled_student_groups(
    page: _SourcePage, title_candidates: list[_Candidate]
) -> list[_StudentGroup]:
    lines = page.text.splitlines()
    groups: list[_StudentGroup] = []
    for title in title_candidates:
        names: list[str] = []
        identifiers: list[str] = []
        started = False
        for line in lines[title.end_line : title.end_line + 7]:
            if not line.strip():
                if started:
                    break
                continue
            name, found_ids = _clean_student_name(line)
            if name is None:
                if started or _ENGLISH_BOILERPLATE.search(line) or _THAI_METADATA.match(line):
                    break
                continue
            started = True
            names.append(name)
            identifiers.extend(value for value in found_ids if value not in identifiers)
        if names:
            groups.append(
                _StudentGroup(
                    names=tuple(names),
                    student_ids=tuple(identifiers),
                    page_number=page.number,
                    line_number=title.end_line,
                    labeled=False,
                    metadata_page=_metadata_page(page.role),
                    title_page=_title_page(page.role),
                )
            )

    for index, line in enumerate(lines[:15]):
        name, identifiers = _clean_student_name(line)
        if name and _THAI_NAME.fullmatch(name):
            groups.append(
                _StudentGroup(
                    (name,),
                    identifiers,
                    page.number,
                    index,
                    False,
                    _metadata_page(page.role),
                    _title_page(page.role),
                )
            )
    return groups


def _student_name_key(name: str) -> str:
    key = _candidate_key(name)
    key = re.sub(r"^(?:นาย|นางสาว|นาง)\s*", "", key)
    return re.sub(r"^(?:mr|mrs|miss|ms)\.?\s+", "", key)


def _unprefixed_thai_student_name(line: str) -> str | None:
    value = " ".join(line.strip().split())
    if (
        not re.fullmatch(r"[ก-๛.]+(?:\s+[ก-๛.]+){1,2}", value)
        or _THAI_METADATA.match(value)
        or _THAI_NON_TITLE.search(value)
    ):
        return None
    return value


def _diagnostic_title_student_groups(
    pages: tuple[_SourcePage, ...],
) -> list[_StudentGroup]:
    """Read conservative unlabelled title-page names for conflict diagnostics only."""

    groups: list[_StudentGroup] = []
    for page in pages:
        if not _title_page(page.role):
            continue
        runs: list[tuple[int, list[str]]] = []
        active: list[str] = []
        active_start = 0
        for index, line in enumerate(page.text.splitlines()[:15]):
            name = _unprefixed_thai_student_name(line)
            if name:
                if not active:
                    active_start = index
                if name not in active:
                    active.append(name)
            elif active:
                runs.append((active_start, active))
                active = []
        if active:
            runs.append((active_start, active))
        if runs:
            start, names = max(runs, key=lambda item: (len(item[1]), -item[0]))
            groups.append(
                _StudentGroup(
                    names=tuple(names),
                    student_ids=(),
                    page_number=page.number,
                    line_number=start,
                    labeled=False,
                    title_page=True,
                )
            )
    return groups


def _best_student_group(groups: list[_StudentGroup]) -> _StudentGroup | None:
    if not groups:
        return None
    name_counts = Counter(_student_name_key(name) for group in groups for name in group.names)
    return max(
        groups,
        key=lambda group: (
            group.metadata_page,
            bool(group.student_ids),
            group.labeled,
            sum(name_counts[_student_name_key(name)] for name in group.names),
            -group.page_number,
            -group.line_number,
        ),
    )


def _student_conflict_warning(
    groups: list[_StudentGroup],
) -> MetadataEvidenceWarning | None:
    metadata = _best_student_group([group for group in groups if group.metadata_page])
    title_groups = [group for group in groups if group.title_page]
    if metadata is not None:
        metadata_is_thai = any("ก" <= char <= "๛" for char in "".join(metadata.names))
        comparable = [
            group
            for group in title_groups
            if any("ก" <= char <= "๛" for char in "".join(group.names)) == metadata_is_thai
        ]
        title_groups = comparable or title_groups
    if metadata is not None and title_groups:
        metadata_keys = {_student_name_key(name) for name in metadata.names}
        title = max(
            title_groups,
            key=lambda group: (
                sum(_student_name_key(name) in metadata_keys for name in group.names),
                len(group.names),
                -group.page_number,
            ),
        )
    else:
        title = _best_student_group(title_groups)
    if metadata is None or title is None:
        return None

    metadata_names = {_student_name_key(name) for name in metadata.names}
    title_names = {_student_name_key(name) for name in title.names}
    if metadata.student_ids and title.student_ids:
        conflict = set(metadata.student_ids) != set(title.student_ids)
    else:
        same_script = any("ก" <= char <= "๛" for char in "".join(metadata_names)) == any(
            "ก" <= char <= "๛" for char in "".join(title_names)
        )
        conflict = len(title.names) != len(metadata.names) or (
            same_script and title_names != metadata_names
        )
    if not conflict:
        return None
    title_only = [name for name in title.names if _student_name_key(name) not in metadata_names]
    details = f" Title-page-only entries: {', '.join(title_only)}." if title_only else ""
    return MetadataEvidenceWarning(
        code="student_metadata_conflict",
        message=(
            f"Student metadata conflict: title page lists {len(title.names)} student(s), "
            f"explicit metadata lists {len(metadata.names)}; explicit metadata records "
            f"were retained.{details}"
        ),
    )


def collect_wiki_source_evidence(source_text: str) -> WikiSourceEvidence:
    """Select exact readable title text and explicit source metadata when possible."""

    source_pages = _source_pages(source_text)
    if not source_pages:
        return WikiSourceEvidence(None, None, (), (), None, None, ())

    # Focused source may contain multiple excerpts from the same physical page.
    # Preserve every excerpt instead of overwriting abstract keywords with metadata.
    pages: dict[int, str] = {}
    for page in source_pages:
        pages[page.number] = "\n\n".join(filter(None, (pages.get(page.number), page.text)))
    english_by_page = {page.number: _english_title_candidates(page) for page in source_pages}
    english_candidates = [
        candidate for candidates in english_by_page.values() for candidate in candidates
    ]
    thai_candidates = [
        candidate for page in source_pages for candidate in _thai_title_candidates(page)
    ]
    best_english = _best_candidate(english_candidates)
    best_thai = _best_candidate(thai_candidates)
    title_en = best_english.value if best_english else None
    title_th = best_thai.value if best_thai else None
    title = title_th or title_en

    student_groups = [group for page in source_pages for group in _labeled_student_groups(page)]
    student_groups.extend(
        group
        for page in source_pages
        if _title_page(page.role)
        for group in _unlabeled_student_groups(
            page,
            [
                candidate
                for candidate in english_by_page[page.number]
                if best_english
                and _candidate_key(candidate.value) == _candidate_key(best_english.value)
            ],
        )
    )
    diagnostic_student_groups = [
        *student_groups,
        *_diagnostic_title_student_groups(source_pages),
    ]
    best_students = _best_student_group(student_groups)
    student_names = best_students.names if best_students else ()
    conflict_warning = _student_conflict_warning(diagnostic_student_groups)

    plain_source = _PAGE_LABEL.sub("", source_text)
    explicit_student_ids = tuple(dict.fromkeys(_STUDENT_ID.findall(plain_source)))
    student_ids = (
        best_students.student_ids
        if best_students and best_students.student_ids
        else explicit_student_ids
    )
    year_candidates = _metadata_candidates(source_pages, _THAI_YEAR_LABEL)
    year_candidates.extend(_metadata_candidates(source_pages, _ENGLISH_YEAR_LABEL))
    academic_year = _best_metadata_value(year_candidates)
    if academic_year is None:
        academic_year = _prefer_repeated_candidate(_ACADEMIC_YEAR.findall(plain_source))

    advisor_candidates = _metadata_candidates(source_pages, _THAI_ADVISOR)
    advisor_candidates.extend(_metadata_candidates(source_pages, _ENGLISH_ADVISOR))
    return WikiSourceEvidence(
        title=title,
        title_en=title_en,
        students=student_names,
        student_ids=student_ids,
        advisor=_best_metadata_value(advisor_candidates),
        academic_year=academic_year,
        keywords=_extract_keywords(pages),
        department=_best_metadata_value(_metadata_candidates(source_pages, _DEPARTMENT)),
        faculty=_best_metadata_value(_metadata_candidates(source_pages, _FACULTY)),
        institution=_best_metadata_value(_metadata_candidates(source_pages, _INSTITUTION)),
        project_type=_best_metadata_value(_metadata_candidates(source_pages, _PROJECT_TYPE)),
        warnings=(conflict_warning,) if conflict_warning is not None else (),
    )
