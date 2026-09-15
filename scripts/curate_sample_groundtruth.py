"""Populate the small JSON records from their supplied front-matter TXT text.

This conservative import fills missing fields only; manual corrections survive
later runs. Fields absent from the TXT stay null or empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.datasets.groundtruth import DatasetIndex, GroundTruthRecord, load_groundtruth

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = DATA_ROOT / "groundtruth"
_THAI = re.compile(r"[\u0e00-\u0e7f]")
_YEAR = re.compile(r"25\d{2}")
_STUDENT_ID = re.compile(r"(?:รหัสนักศึกษา\s*)?\d{8}$")


def _section(lines: list[str], start_label: str, end_label: str) -> list[str]:
    try:
        start = lines.index(start_label) + 1
        end = lines.index(end_label, start)
    except ValueError:
        return []
    return lines[start:end]


def derive_fields(raw_text: str) -> dict:
    """Read only labels visibly present in the supplied Thai ground-truth text."""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    try:
        title_end = lines.index("ชื่อนักศึกษา")
    except ValueError:
        title_end = 0
    title_lines = lines[2:title_end] if title_end > 2 else []
    thai_title = " ".join(line for line in title_lines if _THAI.search(line)) or None
    english_title = (
        " ".join(
            line for line in title_lines if not _THAI.search(line) and re.search(r"[A-Za-z]", line)
        )
        or None
    )

    students = [_STUDENT_ID.sub("", line).strip() for line in _section(lines, "ชื่อนักศึกษา", "ปริญญา")]
    students = [student for student in students if student]

    year_lines = _section(lines, "ปีการศึกษา", "อาจารย์ที่ปรึกษา")
    year_match = _YEAR.search(" ".join(year_lines))
    advisor_lines = _section(lines, "อาจารย์ที่ปรึกษา", "บทคัดย่อ")

    abstract_lines = raw_text.splitlines()
    abstract_start = next(
        (index for index, line in enumerate(abstract_lines) if line.strip() == "บทคัดย่อ"),
        None,
    )
    keyword_start = next(
        (index for index, line in enumerate(abstract_lines) if line.strip().startswith("คำสำคัญ")),
        None,
    )
    if abstract_start is None:
        abstract_th = None
    else:
        end = (
            keyword_start if keyword_start is not None and keyword_start > abstract_start else None
        )
        abstract_th = "\n".join(abstract_lines[abstract_start + 1 : end]).strip() or None

    keywords: list[str] = []
    if keyword_start is not None:
        keyword_line = abstract_lines[keyword_start].strip()
        keyword_value = re.sub(r"^คำสำคัญ\s*[:：]?\s*", "", keyword_line).strip()
        keywords = [part.strip() for part in keyword_value.split(",") if part.strip()]

    return {
        "title_th": thai_title,
        "title_en": english_title,
        "students": students,
        "advisor": " ".join(advisor_lines) or None,
        "academic_year": int(year_match.group()) if year_match else None,
        "abstract_th": abstract_th,
        "abstract_en": None,
        "keywords": keywords,
    }


def enrich(record: GroundTruthRecord) -> GroundTruthRecord:
    """Fill absent fields without replacing later manual curation."""

    if not record.raw_text:
        return record
    merged = record.model_dump()
    for field, value in derive_fields(record.raw_text).items():
        if merged[field] in (None, [], ""):
            merged[field] = value
    return GroundTruthRecord.model_validate(merged)


def main() -> None:
    count = 0
    entries: list[dict[str, str]] = []
    for path in sorted(DATA_DIR.glob("document_*.json")):
        record = enrich(load_groundtruth(path))
        path.write_text(
            json.dumps(record.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entries.append(
            {
                "document_id": record.document_id,
                "pdf": f"sample/{record.document_id}.pdf",
                "groundtruth": f"groundtruth/{path.name}",
            }
        )
        count += 1
    index = DatasetIndex.model_validate({"schema_version": 1, "documents": entries})
    (DATA_ROOT / "index.json").write_text(
        json.dumps(index.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATA_ROOT / "groundtruth.schema.json").write_text(
        json.dumps(GroundTruthRecord.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Enriched {count} ground-truth records")


if __name__ == "__main__":
    main()
