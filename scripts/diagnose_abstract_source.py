"""Source diagnostics; optional bounded chapter OCR, never abstract/LLM generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.config import Settings
from app.services.chapter_ocr import recover_chapter_one
from app.services.pdf_extractor import extract_pdf
from app.services.wiki_source import prepare_wiki_source


def diagnose(
    directory: Path, *, chapter_ocr: bool = False, settings: Settings | None = None
) -> list[dict]:
    results = []
    for path in sorted(directory.glob("*.pdf")):
        extraction = extract_pdf(path)
        recovery = recover_chapter_one(
            path, extraction, settings or Settings(), allow_ocr=chapter_ocr
        )
        extraction = recovery.extraction
        row = {"document": path.name, "page_count": extraction.page_count}
        row["chapter_ocr"] = asdict(recovery.diagnostics)
        try:
            selection = prepare_wiki_source(extraction)
            row.update(asdict(selection.detection))
            row.update(
                selected_pages=selection.selected_pages,
                source_characters=len(selection.source_text),
            )
        except ValueError as exc:
            row["error"] = str(exc)
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("data/sample"))
    parser.add_argument(
        "--chapter-ocr", action="store_true", help="Enable bounded Typhoon chapter OCR"
    )
    parser.add_argument("--output", type=Path, help="Write a NEW JSON report (never overwrite)")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior diagnostics")
    result = json.dumps(
        diagnose(args.directory, chapter_ocr=args.chapter_ocr), ensure_ascii=False, indent=2
    )
    if args.output:
        with args.output.open("x", encoding="utf-8") as report:
            report.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
