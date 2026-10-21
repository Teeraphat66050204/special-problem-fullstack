"""Run PDF -> focused source -> project metadata/summary locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.prompts.project_summary import validate_project_summary
from app.services.llm_service import (
    InvalidProjectSummaryError,
    LLMServiceError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.llm_service import (
    generate_project_summary_result as generate_wiki,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.wiki_evidence import collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source


def run(pdf_path: Path) -> int:
    """Print selected source and combined project output; return a shell exit code."""

    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    try:
        with pdf_path.open("rb") as pdf_file:
            extraction = extract_pdf(pdf_file)
    except (OSError, PdfExtractionError) as exc:
        print(f"PDF extraction failed: {exc}", file=sys.stderr)
        return 1

    try:
        selection = prepare_wiki_source(extraction)
    except ValueError as exc:
        print(f"No suitable Wiki source: {exc}", file=sys.stderr)
        return 1

    if not selection.source_text.strip():
        print("No suitable Wiki source: selected text is empty", file=sys.stderr)
        return 1

    print(f"Selected Wiki source (pages {', '.join(map(str, selection.selected_pages))}):")
    print(selection.summary_source_text)
    print()

    try:
        generation = generate_wiki(selection.summary_source_text)
    except OllamaUnavailableError as exc:
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        return 1
    except OllamaTimeoutError as exc:
        print(f"Ollama timeout: {exc}", file=sys.stderr)
        return 1
    except InvalidProjectSummaryError as exc:
        print("Invalid project summary returned by Ollama:", file=sys.stderr)
        print(", ".join(exc.issues), file=sys.stderr)
        return 1
    except LLMServiceError as exc:
        print(f"Wiki generation failed: {exc}", file=sys.stderr)
        return 1

    summary = generation if isinstance(generation, str) else generation.generated_summary
    evidence = collect_wiki_source_evidence(selection.source_text)

    print("Project metadata:")
    print(json.dumps(evidence.project_metadata(), ensure_ascii=False, indent=2))
    print()
    print("Generated Thai project summary:")
    print(summary)
    print()

    validation = validate_project_summary(summary)
    if not validation.is_valid:
        print("Invalid project summary:", file=sys.stderr)
        print(", ".join(validation.issues), file=sys.stderr)
        return 1

    print("Project summary: valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate project metadata and a Thai summary from focused PDF source"
    )
    parser.add_argument("pdf", type=Path, help="Path to a digital PDF")
    arguments = parser.parse_args(argv)
    return run(arguments.pdf)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
