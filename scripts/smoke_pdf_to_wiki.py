"""Run the existing PDF -> focused Wiki source -> Ollama pipeline locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.prompts import WikiStructureIssue, validate_wiki_markdown
from app.services.llm_service import (
    InvalidWikiMarkdownError,
    LLMServiceError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_wiki,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.wiki_source import prepare_wiki_source


def _print_validation_issues(issues: tuple[WikiStructureIssue, ...]) -> None:
    for issue in issues:
        print(f"  - {issue.message}", file=sys.stderr)


def run(pdf_path: Path) -> int:
    """Print the selected source and generated Markdown; return a shell exit code."""

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
    print(selection.source_text)
    print()

    try:
        markdown = generate_wiki(selection.source_text)
    except OllamaUnavailableError as exc:
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        return 1
    except OllamaTimeoutError as exc:
        print(f"Ollama timeout: {exc}", file=sys.stderr)
        return 1
    except InvalidWikiMarkdownError as exc:
        print("Invalid Wiki Markdown returned by Ollama:", file=sys.stderr)
        _print_validation_issues(exc.issues)
        return 1
    except LLMServiceError as exc:
        print(f"Wiki generation failed: {exc}", file=sys.stderr)
        return 1

    print("Generated Wiki Markdown:")
    print(markdown)
    print()

    validation = validate_wiki_markdown(markdown)
    if not validation.is_valid:
        print("Invalid Wiki Markdown:", file=sys.stderr)
        _print_validation_issues(validation.issues)
        return 1

    print("Wiki Markdown structure: valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Wiki Markdown from a focused PDF source")
    parser.add_argument("pdf", type=Path, help="Path to a digital PDF")
    arguments = parser.parse_args(argv)
    return run(arguments.pdf)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
