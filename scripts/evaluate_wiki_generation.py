"""Evaluate focused PDF-to-Wiki generation against indexed local GroundTruth."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.datasets.groundtruth import DatasetEntry, load_dataset_index, load_groundtruth
from app.evaluation.wiki_generation import aggregate_results, evaluate_markdown
from app.prompts import WIKI_GENERATION_INSTRUCTIONS
from app.services.llm_service import (
    InvalidWikiMarkdownError,
    InvalidWikiOutputError,
    LLMServiceError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_wiki_result,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.wiki_source import prepare_wiki_source


def _failure(result: dict[str, Any], category: str, error: Exception | str) -> dict[str, Any]:
    result.update(
        status="failure",
        failure_category=category,
        error=str(error),
        notes=[f"{category}: {error}"],
        issue_categories=[],
    )
    return result


def evaluate_entry(data_dir: Path, entry: DatasetEntry) -> dict[str, Any]:
    """Evaluate one indexed pair without stopping the surrounding batch."""

    result: dict[str, Any] = {
        "document_id": entry.document_id,
        "status": "failure",
        "selected_pages": [],
        "source_character_count": 0,
        "structure_valid": None,
        "markdown": None,
        "raw_model_markdown": None,
        "refinement_changes": [],
    }
    pdf_path = data_dir / entry.pdf
    groundtruth_path = data_dir / entry.groundtruth
    if not pdf_path.is_file():
        return _failure(result, "missing_pdf", pdf_path)
    if not groundtruth_path.is_file():
        return _failure(result, "missing_groundtruth", groundtruth_path)

    try:
        groundtruth = load_groundtruth(groundtruth_path)
        if groundtruth.document_id != entry.document_id:
            raise ValueError("GroundTruth document ID does not match the dataset index")
    except (OSError, ValueError) as exc:
        return _failure(result, "invalid_groundtruth", exc)

    try:
        with pdf_path.open("rb") as pdf_file:
            extraction = extract_pdf(pdf_file)
    except (OSError, PdfExtractionError) as exc:
        return _failure(result, "extraction_failed", exc)

    try:
        selection = prepare_wiki_source(extraction)
        if not selection.source_text.strip():
            raise ValueError("Focused Wiki source is empty")
    except ValueError as exc:
        return _failure(result, "no_wiki_source", exc)

    result["selected_pages"] = list(selection.selected_pages)
    result["source_character_count"] = len(selection.source_text)
    try:
        # GroundTruth and full_text are never passed to the generator.
        generation = generate_wiki_result(selection.source_text)
    except InvalidWikiOutputError as exc:
        result["raw_model_markdown"] = exc.raw_markdown
        raw_checks = evaluate_markdown(selection.source_text, exc.raw_markdown, groundtruth)
        result["raw_model_checks"] = {
            key: value
            for key, value in raw_checks.items()
            if key not in ("notes", "issue_categories")
        }
        result["raw_model_notes"] = raw_checks["notes"]
        result["raw_model_issue_categories"] = raw_checks["issue_categories"]
        return _failure(result, "invalid_wiki_output", exc)
    except InvalidWikiMarkdownError as exc:
        result["structure_valid"] = False
        result["structure_issues"] = [issue.code for issue in exc.issues]
        return _failure(result, "invalid_markdown", exc)
    except OllamaUnavailableError as exc:
        return _failure(result, "ollama_unavailable", exc)
    except OllamaTimeoutError as exc:
        return _failure(result, "ollama_timeout", exc)
    except OllamaModelNotFoundError as exc:
        return _failure(result, "ollama_model_not_found", exc)
    except LLMServiceError as exc:
        return _failure(result, "llm_failure", exc)
    except Exception as exc:
        return _failure(result, "unexpected_generation_failure", exc)

    try:
        markdown = generation.markdown
        checks = evaluate_markdown(selection.source_text, markdown, groundtruth)
        raw_checks = evaluate_markdown(selection.source_text, generation.raw_markdown, groundtruth)
    except Exception as exc:
        return _failure(result, "evaluation_failed", exc)

    result.update(
        status="success",
        markdown=markdown,
        raw_model_markdown=generation.raw_markdown,
        refinement_changes=list(generation.refinement_changes),
        structure_valid=checks["structure"]["valid"],
        checks={
            key: value for key, value in checks.items() if key not in ("notes", "issue_categories")
        },
        notes=checks["notes"],
        issue_categories=checks["issue_categories"],
        raw_model_checks={
            key: value
            for key, value in raw_checks.items()
            if key not in ("notes", "issue_categories")
        },
        raw_model_notes=raw_checks["notes"],
        raw_model_issue_categories=raw_checks["issue_categories"],
    )
    return result


def evaluate_dataset(
    data_dir: Path, *, document_ids: set[str] | None = None, limit: int | None = None
) -> dict[str, Any]:
    """Process every selected indexed pair, including pairs with missing files."""

    index = load_dataset_index(data_dir / "index.json")
    entries = [
        entry
        for entry in index.documents
        if document_ids is None or entry.document_id in document_ids
    ]
    if limit is not None:
        entries = entries[:limit]
    results: list[dict[str, Any]] = []
    for entry in entries:
        try:
            results.append(evaluate_entry(data_dir, entry))
        except Exception as exc:
            results.append(
                _failure(
                    {"document_id": entry.document_id, "structure_valid": None},
                    "unexpected_document_failure",
                    exc,
                )
            )
    settings = get_settings()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.ollama_model,
        "timeout_seconds": settings.ollama_timeout_seconds,
        "temperature": settings.ollama_temperature,
        "prompt_sha256": hashlib.sha256(WIKI_GENERATION_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "summary": aggregate_results(results),
        "documents": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    rate = summary["structure_pass_rate"]
    rate_text = (
        "not assessed"
        if rate is None
        else f"{rate:.0%} ({summary['structure_passes']}/{summary['structure_assessed']})"
    )
    print(f"Documents: {summary['total_documents']}")
    print(
        f"Generation: {summary['generation_successes']} success(es), "
        f"{summary['generation_failures']} failure(s)"
    )
    print(f"Structure pass rate: {rate_text}")
    print(f"Common failure/quality categories: {summary['common_failure_categories']}")
    print(f"Raw model quality categories: {summary['raw_model_quality_categories']}")
    for item in summary["per_document_notes"]:
        print(f"{item['document_id']} ({item['status']}):")
        if not item["notes"]:
            print("  No automated issues; review factual prose manually")
        for note in item["notes"]:
            print(f"  - {note}")
        if item["refinement_changes"]:
            print(f"  Source-backed refinements: {', '.join(item['refinement_changes'])}")
        if item["raw_model_notes"]:
            print(f"  Raw model review: {', '.join(item['raw_model_notes'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate local sample PDFs against GroundTruth")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ids", nargs="+", help="Optional document IDs from data/index.json")
    parser.add_argument("--limit", type=int, help="Maximum indexed documents to process")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 JSON report path")
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")

    try:
        report = evaluate_dataset(
            arguments.data_dir,
            document_ids=set(arguments.ids) if arguments.ids else None,
            limit=arguments.limit,
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot load dataset index: {exc}", file=sys.stderr)
        return 1
    if report["summary"]["total_documents"] == 0:
        print("No indexed documents matched the selection", file=sys.stderr)
        return 1

    _print_summary(report)
    if arguments.output:
        target = arguments.output.resolve()
        protected_dir = (arguments.data_dir / "groundtruth").resolve()
        protected_index = (arguments.data_dir / "index.json").resolve()
        if target == protected_index or target.is_relative_to(protected_dir):
            print(
                "Report path must not overwrite the dataset index or GroundTruth", file=sys.stderr
            )
            return 1
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"Cannot save evaluation report: {exc}", file=sys.stderr)
            return 1
        print(f"JSON report: {target}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
