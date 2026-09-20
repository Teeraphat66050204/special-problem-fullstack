"""Reproducibly benchmark Ollama models on the indexed PDF-to-Wiki pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TextIO

from app.config import Settings, get_settings
from app.datasets.groundtruth import (
    DatasetEntry,
    GroundTruthRecord,
    load_dataset_index,
    load_groundtruth,
)
from app.evaluation.wiki_generation import evaluate_markdown
from app.prompts import REQUIRED_WIKI_HEADINGS, build_wiki_generation_prompt
from app.services.llm_service import LLMServiceError, OllamaClient, OllamaGenerationResult
from app.services.ocr_service import (
    OcrServiceError,
    add_ocr_failure_warning,
    apply_ocr_fallback,
    load_ocr_provider,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.wiki_evidence import WikiSourceEvidence, collect_wiki_source_evidence
from app.services.wiki_output import WikiOutputError, refine_wiki_markdown
from app.services.wiki_source import prepare_wiki_source
from benchmark_telemetry import ResourceTelemetrySampler

DEFAULT_MODELS = ("qwen3:8b", "qwen3:14b")
SCALAR_FIELDS = ("title", "english_title", "advisor", "academic_year")
MULTI_VALUE_FIELDS = ("students", "student_ids", "keywords")
_GROUND_TRUTH_STUDENT_ID = re.compile(r"(?<!\d)\d{8}(?!\d)")
_MODEL_DIRECTORY_CHARACTER = re.compile(r"[^a-z0-9]+")
DEFAULT_PROGRESS_INTERVAL_SECONDS = 10.0
_MIN_ETA_SAMPLES = 2
HUMAN_EVALUATION_FIELDS = (
    "model",
    "document_id",
    "content_relevance_score",
    "content_relevance_notes",
    "fluency_readability_score",
    "fluency_readability_notes",
    "summary_quality_consistency_score",
    "summary_quality_consistency_notes",
)
HUMAN_SCORE_FIELDS = (
    "content_relevance_score",
    "fluency_readability_score",
    "summary_quality_consistency_score",
)
RESOURCE_PERFORMANCE_FIELDS = (
    "system_cpu_mean_percent",
    "system_cpu_peak_percent",
    "process_cpu_mean_percent",
    "process_cpu_peak_percent",
    "ram_mean_mb",
    "ram_peak_mb",
    "process_rss_mean_mb",
    "process_rss_peak_mb",
    "gpu_mean_percent",
    "gpu_peak_percent",
    "vram_mean_mb",
    "vram_peak_mb",
)


class GenerationClient(Protocol):
    def generate_result(
        self, prompt: str, *, think: bool | None = None
    ) -> OllamaGenerationResult: ...


ClientFactory = Callable[[Settings], GenerationClient]


def parse_progress_interval(value: str) -> float:
    """Parse a finite, positive heartbeat interval for argparse."""

    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("progress interval must be a number") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("progress interval must be greater than zero")
    return interval


def format_elapsed(seconds: float) -> str:
    """Format non-negative elapsed seconds as minutes and seconds."""

    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


def format_run_start(current: int, total: int, model: str, document_id: str) -> str:
    """Format the immediate pre-run progress line."""

    return f"[{current}/{total}] {model} | {document_id} | START"


def format_generation_heartbeat(
    current: int,
    total: int,
    model: str,
    document_id: str,
    elapsed: float,
) -> str:
    """Format a periodic generation heartbeat line."""

    return (
        f"[{current}/{total}] {model} | {document_id} | "
        f"elapsed={format_elapsed(elapsed)} | generating..."
    )


def estimate_remaining_seconds(
    completed_durations: Sequence[float],
    *,
    completed_runs: int,
    total_runs: int,
) -> float | None:
    """Estimate remaining wall time using completed runs only."""

    if completed_runs < _MIN_ETA_SAMPLES or completed_runs >= total_runs:
        return None
    observed = [
        duration
        for duration in completed_durations[:completed_runs]
        if math.isfinite(duration) and duration >= 0
    ]
    if len(observed) < _MIN_ETA_SAMPLES:
        return None
    average = sum(observed) / len(observed)
    return average * (total_runs - completed_runs)


def format_benchmark_progress(
    *,
    completed_runs: int,
    total_runs: int,
    benchmark_elapsed: float,
    completed_durations: Sequence[float],
    now: datetime | None = None,
) -> str:
    """Format benchmark-wide completion, average, ETA, and finish time."""

    observed = [
        duration
        for duration in completed_durations[:completed_runs]
        if math.isfinite(duration) and duration >= 0
    ]
    average = sum(observed) / len(observed) if observed else None
    remaining = estimate_remaining_seconds(
        completed_durations,
        completed_runs=completed_runs,
        total_runs=total_runs,
    )
    average_text = format_elapsed(average) if average is not None else "unknown"
    eta_text = format_elapsed(remaining) if remaining is not None else "unknown"
    finish_text = "unknown"
    if remaining is not None:
        current_time = now or datetime.now().astimezone()
        finish_text = (current_time + timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"Progress: {completed_runs}/{total_runs} completed | "
        f"elapsed={format_elapsed(benchmark_elapsed)} | avg={average_text} | "
        f"ETA={eta_text} | finish={finish_text}"
    )


def format_run_result(
    current: int,
    total: int,
    model: str,
    document_id: str,
    result: dict[str, Any],
) -> str:
    """Format one completed model/document result without changing its metrics."""

    status = (
        "success"
        if result.get("status") == "success"
        else (result.get("failure_category") or "failure")
    )
    performance = result.get("performance", {})

    def seconds_value(field: str) -> str:
        value = performance.get(field)
        return f"{float(value):.3f}s" if isinstance(value, int | float) else "n/a"

    tokens_per_second = performance.get("tokens_per_second")
    tokens_text = (
        f"{float(tokens_per_second):.3f}" if isinstance(tokens_per_second, int | float) else "n/a"
    )
    return (
        f"[{current}/{total}] {model} | {document_id} | DONE status={status} | "
        f"generation={seconds_value('llm_generation_seconds')} | "
        f"total={seconds_value('total_seconds')} | tokens/sec={tokens_text}"
    )


@contextmanager
def generation_heartbeat(
    current: int,
    total: int,
    model: str,
    document_id: str,
    interval_seconds: float,
    *,
    output: TextIO | None = None,
):
    """Print periodic flushed progress while the wrapped run remains active."""

    stream = output or sys.stdout
    stopped = threading.Event()
    started = time.perf_counter()

    def emit() -> None:
        while not stopped.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(
                format_generation_heartbeat(
                    current,
                    total,
                    model,
                    document_id,
                    elapsed,
                ),
                file=stream,
                flush=True,
            )

    thread = threading.Thread(target=emit, name="benchmark-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """Immutable shared input used for every model for one indexed document."""

    entry: DatasetEntry
    ground_truth: GroundTruthRecord
    selected_pages: tuple[int, ...]
    source_text: str
    evidence: WikiSourceEvidence
    prompt: str
    preparation_seconds: float


@dataclass(frozen=True, slots=True)
class PreparationFailure:
    """A document that could not reach the model-independent benchmark boundary."""

    entry: DatasetEntry
    category: str
    error: str
    preparation_seconds: float
    ground_truth: GroundTruthRecord | None = None


PreparedItem = PreparedDocument | PreparationFailure


def normalize_value(value: object | None, *, case_insensitive: bool = False) -> str:
    """Normalize conservatively with NFC and whitespace folding only."""

    if value is None:
        return ""
    normalized = " ".join(unicodedata.normalize("NFC", str(value)).strip().split())
    return normalized.casefold() if case_insensitive else normalized


def compare_scalar(
    expected: object | None,
    actual: object | None,
    *,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Compare one scalar without fuzzy matching or translation."""

    expected_key = normalize_value(expected, case_insensitive=case_insensitive)
    actual_key = normalize_value(actual, case_insensitive=case_insensitive)
    return {
        "expected": expected,
        "actual": actual,
        "match": expected_key == actual_key,
    }


def _precision_recall_f1(
    true_positive: int, false_positive: int, false_negative: int
) -> dict[str, Any]:
    predicted = true_positive + false_positive
    expected = true_positive + false_negative
    precision = true_positive / predicted if predicted else 1.0
    recall = true_positive / expected if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def multi_value_metrics(
    expected: Iterable[object],
    actual: Iterable[object],
    *,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Compute exact set precision/recall/F1 after conservative normalization."""

    expected_values = list(expected)
    actual_values = list(actual)
    expected_keys = {
        normalize_value(value, case_insensitive=case_insensitive)
        for value in expected_values
        if normalize_value(value, case_insensitive=case_insensitive)
    }
    actual_keys = {
        normalize_value(value, case_insensitive=case_insensitive)
        for value in actual_values
        if normalize_value(value, case_insensitive=case_insensitive)
    }
    counts = _precision_recall_f1(
        len(expected_keys & actual_keys),
        len(actual_keys - expected_keys),
        len(expected_keys - actual_keys),
    )
    return {"expected": expected_values, "actual": actual_values, **counts}


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def expected_metadata(record: GroundTruthRecord) -> dict[str, Any]:
    """Map curated Ground Truth to the API metadata contract.

    The current GroundTruth schema has no separate student-ID field, so IDs are
    conservatively read as standalone eight-digit values from its curated raw text.
    """

    student_ids = _unique(_GROUND_TRUTH_STUDENT_ID.findall(record.raw_text or ""))
    return {
        "title": record.title_th,
        "english_title": record.title_en,
        "students": list(record.students),
        "student_ids": student_ids,
        "advisor": record.advisor,
        "academic_year": str(record.academic_year) if record.academic_year is not None else None,
        "keywords": list(record.keywords),
    }


def evidence_metadata(evidence: WikiSourceEvidence) -> dict[str, Any]:
    """Map the same deterministic evidence returned by the production API."""

    return {
        "title": evidence.title,
        "english_title": evidence.title_en,
        "students": list(evidence.students),
        "student_ids": list(evidence.student_ids),
        "advisor": evidence.advisor,
        "academic_year": evidence.academic_year,
        "keywords": list(evidence.keywords),
    }


def compare_metadata(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Score every requested metadata field deterministically."""

    scalar = {
        field: compare_scalar(
            expected.get(field),
            actual.get(field),
            case_insensitive=field in {"english_title", "advisor"},
        )
        for field in SCALAR_FIELDS
    }
    multi_value = {
        field: multi_value_metrics(
            expected.get(field, []),
            actual.get(field, []),
            case_insensitive=field in {"students", "keywords"},
        )
        for field in MULTI_VALUE_FIELDS
    }
    return {
        "scalar": scalar,
        "scalar_accuracy": sum(metric["match"] for metric in scalar.values()) / len(scalar),
        "multi_value": multi_value,
    }


def model_directory_name(model: str) -> str:
    """Convert an Ollama model name into a stable, filesystem-safe directory."""

    directory = _MODEL_DIRECTORY_CHARACTER.sub("-", model.casefold()).strip("-")
    if not directory:
        raise ValueError(f"Model name has no filesystem-safe characters: {model!r}")
    return directory


def prepare_entry(data_dir: Path, entry: DatasetEntry) -> PreparedDocument:
    """Run the shared model-independent production pipeline exactly once."""

    started = time.perf_counter()
    ground_truth = load_groundtruth(data_dir / entry.groundtruth)
    settings = get_settings()
    with (data_dir / entry.pdf).open("rb") as pdf_file:
        native_extraction = extract_pdf(pdf_file)
        extraction = native_extraction
        try:
            extraction = apply_ocr_fallback(
                pdf_file,
                native_extraction,
                load_ocr_provider(settings),
                front_matter_page_limit=settings.ocr_max_pages,
            )
        except OcrServiceError as error:
            extraction = add_ocr_failure_warning(native_extraction, error)
    selection = prepare_wiki_source(extraction)
    evidence = collect_wiki_source_evidence(selection.source_text)
    prompt = build_wiki_generation_prompt(selection.source_text)
    return PreparedDocument(
        entry=entry,
        ground_truth=ground_truth,
        selected_pages=selection.selected_pages,
        source_text=selection.source_text,
        evidence=evidence,
        prompt=prompt,
        preparation_seconds=time.perf_counter() - started,
    )


def _preparation_failure(
    data_dir: Path, entry: DatasetEntry, started: float, error: Exception
) -> PreparationFailure:
    ground_truth: GroundTruthRecord | None = None
    with suppress(OSError, ValueError):
        ground_truth = load_groundtruth(data_dir / entry.groundtruth)
    if isinstance(error, FileNotFoundError):
        category = "missing_input"
    elif isinstance(error, PdfExtractionError):
        category = "extraction_failed"
    elif isinstance(error, ValueError):
        category = "invalid_input"
    else:
        category = "preparation_failed"
    return PreparationFailure(
        entry=entry,
        category=category,
        error=str(error),
        preparation_seconds=time.perf_counter() - started,
        ground_truth=ground_truth,
    )


def prepare_dataset(data_dir: Path, entries: Sequence[DatasetEntry]) -> list[PreparedItem]:
    """Prepare all pairs independently so one broken document cannot abort a run."""

    prepared: list[PreparedItem] = []
    for entry in entries:
        started = time.perf_counter()
        try:
            prepared.append(prepare_entry(data_dir, entry))
        except Exception as exc:
            prepared.append(_preparation_failure(data_dir, entry, started, exc))
    return prepared


def extraction_artifact(item: PreparedItem) -> dict[str, Any]:
    """Report deterministic extraction once, independently of every model."""

    record = item.ground_truth
    artifact: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "deterministic_extraction_metrics",
        "metric_scope": "extraction_pipeline",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "document_id": item.entry.document_id,
        "status": "failure",
        "failure_category": None,
        "error": None,
        "ground_truth": record.model_dump(mode="json") if record else None,
        "expected_metadata": expected_metadata(record) if record else None,
        "actual_metadata": None,
        "deterministic_extraction_metrics": None,
        "selected_pages": [],
        "focused_source": None,
        "evidence": None,
        "prompt": None,
        "prompt_sha256": None,
        "preparation_seconds": item.preparation_seconds,
    }
    if isinstance(item, PreparationFailure):
        artifact.update(failure_category=item.category, error=item.error)
        return artifact

    expected = expected_metadata(item.ground_truth)
    actual = evidence_metadata(item.evidence)
    metrics = compare_metadata(expected, actual)
    metrics["metric_scope"] = "extraction_pipeline"
    artifact.update(
        status="success",
        expected_metadata=expected,
        actual_metadata=actual,
        deterministic_extraction_metrics=metrics,
        selected_pages=list(item.selected_pages),
        focused_source=item.source_text,
        evidence=asdict(item.evidence),
        prompt=item.prompt,
        prompt_sha256=hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
    )
    return artifact


def _base_model_artifact(item: PreparedItem, model: str, settings: Settings) -> dict[str, Any]:
    record = item.ground_truth
    artifact: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "model_generation_metrics",
        "metric_scopes": {
            "raw_model_output": "raw_model_output",
            "finalized_markdown": "finalized_production_output",
            "llm_performance": "raw_model_output",
            "end_to_end_performance": "finalized_production_output",
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "document_id": item.entry.document_id,
        "status": "failure",
        "failure_category": None,
        "error_type": None,
        "error": None,
        "settings": {
            "temperature": settings.ollama_temperature,
            "timeout_seconds": settings.ollama_timeout_seconds,
            "think": False,
            "stream": False,
        },
        "deterministic_extraction_artifact": (f"../extraction/{item.entry.document_id}.json"),
        "ground_truth": record.model_dump(mode="json") if record else None,
        "selected_pages": [],
        "focused_source": None,
        "evidence": None,
        "prompt": None,
        "prompt_sha256": None,
        "raw_model_output": None,
        "finalized_markdown": None,
        "refinement_changes": [],
        "model_generation_metrics": {
            "raw_model_output": None,
            "finalized_production_output": None,
        },
        "performance": {
            "llm_generation_seconds": None,
            "total_seconds": item.preparation_seconds,
            "prompt_eval_count": None,
            "prompt_eval_duration": None,
            "eval_count": None,
            "eval_duration": None,
            "tokens_per_second": None,
            "model_size_bytes": None,
            "resource_telemetry": {
                "status": "unavailable",
                "sample_count": 0,
                "gpu_provider": None,
                **{field: None for field in RESOURCE_PERFORMANCE_FIELDS},
            },
        },
    }
    if isinstance(item, PreparedDocument):
        artifact.update(
            selected_pages=list(item.selected_pages),
            focused_source=item.source_text,
            evidence=asdict(item.evidence),
            prompt=item.prompt,
            prompt_sha256=hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
        )
    return artifact


def generation_metrics(
    source_text: str,
    markdown: str,
    ground_truth: GroundTruthRecord,
    *,
    metric_scope: str,
    output_token_count: int | None = None,
) -> dict[str, Any]:
    """Compute deterministic review metrics for one model-dependent output layer."""

    evaluation = evaluate_markdown(source_text, markdown, ground_truth)
    canonical_headings = [line for line in markdown.splitlines() if line in REQUIRED_WIKI_HEADINGS]
    canonical_present = set(canonical_headings)
    missing_sections = [
        heading for heading in REQUIRED_WIKI_HEADINGS if heading not in canonical_present
    ]
    marker_usage = evaluation["missing_marker_usage"]
    unsupported_flags = [
        {"detector": "out_of_source_lexical_indicator", **indicator}
        for indicator in evaluation["hallucination_indicators"]
    ]
    unsupported_flags.extend(
        {
            "detector": "missing_marker_for_likely_unsupported_section",
            "kind": "section_without_source_cue",
            "value": heading,
        }
        for heading in marker_usage["possible_omissions"]
    )
    return {
        "metric_scope": metric_scope,
        "structure_valid": evaluation["structure"]["valid"],
        "structure_issues": evaluation["structure"]["issues"],
        "canonical_section_count": len(canonical_headings),
        "canonical_sections_present_count": len(canonical_present),
        "missing_canonical_sections": missing_sections,
        "missing_marker_section_count": len(marker_usage["used_in_sections"]),
        "missing_marker_sections": marker_usage["used_in_sections"],
        "missing_marker_exact_format_valid": marker_usage["exact_format_valid"],
        "unsupported_claim_flag_count": len(unsupported_flags),
        "unsupported_claim_flags": unsupported_flags,
        "output_token_count": output_token_count,
        "output_character_count": len(markdown),
    }


def _provider_performance(result: OllamaGenerationResult) -> dict[str, Any]:
    return {
        "prompt_eval_count": result.prompt_eval_count,
        "prompt_eval_duration": result.prompt_eval_duration,
        "eval_count": result.eval_count,
        "eval_duration": result.eval_duration,
        "tokens_per_second": result.tokens_per_second,
    }


def _record_failure(
    artifact: dict[str, Any], category: str, error: Exception | str
) -> dict[str, Any]:
    artifact.update(
        status="failure",
        failure_category=category,
        error_type=type(error).__name__ if isinstance(error, Exception) else None,
        error=str(error),
    )
    return artifact


def run_model_document(
    item: PreparedItem,
    model: str,
    settings: Settings,
    client: GenerationClient,
) -> dict[str, Any]:
    """Generate, finalize, validate, and score one model/document without raising."""

    artifact = _base_model_artifact(item, model, settings)
    if isinstance(item, PreparationFailure):
        return _record_failure(artifact, item.category, item.error)

    telemetry = ResourceTelemetrySampler()
    telemetry.start()
    run_started = time.perf_counter()
    llm_started = time.perf_counter()
    try:
        provider_result = client.generate_result(item.prompt, think=False)
    except LLMServiceError as exc:
        failed_at = time.perf_counter()
        artifact["performance"]["llm_generation_seconds"] = failed_at - llm_started
        artifact["performance"]["total_seconds"] += failed_at - run_started
        artifact["performance"]["resource_telemetry"] = telemetry.stop()
        return _record_failure(artifact, "llm_generation_failed", exc)
    except Exception as exc:
        failed_at = time.perf_counter()
        artifact["performance"]["llm_generation_seconds"] = failed_at - llm_started
        artifact["performance"]["total_seconds"] += failed_at - run_started
        artifact["performance"]["resource_telemetry"] = telemetry.stop()
        return _record_failure(artifact, "unexpected_generation_failure", exc)

    llm_finished = time.perf_counter()
    artifact["performance"]["llm_generation_seconds"] = llm_finished - llm_started
    artifact["performance"]["resource_telemetry"] = telemetry.stop()
    telemetry_overhead = time.perf_counter() - llm_finished
    artifact["performance"].update(_provider_performance(provider_result))
    artifact["raw_model_output"] = provider_result.response
    raw_metrics = generation_metrics(
        item.source_text,
        provider_result.response,
        item.ground_truth,
        metric_scope="raw_model_output",
        output_token_count=provider_result.eval_count,
    )
    artifact["model_generation_metrics"]["raw_model_output"] = raw_metrics
    try:
        refinement = refine_wiki_markdown(item.source_text, provider_result.response)
    except WikiOutputError as exc:
        artifact["performance"]["total_seconds"] += (
            time.perf_counter() - run_started - telemetry_overhead
        )
        return _record_failure(artifact, "finalization_failed", exc)
    except Exception as exc:
        artifact["performance"]["total_seconds"] += (
            time.perf_counter() - run_started - telemetry_overhead
        )
        return _record_failure(artifact, "unexpected_finalization_failure", exc)

    artifact["finalized_markdown"] = refinement.markdown
    artifact["refinement_changes"] = list(refinement.changes)
    finalized_metrics = generation_metrics(
        item.source_text,
        refinement.markdown,
        item.ground_truth,
        metric_scope="finalized_production_output",
    )
    artifact["model_generation_metrics"]["finalized_production_output"] = finalized_metrics
    artifact["performance"]["total_seconds"] += (
        time.perf_counter() - run_started - telemetry_overhead
    )
    if not finalized_metrics["structure_valid"]:
        return _record_failure(artifact, "validation_failed", "Strict Wiki validation failed")
    artifact["status"] = "success"
    return artifact


def aggregate_extraction_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate extraction quality once per document, never once per model."""

    scalar_totals = {field: [0, 0] for field in SCALAR_FIELDS}
    multi_totals = {field: [0, 0, 0] for field in MULTI_VALUE_FIELDS}
    failures: Counter[str] = Counter()
    preparation_seconds: list[float] = []

    for result in results:
        if result["status"] != "success":
            failures[result.get("failure_category") or "unknown_failure"] += 1
        metrics = result.get("deterministic_extraction_metrics")
        if metrics:
            for field, metric in metrics["scalar"].items():
                scalar_totals[field][0] += int(metric["match"])
                scalar_totals[field][1] += 1
            for field, metric in metrics["multi_value"].items():
                multi_totals[field][0] += metric["true_positive"]
                multi_totals[field][1] += metric["false_positive"]
                multi_totals[field][2] += metric["false_negative"]
        value = result.get("preparation_seconds")
        if isinstance(value, int | float) and not isinstance(value, bool):
            preparation_seconds.append(float(value))

    scalar_accuracy = {
        field: correct / assessed if assessed else None
        for field, (correct, assessed) in scalar_totals.items()
    }
    overall_correct = sum(correct for correct, _ in scalar_totals.values())
    overall_assessed = sum(assessed for _, assessed in scalar_totals.values())
    return {
        "metric_scope": "extraction_pipeline",
        "total_documents": len(results),
        "successful_extractions": sum(result["status"] == "success" for result in results),
        "failed_extractions": sum(result["status"] != "success" for result in results),
        "failure_categories": dict(failures),
        "scalar_field_accuracy": scalar_accuracy,
        "scalar_accuracy_overall": (
            overall_correct / overall_assessed if overall_assessed else None
        ),
        "multi_value_metrics": {
            field: _precision_recall_f1(*counts) for field, counts in multi_totals.items()
        },
        "preparation_seconds_mean": (
            sum(preparation_seconds) / len(preparation_seconds) if preparation_seconds else None
        ),
    }


def _aggregate_generation_layer(results: Sequence[dict[str, Any]], layer: str) -> dict[str, Any]:
    metrics = [result.get("model_generation_metrics", {}).get(layer) for result in results]
    assessed = [metric for metric in metrics if metric is not None]
    missing_sections: Counter[str] = Counter()
    unsupported_kinds: Counter[str] = Counter()
    for metric in assessed:
        missing_sections.update(metric["missing_canonical_sections"])
        unsupported_kinds.update(flag["detector"] for flag in metric["unsupported_claim_flags"])

    def mean(field: str) -> float | None:
        values = [metric[field] for metric in assessed if metric.get(field) is not None]
        return sum(values) / len(values) if values else None

    scope = "raw_model_output" if layer == "raw_model_output" else "finalized_production_output"
    outputs_with_flags = sum(metric["unsupported_claim_flag_count"] > 0 for metric in assessed)
    return {
        "metric_scope": scope,
        "outputs_assessed": len(assessed),
        "structure_valid_count": sum(metric["structure_valid"] is True for metric in assessed),
        "structure_valid_rate": (
            sum(metric["structure_valid"] is True for metric in assessed) / len(assessed)
            if assessed
            else None
        ),
        "canonical_section_count_mean": mean("canonical_section_count"),
        "missing_marker_section_count_mean": mean("missing_marker_section_count"),
        "unsupported_claim_flag_count": sum(
            metric["unsupported_claim_flag_count"] for metric in assessed
        ),
        "outputs_with_unsupported_claim_flags": outputs_with_flags,
        "outputs_without_unsupported_claim_flags": len(assessed) - outputs_with_flags,
        "unsupported_claim_review_pass_rate": (
            (len(assessed) - outputs_with_flags) / len(assessed) if assessed else None
        ),
        "unsupported_claim_flag_types": dict(unsupported_kinds),
        "missing_canonical_sections": dict(missing_sections),
        "output_token_count_mean": mean("output_token_count"),
        "output_character_count_mean": mean("output_character_count"),
    }


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    """Return a linearly interpolated percentile for finite numeric observations."""

    if not 0 <= percentile_value <= 1:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rate_rubric_score(
    rate: float | None,
    *,
    score_4_at: float,
    score_3_at: float,
    score_2_at: float,
) -> int | None:
    """Map a higher-is-better rate to the requested four-point rubric."""

    if rate is None:
        return None
    if rate >= score_4_at:
        return 4
    if rate >= score_3_at:
        return 3
    if rate >= score_2_at:
        return 2
    return 1


def latency_rubric_score(mean_total_seconds: float | None) -> int | None:
    """Apply the researcher-defined operational total-latency thresholds."""

    if mean_total_seconds is None:
        return None
    if mean_total_seconds <= 120:
        return 4
    if mean_total_seconds <= 300:
        return 3
    if mean_total_seconds <= 600:
        return 2
    return 1


def _aggregate_resource_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {field: [] for field in RESOURCE_PERFORMANCE_FIELDS}
    statuses: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    model_sizes: list[float] = []
    tokens_per_second: list[float] = []
    for result in results:
        performance = result.get("performance", {})
        telemetry = performance.get("resource_telemetry") or {}
        statuses[telemetry.get("status") or "unavailable"] += 1
        if telemetry.get("gpu_provider"):
            providers[str(telemetry["gpu_provider"])] += 1
        for field in RESOURCE_PERFORMANCE_FIELDS:
            value = telemetry.get(field)
            if isinstance(value, int | float) and not isinstance(value, bool):
                values[field].append(float(value))
        model_size = performance.get("model_size_bytes")
        if isinstance(model_size, int | float) and not isinstance(model_size, bool):
            model_sizes.append(float(model_size))
        token_rate = performance.get("tokens_per_second")
        if isinstance(token_rate, int | float) and not isinstance(token_rate, bool):
            tokens_per_second.append(float(token_rate))

    def aggregate(field: str) -> float | None:
        observations = values[field]
        if not observations:
            return None
        return max(observations) if "_peak_" in field else sum(observations) / len(observations)

    cpu_gpu_status = "cpu_ram_available_gpu_available"
    cpu_only_status = "cpu_ram_available_gpu_unavailable"
    if results and statuses.get(cpu_gpu_status, 0) == len(results):
        status = "measured"
    elif statuses.get(cpu_gpu_status, 0) or statuses.get(cpu_only_status, 0):
        status = "partial"
    else:
        status = "unavailable"
    return {
        "mean_cpu_percent": aggregate("system_cpu_mean_percent"),
        "peak_cpu_percent": aggregate("system_cpu_peak_percent"),
        "mean_process_cpu_percent": aggregate("process_cpu_mean_percent"),
        "peak_process_cpu_percent": aggregate("process_cpu_peak_percent"),
        "mean_ram_mb": aggregate("ram_mean_mb"),
        "peak_ram_mb": aggregate("ram_peak_mb"),
        "mean_process_rss_mb": aggregate("process_rss_mean_mb"),
        "peak_process_rss_mb": aggregate("process_rss_peak_mb"),
        "mean_gpu_percent": aggregate("gpu_mean_percent"),
        "peak_gpu_percent": aggregate("gpu_peak_percent"),
        "mean_vram_mb": aggregate("vram_mean_mb"),
        "peak_vram_mb": aggregate("vram_peak_mb"),
        "mean_tokens_per_second": (
            sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else None
        ),
        "model_size_bytes": model_sizes[0] if model_sizes and len(set(model_sizes)) == 1 else None,
        "telemetry_status_counts": dict(statuses),
        "gpu_provider_counts": dict(providers),
        "resource_efficiency_score_status": status,
    }


def aggregate_model_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only model-dependent generation quality and performance."""

    failures: Counter[str] = Counter()
    performance_fields = (
        "llm_generation_seconds",
        "total_seconds",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "tokens_per_second",
    )
    performance_values: dict[str, list[float]] = {field: [] for field in performance_fields}

    for result in results:
        if result["status"] != "success":
            failures[result.get("failure_category") or "unknown_failure"] += 1
        for field in performance_fields:
            value = result.get("performance", {}).get(field)
            if isinstance(value, int | float) and not isinstance(value, bool):
                performance_values[field].append(float(value))

    total_seconds = performance_values["total_seconds"]
    return {
        "total_runs": len(results),
        "generation_success_count": sum(result["status"] == "success" for result in results),
        "failed_generation_count": sum(result["status"] != "success" for result in results),
        "failure_categories": dict(failures),
        "raw_model_output": _aggregate_generation_layer(results, "raw_model_output"),
        "finalized_production_output": _aggregate_generation_layer(
            results, "finalized_production_output"
        ),
        "performance_metric_scopes": {
            "llm_generation_seconds": "raw_model_output",
            "prompt_eval_count": "raw_model_output",
            "prompt_eval_duration": "raw_model_output",
            "eval_count": "raw_model_output",
            "eval_duration": "raw_model_output",
            "tokens_per_second": "raw_model_output",
            "total_seconds": "finalized_production_output",
        },
        "performance_mean": {
            field: sum(values) / len(values) if values else None
            for field, values in performance_values.items()
        },
        "performance_distribution": {
            "median_total_seconds": percentile(total_seconds, 0.5),
            "p95_total_seconds": percentile(total_seconds, 0.95),
        },
        "resource_metrics": _aggregate_resource_metrics(results),
    }


def validate_human_score(value: object, *, field: str, row_number: int) -> int | None:
    """Validate an optional human rubric score without coercing missing values to zero."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"[1-4]", text):
        raise ValueError(f"{field} on human evaluation row {row_number} must be an integer 1-4")
    return int(text)


def read_human_evaluations(path: Path) -> list[dict[str, str]]:
    """Read and validate an existing human evaluation worksheet."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing_columns = [field for field in HUMAN_EVALUATION_FIELDS if field not in fieldnames]
        if missing_columns:
            raise ValueError(
                "Human evaluation CSV is missing columns: " + ", ".join(missing_columns)
            )
        rows = []
        seen: set[tuple[str, str]] = set()
        for row_number, source_row in enumerate(reader, start=2):
            row = {field: (source_row.get(field) or "") for field in HUMAN_EVALUATION_FIELDS}
            key = (row["model"], row["document_id"])
            if not all(key):
                raise ValueError(f"Human evaluation row {row_number} needs model and document_id")
            if key in seen:
                raise ValueError(f"Duplicate human evaluation row for {key[0]} / {key[1]}")
            seen.add(key)
            for field in HUMAN_SCORE_FIELDS:
                validate_human_score(row[field], field=field, row_number=row_number)
            rows.append(row)
    return rows


def write_human_evaluation_csv(
    path: Path,
    results: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Add blank successful runs while retaining every existing human-entered value."""

    existing = read_human_evaluations(path)
    rows_by_key = {(row["model"], row["document_id"]): row for row in existing}
    ordered_keys = list(rows_by_key)
    for result in results:
        if result.get("status") != "success":
            continue
        key = (str(result["model"]), str(result["document_id"]))
        if key not in rows_by_key:
            rows_by_key[key] = {
                field: key[0] if field == "model" else key[1] if field == "document_id" else ""
                for field in HUMAN_EVALUATION_FIELDS
            }
            ordered_keys.append(key)

    rows = [rows_by_key[key] for key in ordered_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HUMAN_EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def aggregate_human_evaluations(
    rows: Sequence[dict[str, str]],
    *,
    models: Sequence[str] | None = None,
    eligible_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Aggregate validated 1-4 human ratings per criterion and model."""

    selected_rows = [
        row
        for row in rows
        if eligible_keys is None or (row["model"], row["document_id"]) in eligible_keys
    ]
    model_order = (
        list(models)
        if models is not None
        else list(dict.fromkeys(row["model"] for row in selected_rows))
    )
    summaries: dict[str, Any] = {}
    for model in model_order:
        model_rows = [row for row in selected_rows if row["model"] == model]
        criterion_summaries: dict[str, Any] = {}
        all_scores: list[int] = []
        for field in HUMAN_SCORE_FIELDS:
            scores = [
                score
                for index, row in enumerate(model_rows, start=2)
                if (score := validate_human_score(row[field], field=field, row_number=index))
                is not None
            ]
            all_scores.extend(scores)
            criterion = field.removesuffix("_score")
            criterion_summaries[criterion] = {
                "mean": sum(scores) / len(scores) if scores else None,
                "rated_documents": len(scores),
                "missing_ratings": len(model_rows) - len(scores),
            }
        summaries[model] = {
            "successful_outputs_in_worksheet": len(model_rows),
            "criteria": criterion_summaries,
            "documents_with_any_rating": sum(
                any(row[field].strip() for field in HUMAN_SCORE_FIELDS) for row in model_rows
            ),
            "fully_rated_documents": sum(
                all(row[field].strip() for field in HUMAN_SCORE_FIELDS) for row in model_rows
            ),
            "overall_mean_human_score": (sum(all_scores) / len(all_scores) if all_scores else None),
        }
    return {"model_summaries": summaries}


def build_rubric_ready_metrics(
    model_results: dict[str, Sequence[dict[str, Any]]],
    model_summaries: dict[str, dict[str, Any]],
    human_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project existing benchmark observations onto the ten-criterion rubric."""

    output: dict[str, dict[str, Any]] = {}
    human_models = human_summary.get("model_summaries", {})
    for model, results in model_results.items():
        summary = model_summaries[model]
        raw = summary["raw_model_output"]
        finalized = summary["finalized_production_output"]
        total_runs = summary["total_runs"]
        llm_failures = sum(
            result.get("failure_category") == "llm_generation_failed" for result in results
        )
        robustness_rate = summary["generation_success_count"] / total_runs if total_runs else None
        runtime_rate = (total_runs - llm_failures) / total_runs if total_runs else None
        performance_mean = summary["performance_mean"]
        performance_distribution = summary["performance_distribution"]
        resource_metrics = summary["resource_metrics"]
        human_criteria = human_models.get(model, {}).get("criteria", {})
        instruction_rate = raw["structure_valid_rate"]
        structure_rate = finalized["structure_valid_rate"]
        grounding_rate = finalized["unsupported_claim_review_pass_rate"]
        output[model] = {
            "instruction_following_rate": instruction_rate,
            "instruction_following_rubric_score": rate_rubric_score(
                instruction_rate, score_4_at=0.95, score_3_at=0.80, score_2_at=0.60
            ),
            "structure_compliance_rate": structure_rate,
            "structure_compliance_rubric_score": rate_rubric_score(
                structure_rate, score_4_at=0.95, score_3_at=0.80, score_2_at=0.60
            ),
            "unsupported_claim_review_pass_rate": grounding_rate,
            "grounding_screen_rubric_score": rate_rubric_score(
                grounding_rate, score_4_at=0.90, score_3_at=0.75, score_2_at=0.60
            ),
            "content_relevance_mean": human_criteria.get("content_relevance", {}).get("mean"),
            "fluency_readability_mean": human_criteria.get("fluency_readability", {}).get("mean"),
            "summary_quality_consistency_mean": human_criteria.get(
                "summary_quality_consistency", {}
            ).get("mean"),
            "robustness_rate": robustness_rate,
            "robustness_rubric_score": rate_rubric_score(
                robustness_rate, score_4_at=0.95, score_3_at=0.80, score_2_at=0.60
            ),
            "runtime_reliability_rate": runtime_rate,
            "runtime_reliability_rubric_score": rate_rubric_score(
                runtime_rate, score_4_at=0.95, score_3_at=0.85, score_2_at=0.70
            ),
            "mean_llm_generation_seconds": performance_mean["llm_generation_seconds"],
            "mean_total_seconds": performance_mean["total_seconds"],
            "median_total_seconds": performance_distribution["median_total_seconds"],
            "p95_total_seconds": performance_distribution["p95_total_seconds"],
            "latency_rubric_score": latency_rubric_score(performance_mean["total_seconds"]),
            "mean_tokens_per_second": resource_metrics["mean_tokens_per_second"],
            "resource_metrics": resource_metrics,
            "resource_efficiency_score_status": resource_metrics[
                "resource_efficiency_score_status"
            ],
        }
    return output


def _extraction_csv_row(result: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    metrics = result.get("deterministic_extraction_metrics") or {}
    scalar = metrics.get("scalar", {})
    multi = metrics.get("multi_value", {})
    row: dict[str, Any] = {
        "record_type": "deterministic_extraction_metrics",
        "metric_scope": "extraction_pipeline",
        "model": None,
        "document_id": result["document_id"],
        "status": result["status"],
        "failure_category": result.get("failure_category"),
        "extraction_scalar_accuracy": metrics.get("scalar_accuracy"),
        "preparation_seconds": result.get("preparation_seconds"),
        "artifact": artifact_path.as_posix(),
    }
    for field in SCALAR_FIELDS:
        row[f"extraction_{field}_correct"] = scalar.get(field, {}).get("match")
    for field in MULTI_VALUE_FIELDS:
        for metric in ("precision", "recall", "f1"):
            row[f"extraction_{field}_{metric}"] = multi.get(field, {}).get(metric)
    return row


def _model_csv_rows(result: dict[str, Any], artifact_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    performance = result["performance"]
    resource = performance.get("resource_telemetry") or {}
    generation_layers = result.get("model_generation_metrics", {})
    raw_metrics = generation_layers.get("raw_model_output") or {}
    finalized_metrics = generation_layers.get("finalized_production_output") or {}
    for layer, scope in (
        ("raw_model_output", "raw_model_output"),
        ("finalized_production_output", "finalized_production_output"),
    ):
        metrics = result.get("model_generation_metrics", {}).get(layer) or {}
        row: dict[str, Any] = {
            "record_type": "model_generation_metrics",
            "metric_scope": scope,
            "model": result["model"],
            "document_id": result["document_id"],
            "status": result["status"],
            "failure_category": result.get("failure_category"),
            "raw_structure_valid": raw_metrics.get("structure_valid"),
            "finalized_structure_valid": finalized_metrics.get("structure_valid"),
            "unsupported_claim_review_pass": (
                metrics.get("unsupported_claim_flag_count") == 0 if metrics else None
            ),
            "llm_generation_failed": (result.get("failure_category") == "llm_generation_failed"),
            "validation_failed": result.get("failure_category") == "validation_failed",
            "structure_valid": metrics.get("structure_valid"),
            "canonical_section_count": metrics.get("canonical_section_count"),
            "canonical_sections_present_count": metrics.get("canonical_sections_present_count"),
            "missing_canonical_sections": json.dumps(
                metrics.get("missing_canonical_sections", []), ensure_ascii=False
            ),
            "missing_marker_section_count": metrics.get("missing_marker_section_count"),
            "unsupported_claim_flag_count": metrics.get("unsupported_claim_flag_count"),
            "unsupported_claim_flags": json.dumps(
                metrics.get("unsupported_claim_flags", []), ensure_ascii=False
            ),
            "output_token_count": metrics.get("output_token_count"),
            "output_character_count": metrics.get("output_character_count"),
            "llm_generation_seconds": performance.get("llm_generation_seconds"),
            "total_seconds": performance.get("total_seconds"),
            "prompt_eval_count": (
                performance.get("prompt_eval_count") if scope == "raw_model_output" else None
            ),
            "prompt_eval_duration": (
                performance.get("prompt_eval_duration") if scope == "raw_model_output" else None
            ),
            "eval_count": (performance.get("eval_count") if scope == "raw_model_output" else None),
            "eval_duration": (
                performance.get("eval_duration") if scope == "raw_model_output" else None
            ),
            "tokens_per_second": (
                performance.get("tokens_per_second") if scope == "raw_model_output" else None
            ),
            "cpu_mean_percent": resource.get("system_cpu_mean_percent"),
            "cpu_peak_percent": resource.get("system_cpu_peak_percent"),
            "process_cpu_mean_percent": resource.get("process_cpu_mean_percent"),
            "process_cpu_peak_percent": resource.get("process_cpu_peak_percent"),
            "ram_mean_mb": resource.get("ram_mean_mb"),
            "ram_peak_mb": resource.get("ram_peak_mb"),
            "process_rss_mean_mb": resource.get("process_rss_mean_mb"),
            "process_rss_peak_mb": resource.get("process_rss_peak_mb"),
            "gpu_mean_percent": resource.get("gpu_mean_percent"),
            "gpu_peak_percent": resource.get("gpu_peak_percent"),
            "vram_mean_mb": resource.get("vram_mean_mb"),
            "vram_peak_mb": resource.get("vram_peak_mb"),
            "gpu_telemetry_provider": resource.get("gpu_provider"),
            "resource_telemetry_status": resource.get("status"),
            "model_size_bytes": performance.get("model_size_bytes"),
            "artifact": artifact_path.as_posix(),
        }
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else ["record_type", "metric_scope", "model", "document_id", "status"]
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_prepared_documents(
    prepared: Sequence[PreparedItem],
    models: Sequence[str],
    output_dir: Path,
    base_settings: Settings,
    *,
    client_factory: ClientFactory = OllamaClient,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run every model over exactly the same prepared objects and save artifacts."""

    if not math.isfinite(progress_interval) or progress_interval <= 0:
        raise ValueError("progress interval must be greater than zero")
    directories = {model: model_directory_name(model) for model in models}
    if len(set(directories.values())) != len(directories):
        raise ValueError("Selected model names resolve to duplicate output directories")

    all_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    extraction_results: list[dict[str, Any]] = []
    for item in prepared:
        result = extraction_artifact(item)
        artifact_relative = Path("extraction") / f"{item.entry.document_id}.json"
        _write_json(output_dir / artifact_relative, result)
        extraction_results.append(result)
        csv_rows.append(_extraction_csv_row(result, artifact_relative))

    model_summaries: dict[str, Any] = {}
    results_by_model: dict[str, list[dict[str, Any]]] = {}
    total_runs = len(models) * len(prepared)
    completed_runs = 0
    completed_durations: list[float] = []
    benchmark_started = time.perf_counter()
    for model in models:
        model_settings = base_settings.model_copy(update={"ollama_model": model})
        client = client_factory(model_settings)
        model_results: list[dict[str, Any]] = []
        for item in prepared:
            current_run = completed_runs + 1
            document_id = item.entry.document_id
            print(
                format_run_start(current_run, total_runs, model, document_id),
                flush=True,
            )
            run_wall_started = time.perf_counter()
            try:
                with generation_heartbeat(
                    current_run,
                    total_runs,
                    model,
                    document_id,
                    progress_interval,
                ):
                    result = run_model_document(item, model, model_settings, client)
            except Exception as exc:
                result = _record_failure(
                    _base_model_artifact(item, model, model_settings),
                    "unexpected_run_failure",
                    exc,
                )
            run_wall_seconds = time.perf_counter() - run_wall_started
            completed_durations.append(run_wall_seconds)
            completed_runs += 1
            print(
                format_run_result(current_run, total_runs, model, document_id, result),
                flush=True,
            )
            print(
                format_benchmark_progress(
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    benchmark_elapsed=time.perf_counter() - benchmark_started,
                    completed_durations=completed_durations,
                ),
                flush=True,
            )
            artifact_relative = Path(directories[model]) / f"{item.entry.document_id}.json"
            _write_json(output_dir / artifact_relative, result)
            model_results.append(result)
            all_results.append(result)
            csv_rows.extend(_model_csv_rows(result, artifact_relative))
        model_summaries[model] = aggregate_model_results(model_results)
        results_by_model[model] = model_results

    _write_csv(output_dir / "benchmark_results.csv", csv_rows)
    human_rows = write_human_evaluation_csv(
        output_dir / "benchmark_human_evaluation.csv", all_results
    )
    successful_keys = {
        (str(result["model"]), str(result["document_id"]))
        for result in all_results
        if result.get("status") == "success"
    }
    human_summary = aggregate_human_evaluations(
        human_rows,
        models=models,
        eligible_keys=successful_keys,
    )
    rubric_ready_metrics = build_rubric_ready_metrics(
        results_by_model,
        model_summaries,
        human_summary,
    )
    summary = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "models": list(models),
        "document_ids": [item.entry.document_id for item in prepared],
        "temperature": base_settings.ollama_temperature,
        "thinking_enabled": False,
        "duration_units": {
            "llm_generation_seconds": "seconds",
            "total_seconds": "seconds",
            "prompt_eval_duration": "nanoseconds",
            "eval_duration": "nanoseconds",
        },
        "metric_scope_definitions": {
            "extraction_pipeline": (
                "Model-independent PDF extraction, focused selection, and evidence extraction"
            ),
            "raw_model_output": "Direct Ollama response before production finalization",
            "finalized_production_output": (
                "Raw response after the production finalizer and strict validator"
            ),
        },
        "deterministic_extraction_metrics": aggregate_extraction_results(extraction_results),
        "model_generation_metrics": {"model_summaries": model_summaries},
        "human_evaluation_metrics": human_summary,
        "rubric_ready_metrics": rubric_ready_metrics,
        "rubric_metadata": {
            "scale": "1-4",
            "automatic_criteria": [1, 2, 7, 8, 9],
            "automatic_screening_plus_human_verification": [3],
            "human_criteria": [4, 5, 6],
            "resource_telemetry_criterion": [10],
            "grounding_screen_note": (
                "Unsupported-claim flags are deterministic review heuristics, not proof of "
                "factual correctness; flagged and unflagged outputs still require human review."
            ),
            "researcher_defined_thresholds": [
                "Criteria 1 and 2: rates >=0.95/0.80/0.60 receive scores 4/3/2; "
                "lower rates receive 1.",
                "Criterion 3 screening: rates >=0.90/0.75/0.60 receive scores "
                "4/3/2; lower rates receive 1.",
                "Criterion 7: rates >=0.95/0.80/0.60 receive scores 4/3/2; lower rates receive 1.",
                "Criterion 8: rates >=0.95/0.85/0.70 receive scores 4/3/2; lower rates receive 1.",
                "Criterion 9 mean total seconds <=120/300/600 receive scores 4/3/2; "
                "higher latency receives 1.",
            ],
            "latency_threshold_note": (
                "Operational thresholds were defined by the researcher and were not taken "
                "directly from prior literature."
            ),
        },
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    return {
        "summary": summary,
        "extraction_results": extraction_results,
        "results": all_results,
        "human_evaluation_rows": human_rows,
    }


def select_dataset_entries(
    entries: Sequence[DatasetEntry],
    *,
    document_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[DatasetEntry]:
    """Select indexed documents without changing index or requested ordering."""

    if document_ids is not None and limit is not None:
        raise ValueError("--limit cannot be used together with --documents")
    if document_ids is None:
        return list(entries[:limit] if limit is not None else entries)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("--documents must not contain duplicate document IDs")

    entries_by_id = {entry.document_id: entry for entry in entries}
    unknown = [document_id for document_id in document_ids if document_id not in entries_by_id]
    if unknown:
        raise ValueError(f"Unknown document ID(s): {', '.join(unknown)}")
    return [entries_by_id[document_id] for document_id in document_ids]


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    *,
    limit: int | None = None,
    document_ids: Sequence[str] | None = None,
    client_factory: ClientFactory = OllamaClient,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Load indexed pairs, prepare shared inputs, and run all selected models."""

    index = load_dataset_index(data_dir / "index.json")
    entries = select_dataset_entries(
        index.documents,
        document_ids=document_ids,
        limit=limit,
    )
    prepared = prepare_dataset(data_dir, entries)
    return benchmark_prepared_documents(
        prepared,
        models,
        output_dir,
        get_settings(),
        client_factory=client_factory,
        progress_interval=progress_interval,
    )


def _print_summary(report: dict[str, Any]) -> None:
    extraction = report["summary"]["deterministic_extraction_metrics"]
    print(
        "Extraction: "
        f"{extraction['successful_extractions']} success, "
        f"{extraction['failed_extractions']} failed, "
        f"scalar_accuracy={extraction['scalar_accuracy_overall']}"
    )
    for model, summary in report["summary"]["model_generation_metrics"]["model_summaries"].items():
        print(
            f"{model}: {summary['generation_success_count']} success, "
            f"{summary['failed_generation_count']} failed, "
            "raw_structure_valid="
            f"{summary['raw_model_output']['structure_valid_rate']}, "
            "finalized_structure_valid="
            f"{summary['finalized_production_output']['structure_valid_rate']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models on all indexed PDF/GroundTruth pairs"
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, help="Maximum indexed documents to benchmark")
    parser.add_argument(
        "--documents",
        nargs="+",
        metavar="DOCUMENT_ID",
        help="Benchmark only these indexed document IDs, in the supplied order",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark"))
    parser.add_argument(
        "--progress-interval",
        type=parse_progress_interval,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        metavar="SECONDS",
        help="Seconds between generation heartbeat messages (default: 10)",
    )
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if arguments.limit is not None and arguments.documents is not None:
        parser.error("--limit cannot be used together with --documents")
    if len(arguments.models) != len(set(arguments.models)):
        parser.error("--models must not contain duplicates")

    try:
        report = run_benchmark(
            arguments.data_dir,
            arguments.output_dir,
            arguments.models,
            limit=arguments.limit,
            document_ids=arguments.documents,
            progress_interval=arguments.progress_interval,
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot start benchmark: {exc}", file=sys.stderr)
        return 1
    _print_summary(report)
    print(f"Results: {(arguments.output_dir / 'benchmark_results.csv').resolve()}")
    print(f"Summary: {(arguments.output_dir / 'benchmark_summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
