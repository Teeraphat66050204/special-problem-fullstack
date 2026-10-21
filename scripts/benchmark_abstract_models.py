"""Run smoke or resumable full benchmarks for the production project-summary pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app.config import Settings, get_settings
from app.datasets.groundtruth import (
    DatasetEntry,
    GroundTruthRecord,
    load_dataset_index,
    load_groundtruth,
)
from app.prompts.project_summary import (
    build_project_summary_prompt,
    normalize_project_summary,
    validate_project_summary,
)
from app.services.chapter_ocr import ChapterOcrDiagnostics, recover_chapter_one
from app.services.llm_service import LLMServiceError, OllamaClient, OllamaGenerationResult
from app.services.ocr_service import (
    OcrServiceError,
    add_ocr_failure_warning,
    apply_ocr_fallback,
    load_ocr_provider,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.wiki_evidence import WikiSourceEvidence, collect_wiki_source_evidence
from app.services.wiki_source import WikiSourceSelection, prepare_wiki_source

PROJECT_SUMMARY_BENCHMARK_MODELS: tuple[str, ...] = (
    "qwen3:8b",
    "hf.co/mradermacher/typhoon2-qwen2.5-7b-instruct-GGUF:Q4_K_M",
    "hf.co/openthaigpt/openthaigpt1.5-7b-instruct-GGUF:Q4_K_M",
)
DEFAULT_SMOKE_DOCUMENT = "document_071"
FULL_BENCHMARK_DOCUMENTS: tuple[str, ...] = (
    "document_007",
    "document_009",
    "document_022",
    "document_064",
    "document_071",
    "document_114",
    "document_123",
    "document_127",
    "document_137",
    "document_191",
    "document_213",
    "document_222",
    "document_251",
    "document_253",
    "document_254",
    "document_266",
    "document_283",
    "document_295",
    "document_335",
    "document_339",
)
PIPELINE_NAME = "production_project_summary_v1"
MODEL_RESULT_ARTIFACT_TYPE = "project_summary_smoke_model_result"
_SAFE_NAME = re.compile(r"[^a-z0-9]+")


class GenerationClient(Protocol):
    def generate_result(
        self,
        prompt: str,
        *,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> OllamaGenerationResult: ...


@dataclass(frozen=True, slots=True)
class PreparedProjectDocument:
    entry: DatasetEntry
    ground_truth: GroundTruthRecord
    selection: WikiSourceSelection
    evidence: WikiSourceEvidence
    chapter_ocr: ChapterOcrDiagnostics
    prompt: str
    preparation_seconds: float


def _write_new_json(path: Path, value: Any) -> None:
    """Create one artifact exactly once; benchmark history is never overwritten."""
    _write_json_atomic(path, value, replace_existing=False)


def _write_json_atomic(path: Path, value: Any, *, replace_existing: bool) -> None:
    """Publish complete JSON from the destination directory in one filesystem step."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(f"Artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_model_name(model: str) -> str:
    value = _SAFE_NAME.sub("-", model.casefold()).strip("-")
    if not value:
        raise ValueError(f"Model name has no filesystem-safe characters: {model!r}")
    return value


def _select_entry(data_dir: Path, document_id: str) -> DatasetEntry:
    index = load_dataset_index(data_dir / "index.json")
    matches = [entry for entry in index.documents if entry.document_id == document_id]
    if not matches:
        raise ValueError(f"Unknown smoke document: {document_id}")
    return matches[0]


def prepare_smoke_document(
    data_dir: Path,
    document_id: str,
    settings: Settings,
) -> PreparedProjectDocument:
    """Run the production source path once; GroundTruth never enters the prompt."""
    started = time.perf_counter()
    entry = _select_entry(data_dir, document_id)
    pdf_path = data_dir / entry.pdf
    ground_truth = load_groundtruth(data_dir / entry.groundtruth)
    with pdf_path.open("rb") as pdf_file:
        native = extract_pdf(pdf_file)
        provider = None
        try:
            provider = load_ocr_provider(settings)
            extraction = apply_ocr_fallback(
                pdf_file,
                native,
                provider,
                front_matter_page_limit=settings.ocr_max_pages,
            )
        except OcrServiceError as error:
            extraction = add_ocr_failure_warning(native, error)
        recovery = recover_chapter_one(
            pdf_file,
            extraction,
            settings,
            provider=provider,
            allow_ocr=provider is not None,
        )
    selection = prepare_wiki_source(recovery.extraction)
    evidence = collect_wiki_source_evidence(selection.source_text)
    prompt = build_project_summary_prompt(selection.summary_source_text)
    return PreparedProjectDocument(
        entry=entry,
        ground_truth=ground_truth,
        selection=selection,
        evidence=evidence,
        chapter_ocr=recovery.diagnostics,
        prompt=prompt,
        preparation_seconds=time.perf_counter() - started,
    )


def _normalized_similarity(left: str, right: str | None) -> float | None:
    """Transparent reference heuristic; not a factuality or Thai quality score."""
    if not right:
        return None
    normalized_left = re.sub(r"\s+", "", unicodedata.normalize("NFC", left)).casefold()
    normalized_right = re.sub(r"\s+", "", unicodedata.normalize("NFC", right)).casefold()
    return SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio()


def _think_value(model: str) -> bool | None:
    """Disable native thinking only for Qwen3; omit the field for Qwen2.5 models."""
    return False if model.casefold().startswith("qwen3:") else None


def _source_artifact(prepared: PreparedProjectDocument) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "project_summary_smoke_source",
        "document_id": prepared.entry.document_id,
        "selected_pages": list(prepared.selection.selected_pages),
        "source_detection": asdict(prepared.selection.detection),
        "chapter_ocr": asdict(prepared.chapter_ocr),
        "source_character_count": len(prepared.selection.summary_source_text),
        "focused_source": prepared.selection.summary_source_text,
        "summary_selected_pages": list(prepared.selection.summary_selected_pages),
        "metadata_evidence_source": prepared.selection.source_text,
        "metadata": prepared.evidence.project_metadata(),
        "metadata_warnings": [asdict(warning) for warning in prepared.evidence.warnings],
        "prompt": prepared.prompt,
        "prompt_sha256": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
        "preparation_seconds": prepared.preparation_seconds,
        "reference_abstract_th": prepared.ground_truth.abstract_th,
        "reference_usage": "evaluation_only_never_added_to_prompt",
    }


def run_model(
    prepared: PreparedProjectDocument,
    model: str,
    settings: Settings,
    client: GenerationClient,
) -> dict[str, Any]:
    """Generate one project result and record raw/final details without raising."""
    think = _think_value(model)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": MODEL_RESULT_ARTIFACT_TYPE,
        "pipeline": PIPELINE_NAME,
        "model": model,
        "document_id": prepared.entry.document_id,
        "settings": {
            "temperature": settings.ollama_temperature,
            "timeout_seconds": settings.ollama_timeout_seconds,
            "num_predict": settings.ollama_num_predict,
            "think": think,
            "stream": False,
        },
        "status": "failure",
        "failure_category": None,
        "error_type": None,
        "error": None,
        "error_diagnostics": None,
        "raw_model_output": None,
        "metadata": prepared.evidence.project_metadata(),
        "generated_summary": None,
        "cleanup_changes": [],
        "validation": None,
        "reference_similarity": None,
        "performance": {
            "generation_seconds": None,
            "done": None,
            "done_reason": None,
            "total_duration": None,
            "load_duration": None,
            "prompt_eval_count": None,
            "prompt_eval_duration": None,
            "eval_count": None,
            "eval_duration": None,
            "tokens_per_second": None,
            "context_present": False,
            "context_length": None,
        },
    }
    started = time.perf_counter()
    try:
        result = client.generate_result(
            prepared.prompt,
            think=think,
            num_predict=settings.ollama_num_predict,
        )
    except Exception as error:
        artifact["performance"]["generation_seconds"] = time.perf_counter() - started
        diagnostics = getattr(error, "diagnostics", None)
        artifact.update(
            failure_category=(
                "llm_generation_failed"
                if isinstance(error, LLMServiceError)
                else "unexpected_generation_failure"
            ),
            error_type=type(error).__name__,
            error=str(error),
            error_diagnostics=diagnostics.as_dict() if diagnostics is not None else None,
        )
        return artifact

    artifact["performance"].update(
        generation_seconds=time.perf_counter() - started,
        done=result.done,
        done_reason=result.done_reason,
        total_duration=result.total_duration,
        load_duration=result.load_duration,
        prompt_eval_count=result.prompt_eval_count,
        prompt_eval_duration=result.prompt_eval_duration,
        eval_count=result.eval_count,
        eval_duration=result.eval_duration,
        tokens_per_second=result.tokens_per_second,
        context_present=result.context is not None,
        context_length=len(result.context) if result.context is not None else None,
    )
    artifact["raw_model_output"] = result.response
    summary = normalize_project_summary(result.response)
    validation = validate_project_summary(summary)
    truncated = bool(result.done_reason and result.done_reason.casefold() == "length")
    validation_issues = list(validation.issues)
    if truncated:
        validation_issues.append("summary_truncated")
    artifact.update(
        generated_summary=summary,
        cleanup_changes=(["boundary_whitespace_removed"] if summary != result.response else []),
        validation={
            "is_valid": validation.is_valid and not truncated,
            "issues": validation_issues,
            "character_count": len(summary),
        },
        reference_similarity={
            "metric": "normalized_sequence_match_ratio",
            "scope": "reference_comparison_not_factuality",
            "value": _normalized_similarity(summary, prepared.ground_truth.abstract_th),
        },
    )
    if truncated:
        artifact.update(
            failure_category="project_summary_truncated",
            error="Ollama stopped generation because the output length limit was reached",
        )
        return artifact
    if not validation.is_valid:
        artifact.update(
            failure_category="project_summary_validation_failed",
            error="Generated output violates the project summary contract",
        )
        return artifact
    artifact["status"] = "success"
    return artifact


def run_smoke_benchmark(
    data_dir: Path,
    output_dir: Path,
    document_id: str = DEFAULT_SMOKE_DOCUMENT,
    *,
    settings: Settings | None = None,
    client_factory: Any = OllamaClient,
) -> dict[str, Any]:
    """Run exactly one document by the three fixed models into a new directory."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    settings = settings or get_settings()
    prepared = prepare_smoke_document(data_dir, document_id, settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    results_dir = output_dir / "results"
    results_dir.mkdir()
    _write_new_json(output_dir / "source.json", _source_artifact(prepared))

    results = []
    total = len(PROJECT_SUMMARY_BENCHMARK_MODELS)
    for index, model in enumerate(PROJECT_SUMMARY_BENCHMARK_MODELS, start=1):
        print(f"[{index}/{total}] {model} | {document_id}", flush=True)
        model_settings = settings.model_copy(
            update={"ollama_model": model, "ollama_temperature": 0.0}
        )
        result = run_model(prepared, model, model_settings, client_factory(model_settings))
        results.append(result)
        _write_new_json(results_dir / f"{_safe_model_name(model)}.json", result)

    summary = {
        "schema_version": 1,
        "artifact_type": "project_summary_smoke_summary",
        "pipeline": PIPELINE_NAME,
        "mode": "smoke_1_document_x_3_models",
        "document_id": document_id,
        "models": list(PROJECT_SUMMARY_BENCHMARK_MODELS),
        "total_runs": total,
        "successful_runs": sum(result.get("status") == "success" for result in results),
        "failed_runs": sum(result.get("status") != "success" for result in results),
        "format_valid_runs": sum(
            bool((result.get("validation") or {}).get("is_valid")) for result in results
        ),
        "results": [
            {
                "model": result.get("model"),
                "status": result.get("status"),
                "failure_category": result.get("failure_category"),
                "validation": result.get("validation"),
                "performance": result.get("performance"),
            }
            for result in results
        ],
    }
    _write_new_json(output_dir / "summary.json", summary)
    return summary


def _result_path(output_dir: Path, document_id: str, model: str) -> Path:
    return output_dir / "results" / document_id / f"{_safe_model_name(model)}.json"


def _source_path(output_dir: Path, document_id: str) -> Path:
    return output_dir / "sources" / f"{document_id}.json"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _complete_result_artifact(
    artifact: dict[str, Any] | None,
    document_id: str,
    model: str,
) -> bool:
    if artifact is None:
        return False
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != MODEL_RESULT_ARTIFACT_TYPE
        or artifact.get("pipeline") != PIPELINE_NAME
        or artifact.get("document_id") != document_id
        or artifact.get("model") != model
        or artifact.get("status") not in {"success", "failure"}
        or not isinstance(artifact.get("settings"), dict)
        or not isinstance(artifact.get("performance"), dict)
    ):
        return False
    if artifact["status"] == "success":
        validation = artifact.get("validation")
        return (
            isinstance(artifact.get("raw_model_output"), str)
            and isinstance(artifact.get("generated_summary"), str)
            and isinstance(validation, dict)
            and validation.get("is_valid") is True
        )
    return isinstance(artifact.get("failure_category"), str) and bool(artifact["failure_category"])


def _complete_source_artifact(
    artifact: dict[str, Any] | None,
    document_id: str,
) -> bool:
    return bool(
        artifact
        and artifact.get("schema_version") == 1
        and artifact.get("artifact_type") == "project_summary_smoke_source"
        and artifact.get("document_id") == document_id
        and isinstance(artifact.get("focused_source"), str)
        and isinstance(artifact.get("prompt"), str)
        and isinstance(artifact.get("metadata"), dict)
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_full_summary(
    output_dir: Path,
    *,
    document_ids: tuple[str, ...] = FULL_BENCHMARK_DOCUMENTS,
    models: tuple[str, ...] = PROJECT_SUMMARY_BENCHMARK_MODELS,
    skipped_runs: int = 0,
    new_runs: int = 0,
) -> dict[str, Any]:
    """Reconstruct the aggregate exclusively from complete per-run artifacts."""

    rows: list[dict[str, Any]] = []
    per_model: dict[str, dict[str, Any]] = {
        model: {
            "completed_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "generation_seconds": 0.0,
            "tokens_per_second_mean": None,
            "prompt_eval_count": 0,
            "eval_count": 0,
        }
        for model in models
    }
    model_speeds: dict[str, list[float]] = {model: [] for model in models}
    all_speeds: list[float] = []
    done_reasons: Counter[str] = Counter()
    generation_seconds = 0.0
    prompt_eval_count = 0
    eval_count = 0

    for document_id in document_ids:
        for model in models:
            artifact = _read_json_object(_result_path(output_dir, document_id, model))
            if not _complete_result_artifact(artifact, document_id, model):
                continue
            assert artifact is not None
            performance = artifact["performance"]
            validation = artifact.get("validation") or {}
            similarity = artifact.get("reference_similarity") or {}
            seconds = performance.get("generation_seconds")
            speed = performance.get("tokens_per_second")
            prompt_count = performance.get("prompt_eval_count")
            generated_count = performance.get("eval_count")
            done_reason = performance.get("done_reason")
            row = {
                "document_id": document_id,
                "model": model,
                "status": artifact["status"],
                "failure_category": artifact.get("failure_category"),
                "generation_seconds": seconds,
                "tokens_per_second": speed,
                "prompt_eval_count": prompt_count,
                "eval_count": generated_count,
                "done_reason": done_reason,
                "validation_character_count": validation.get("character_count"),
                "reference_similarity": similarity.get("value"),
            }
            rows.append(row)
            counts = per_model[model]
            counts["completed_runs"] += 1
            counts["successful_runs" if artifact["status"] == "success" else "failed_runs"] += 1
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                counts["generation_seconds"] += seconds
                generation_seconds += seconds
            if isinstance(speed, (int, float)) and not isinstance(speed, bool):
                model_speeds[model].append(float(speed))
                all_speeds.append(float(speed))
            if isinstance(prompt_count, int) and not isinstance(prompt_count, bool):
                counts["prompt_eval_count"] += prompt_count
                prompt_eval_count += prompt_count
            if isinstance(generated_count, int) and not isinstance(generated_count, bool):
                counts["eval_count"] += generated_count
                eval_count += generated_count
            if isinstance(done_reason, str):
                done_reasons[done_reason] += 1

    for model in models:
        per_model[model]["tokens_per_second_mean"] = _mean(model_speeds[model])
    completed = len(rows)
    successful = sum(row["status"] == "success" for row in rows)
    planned = len(document_ids) * len(models)
    return {
        "schema_version": 1,
        "artifact_type": "project_summary_full_summary",
        "pipeline": PIPELINE_NAME,
        "mode": "full_20_documents_x_3_models",
        "documents": list(document_ids),
        "models": list(models),
        "total_planned_runs": planned,
        "completed_runs": completed,
        "pending_runs": planned - completed,
        "successful_runs": successful,
        "failed_runs": completed - successful,
        "skipped_runs": skipped_runs,
        "resumed_runs": skipped_runs,
        "new_runs": new_runs,
        "generation_seconds": generation_seconds,
        "tokens_per_second_mean": _mean(all_speeds),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "done_reason_counts": dict(sorted(done_reasons.items())),
        "per_model": per_model,
        "runs": rows,
    }


def run_full_benchmark(
    data_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    rerun_failures: bool = False,
    settings: Settings | None = None,
    client_factory: Any = OllamaClient,
    document_ids: tuple[str, ...] = FULL_BENCHMARK_DOCUMENTS,
    models: tuple[str, ...] = PROJECT_SUMMARY_BENCHMARK_MODELS,
) -> dict[str, Any]:
    """Run the fixed benchmark sequentially with per-run atomic checkpoints."""

    if rerun_failures and not resume:
        raise ValueError("--rerun-failures requires --resume")
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "sources").mkdir(exist_ok=True)
    settings = settings or get_settings()
    total = len(document_ids) * len(models)
    skipped = 0
    executed = 0
    prepared_documents: dict[str, PreparedProjectDocument] = {}

    for index, (document_id, model) in enumerate(
        ((document, model) for document in document_ids for model in models), start=1
    ):
        path = _result_path(output_dir, document_id, model)
        existing = _read_json_object(path)
        complete = _complete_result_artifact(existing, document_id, model)
        rerun_failure = bool(
            complete and existing and existing.get("status") == "failure" and rerun_failures
        )
        if resume and complete and not rerun_failure:
            skipped += 1
            print(f"[{index}/{total}] {document_id} | {model} | SKIPPED (resume)", flush=True)
            continue

        prepared = prepared_documents.get(document_id)
        if prepared is None:
            prepared = prepare_smoke_document(data_dir, document_id, settings)
            prepared_documents[document_id] = prepared
            source_path = _source_path(output_dir, document_id)
            if not _complete_source_artifact(_read_json_object(source_path), document_id):
                _write_json_atomic(
                    source_path,
                    _source_artifact(prepared),
                    replace_existing=source_path.exists(),
                )

        print(f"[{index}/{total}] {document_id} | {model} | RUNNING", flush=True)
        model_settings = settings.model_copy(
            update={"ollama_model": model, "ollama_temperature": 0.0}
        )
        started = time.perf_counter()
        result = run_model(prepared, model, model_settings, client_factory(model_settings))
        elapsed = time.perf_counter() - started
        _write_json_atomic(path, result, replace_existing=path.exists())
        executed += 1
        summary = build_full_summary(
            output_dir,
            document_ids=document_ids,
            models=models,
            skipped_runs=skipped,
            new_runs=executed,
        )
        _write_json_atomic(output_dir / "summary.json", summary, replace_existing=True)
        state = "SUCCESS" if result["status"] == "success" else "FAILURE"
        print(
            f"[{index}/{total}] {document_id} | {model} | {state} | {elapsed:.1f}s",
            flush=True,
        )

    summary = build_full_summary(
        output_dir,
        document_ids=document_ids,
        models=models,
        skipped_runs=skipped,
        new_runs=executed,
    )
    _write_json_atomic(output_dir / "summary.json", summary, replace_existing=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the project-summary smoke or resumable full benchmark."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--document", default=DEFAULT_SMOKE_DOCUMENT)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the fixed 20-document x 3-model benchmark sequentially.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a full benchmark and skip complete success or failure artifacts.",
    )
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="With --resume, rerun complete failure artifacts while keeping successes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required directory; existing paths are accepted only with --full --resume.",
    )
    arguments = parser.parse_args(argv)
    if (arguments.resume or arguments.rerun_failures) and not arguments.full:
        parser.error("--resume and --rerun-failures require --full")
    try:
        if arguments.full:
            summary = run_full_benchmark(
                arguments.data_dir,
                arguments.output_dir,
                resume=arguments.resume,
                rerun_failures=arguments.rerun_failures,
            )
        else:
            summary = run_smoke_benchmark(
                arguments.data_dir,
                arguments.output_dir,
                arguments.document,
            )
    except (OSError, ValueError, PdfExtractionError) as error:
        print(f"Project-summary smoke benchmark failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_runs"] == 0 else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
