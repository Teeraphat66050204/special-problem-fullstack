"""Deterministic Wiki model benchmark harness tests without a live Ollama runtime."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.datasets.groundtruth import DatasetEntry, GroundTruthRecord
from app.prompts import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS
from app.services.llm_service import OllamaGenerationResult, OllamaRuntimeError
from app.services.wiki_evidence import WikiSourceEvidence
from scripts import benchmark_telemetry as telemetry
from scripts import benchmark_wiki_models as benchmark
from scripts.benchmark_telemetry import summarize_samples


def test_progress_lines_have_stable_human_readable_format() -> None:
    start = benchmark.format_run_start(3, 40, "qwen3:14b", "document_064")
    heartbeat = benchmark.format_generation_heartbeat(
        3,
        40,
        "qwen3:14b",
        "document_064",
        125.9,
    )
    completed = benchmark.format_run_result(
        3,
        40,
        "qwen3:14b",
        "document_064",
        {
            "status": "success",
            "failure_category": None,
            "performance": {
                "llm_generation_seconds": 18.7341,
                "total_seconds": 19.2264,
                "tokens_per_second": 22.5,
            },
        },
    )

    assert start == "[3/40] qwen3:14b | document_064 | START"
    assert heartbeat == ("[3/40] qwen3:14b | document_064 | elapsed=02:05 | generating...")
    assert "DONE status=success" in completed
    assert "generation=18.734s" in completed
    assert "total=19.226s" in completed
    assert "tokens/sec=22.500" in completed


def test_eta_uses_only_completed_runs_and_formats_finish_time() -> None:
    durations = [10.0, 20.0, 9_999.0]

    remaining = benchmark.estimate_remaining_seconds(
        durations,
        completed_runs=2,
        total_runs=5,
    )
    line = benchmark.format_benchmark_progress(
        completed_runs=2,
        total_runs=5,
        benchmark_elapsed=30.0,
        completed_durations=durations,
        now=datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC),
    )

    assert remaining == 45.0
    assert "2/5 completed" in line
    assert "elapsed=00:30" in line
    assert "avg=00:15" in line
    assert "ETA=00:45" in line
    assert "finish=2026-01-02 12:00:45" in line


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_progress_interval_rejects_non_positive_or_non_finite_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_progress_interval(value)


def test_progress_interval_parses_positive_seconds() -> None:
    assert benchmark.parse_progress_interval("10") == 10.0
    assert benchmark.parse_progress_interval("0.25") == 0.25


def test_eta_is_unknown_without_enough_completed_runs_or_with_empty_input() -> None:
    assert benchmark.estimate_remaining_seconds([], completed_runs=0, total_runs=40) is None
    assert benchmark.estimate_remaining_seconds([12.0], completed_runs=1, total_runs=40) is None
    line = benchmark.format_benchmark_progress(
        completed_runs=0,
        total_runs=0,
        benchmark_elapsed=0,
        completed_durations=[],
    )
    assert "avg=unknown" in line
    assert "ETA=unknown" in line
    assert "finish=unknown" in line


def test_normalization_is_unicode_safe_and_conservative() -> None:
    composed = "Caf\u00e9  PROJECT"
    decomposed = "  Cafe\u0301\n\tPROJECT  "

    assert benchmark.normalize_value(composed) == "Caf\u00e9 PROJECT"
    assert benchmark.normalize_value(decomposed) == "Caf\u00e9 PROJECT"
    assert benchmark.normalize_value(composed, case_insensitive=True) == "caf\u00e9 project"
    assert benchmark.normalize_value("project-title") != benchmark.normalize_value("project title")


def test_scalar_comparison_uses_exact_or_explicit_case_insensitive_matching() -> None:
    assert benchmark.compare_scalar("Qwen Wiki", "  Qwen\nWiki ")["match"] is True
    assert benchmark.compare_scalar("Qwen Wiki", "qwen wiki")["match"] is False
    assert (
        benchmark.compare_scalar("Qwen Wiki", "qwen wiki", case_insensitive=True)["match"] is True
    )
    assert benchmark.compare_scalar(None, "")["match"] is True


def test_multi_value_metrics_are_exact_set_metrics_after_normalization() -> None:
    metrics = benchmark.multi_value_metrics(
        ["Alice Smith", "BETA"],
        [" alice   smith ", "Gamma"],
        case_insensitive=True,
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert benchmark.multi_value_metrics([], [])["f1"] == 1.0


def _prepared(document_id: str, prompt: str) -> benchmark.PreparedDocument:
    entry = DatasetEntry(
        document_id=document_id,
        pdf=f"sample/{document_id}.pdf",
        groundtruth=f"groundtruth/{document_id}.json",
    )
    ground_truth = GroundTruthRecord(
        document_id=document_id,
        title_th="Project Title",
        students=["Alice Smith"],
        advisor="Advisor Name",
        academic_year=2566,
        keywords=["Qwen"],
        raw_text="Alice Smith 63050148",
    )
    evidence = WikiSourceEvidence(
        title="Project Title",
        title_en=None,
        students=("Alice Smith",),
        student_ids=("63050148",),
        advisor="Advisor Name",
        academic_year="2566",
        keywords=("Qwen",),
    )
    return benchmark.PreparedDocument(
        entry=entry,
        ground_truth=ground_truth,
        selected_pages=(1, 4),
        source_text="Source page 1 (title page):\nProject Title",
        evidence=evidence,
        prompt=prompt,
        preparation_seconds=0.25,
    )


def _entries(*document_ids: str) -> list[DatasetEntry]:
    return [
        DatasetEntry(
            document_id=document_id,
            pdf=f"sample/{document_id}.pdf",
            groundtruth=f"groundtruth/{document_id}.json",
        )
        for document_id in document_ids
    ]


def test_document_subset_preserves_requested_order_and_artifact_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = _entries("document_001", "document_002", "document_003")
    monkeypatch.setattr(
        benchmark,
        "load_dataset_index",
        lambda path: type("Index", (), {"documents": entries})(),
    )
    monkeypatch.setattr(
        benchmark,
        "prepare_dataset",
        lambda data_dir, selected: [
            _prepared(entry.document_id, f"prompt for {entry.document_id}") for entry in selected
        ],
    )

    class FakeClient:
        def generate_result(
            self, prompt: str, *, think: bool | None = None
        ) -> OllamaGenerationResult:
            del prompt
            assert think is False
            return OllamaGenerationResult(response=_valid_markdown())

    report = benchmark.run_benchmark(
        tmp_path / "data",
        tmp_path,
        ["qwen3:8b"],
        document_ids=["document_003", "document_001"],
        client_factory=lambda settings: FakeClient(),
    )

    assert report["summary"]["document_ids"] == ["document_003", "document_001"]
    saved_summary = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert saved_summary["document_ids"] == ["document_003", "document_001"]
    with (tmp_path / "benchmark_results.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert {row["document_id"] for row in rows} == {"document_003", "document_001"}
    assert all(row["document_id"] != "document_002" for row in rows)


def test_document_selection_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match=r"Unknown document ID\(s\): document_999"):
        benchmark.select_dataset_entries(
            _entries("document_001", "document_002"),
            document_ids=["document_999"],
        )


def test_document_selection_defaults_to_full_dataset_and_limit_still_applies() -> None:
    entries = _entries("document_001", "document_002", "document_003")

    assert benchmark.select_dataset_entries(entries) == entries
    assert benchmark.select_dataset_entries(entries, limit=2) == entries[:2]


def test_limit_and_documents_cli_options_conflict(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--limit", "1", "--documents", "document_001"])

    assert error.value.code == 2
    assert "--limit cannot be used together with --documents" in capsys.readouterr().err


def _generation_layer(*, scope: str, structure_valid: bool, unsupported_count: int) -> dict:
    return {
        "metric_scope": scope,
        "structure_valid": structure_valid,
        "canonical_section_count": 7,
        "missing_marker_section_count": 2 if unsupported_count == 0 else 0,
        "missing_canonical_sections": [] if structure_valid else ["## Missing"],
        "unsupported_claim_flag_count": unsupported_count,
        "unsupported_claim_flags": [
            {
                "detector": "out_of_source_lexical_indicator",
                "kind": "strengthened_claim",
                "value": "guaranteed",
            }
        ]
        * unsupported_count,
        "output_token_count": 20 if scope == "raw_model_output" else None,
        "output_character_count": 200,
    }


def _model_result(*, status: str, structure_valid: bool, unsupported_count: int) -> dict:
    return {
        "status": status,
        "failure_category": "validation_failed" if status == "failure" else None,
        "model_generation_metrics": {
            "raw_model_output": _generation_layer(
                scope="raw_model_output",
                structure_valid=structure_valid,
                unsupported_count=unsupported_count,
            ),
            "finalized_production_output": _generation_layer(
                scope="finalized_production_output",
                structure_valid=structure_valid,
                unsupported_count=unsupported_count,
            ),
        },
        "performance": {
            "llm_generation_seconds": 2.0,
            "total_seconds": 3.0,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 1_000,
            "eval_count": 20,
            "eval_duration": 2_000,
            "tokens_per_second": 10.0,
        },
    }


def test_aggregation_separates_extraction_from_model_generation() -> None:
    extraction = benchmark.extraction_artifact(_prepared("document_001", "prompt"))
    extraction_summary = benchmark.aggregate_extraction_results([extraction])
    model_summary = benchmark.aggregate_model_results(
        [
            _model_result(status="success", structure_valid=True, unsupported_count=0),
            _model_result(status="failure", structure_valid=False, unsupported_count=1),
        ]
    )

    assert extraction_summary["metric_scope"] == "extraction_pipeline"
    assert extraction_summary["scalar_field_accuracy"]["title"] == 1.0
    assert model_summary["generation_success_count"] == 1
    assert model_summary["failed_generation_count"] == 1
    assert model_summary["raw_model_output"]["structure_valid_rate"] == 0.5
    assert model_summary["raw_model_output"]["unsupported_claim_flag_count"] == 1
    assert model_summary["raw_model_output"]["missing_canonical_sections"] == {"## Missing": 1}
    assert model_summary["performance_mean"]["total_seconds"] == 3.0
    assert "scalar_field_accuracy" not in model_summary


def _valid_markdown(body: str = "Source-backed body") -> str:
    sections = "\n\n".join(f"{heading}\n{body}" for heading in REQUIRED_WIKI_HEADINGS)
    return f"# Project Title\n\n{sections}"


def _conservative_markdown() -> str:
    sections = "\n\n".join(
        f"{heading}\n{MISSING_INFORMATION_MARKER}" for heading in REQUIRED_WIKI_HEADINGS
    )
    return f"# Project Title\n\n{sections}"


def test_failed_run_is_saved_and_does_not_abort_following_documents(tmp_path: Path) -> None:
    calls: list[tuple[str, bool | None]] = []

    class FakeClient:
        def generate_result(
            self, prompt: str, *, think: bool | None = None
        ) -> OllamaGenerationResult:
            calls.append((prompt, think))
            if prompt == "fail prompt":
                raise OllamaRuntimeError("forced failure")
            return OllamaGenerationResult(
                response=_valid_markdown(),
                prompt_eval_count=10,
                prompt_eval_duration=1_000_000_000,
                eval_count=20,
                eval_duration=2_000_000_000,
            )

    report = benchmark.benchmark_prepared_documents(
        [_prepared("document_001", "fail prompt"), _prepared("document_002", "ok prompt")],
        ["qwen3:8b"],
        tmp_path,
        Settings(_env_file=None),
        client_factory=lambda settings: FakeClient(),
    )

    assert calls == [("fail prompt", False), ("ok prompt", False)]
    model_summary = report["summary"]["model_generation_metrics"]["model_summaries"]["qwen3:8b"]
    assert model_summary["failed_generation_count"] == 1
    assert model_summary["generation_success_count"] == 1

    failed = json.loads((tmp_path / "qwen3-8b" / "document_001.json").read_text(encoding="utf-8"))
    succeeded = json.loads(
        (tmp_path / "qwen3-8b" / "document_002.json").read_text(encoding="utf-8")
    )
    assert failed["failure_category"] == "llm_generation_failed"
    assert failed["ground_truth"]["document_id"] == "document_001"
    assert failed["raw_model_output"] is None
    assert succeeded["status"] == "success"
    assert succeeded["raw_model_output"] == _valid_markdown()
    assert succeeded["performance"]["tokens_per_second"] == 10.0
    assert (tmp_path / "extraction" / "document_001.json").is_file()
    assert (tmp_path / "benchmark_results.csv").is_file()
    assert (tmp_path / "benchmark_summary.json").is_file()


def test_document_007_like_outputs_separate_shared_extraction_from_model_quality(
    tmp_path: Path,
) -> None:
    outputs = {
        "qwen3:8b": _valid_markdown("This guaranteed system has 99 percent accuracy."),
        "qwen3:14b": _conservative_markdown(),
    }

    class FakeClient:
        def __init__(self, model: str) -> None:
            self.model = model

        def generate_result(
            self, prompt: str, *, think: bool | None = None
        ) -> OllamaGenerationResult:
            del prompt
            assert think is False
            token_count = 120 if self.model == "qwen3:8b" else 80
            return OllamaGenerationResult(
                response=outputs[self.model],
                eval_count=token_count,
                eval_duration=2_000_000_000,
            )

    prepared = _prepared("document_007", "identical prompt")
    report = benchmark.benchmark_prepared_documents(
        [prepared],
        ["qwen3:8b", "qwen3:14b"],
        tmp_path,
        Settings(_env_file=None),
        client_factory=lambda settings: FakeClient(settings.ollama_model),
    )

    extraction_summary = report["summary"]["deterministic_extraction_metrics"]
    model_summaries = report["summary"]["model_generation_metrics"]["model_summaries"]
    assert extraction_summary["total_documents"] == 1
    assert extraction_summary["scalar_field_accuracy"]["title"] == 1.0
    assert "scalar_field_accuracy" not in model_summaries["qwen3:8b"]
    assert "scalar_field_accuracy" not in model_summaries["qwen3:14b"]

    output_8b = json.loads(
        (tmp_path / "qwen3-8b" / "document_007.json").read_text(encoding="utf-8")
    )
    output_14b = json.loads(
        (tmp_path / "qwen3-14b" / "document_007.json").read_text(encoding="utf-8")
    )
    extraction = json.loads(
        (tmp_path / "extraction" / "document_007.json").read_text(encoding="utf-8")
    )
    metrics_8b = output_8b["model_generation_metrics"]["raw_model_output"]
    metrics_14b = output_14b["model_generation_metrics"]["raw_model_output"]

    assert extraction["metric_scope"] == "extraction_pipeline"
    assert extraction["actual_metadata"] == benchmark.evidence_metadata(prepared.evidence)
    assert "actual_metadata" not in output_8b
    assert "actual_metadata" not in output_14b
    assert metrics_8b["metric_scope"] == "raw_model_output"
    assert metrics_14b["metric_scope"] == "raw_model_output"
    assert metrics_8b["unsupported_claim_flag_count"] > 0
    assert metrics_14b["unsupported_claim_flag_count"] == 0
    assert metrics_8b["missing_marker_section_count"] == 0
    assert metrics_14b["missing_marker_section_count"] == len(REQUIRED_WIKI_HEADINGS)
    assert metrics_8b["output_token_count"] == 120
    assert metrics_14b["output_token_count"] == 80
    assert output_8b["raw_model_output"] != output_8b["finalized_markdown"]
    assert output_14b["raw_model_output"] != output_14b["finalized_markdown"]

    with (tmp_path / "benchmark_results.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    scopes = [row["metric_scope"] for row in rows]
    assert scopes.count("extraction_pipeline") == 1
    assert scopes.count("raw_model_output") == 2
    assert scopes.count("finalized_production_output") == 2


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.95, 4),
        (0.9499, 3),
        (0.80, 3),
        (0.7999, 2),
        (0.60, 2),
        (0.5999, 1),
        (None, None),
    ],
)
def test_standard_rate_rubric_threshold_boundaries(
    rate: float | None, expected: int | None
) -> None:
    assert (
        benchmark.rate_rubric_score(
            rate,
            score_4_at=0.95,
            score_3_at=0.80,
            score_2_at=0.60,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("thresholds", "rate", "expected"),
    [
        ((0.90, 0.75, 0.60), 0.90, 4),
        ((0.90, 0.75, 0.60), 0.8999, 3),
        ((0.90, 0.75, 0.60), 0.75, 3),
        ((0.90, 0.75, 0.60), 0.7499, 2),
        ((0.90, 0.75, 0.60), 0.60, 2),
        ((0.90, 0.75, 0.60), 0.5999, 1),
        ((0.95, 0.85, 0.70), 0.95, 4),
        ((0.95, 0.85, 0.70), 0.9499, 3),
        ((0.95, 0.85, 0.70), 0.85, 3),
        ((0.95, 0.85, 0.70), 0.8499, 2),
        ((0.95, 0.85, 0.70), 0.70, 2),
        ((0.95, 0.85, 0.70), 0.6999, 1),
    ],
)
def test_grounding_and_runtime_rate_threshold_boundaries(
    thresholds: tuple[float, float, float], rate: float, expected: int
) -> None:
    assert (
        benchmark.rate_rubric_score(
            rate,
            score_4_at=thresholds[0],
            score_3_at=thresholds[1],
            score_2_at=thresholds[2],
        )
        == expected
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(120, 4), (120.01, 3), (300, 3), (300.01, 2), (600, 2), (600.01, 1), (None, None)],
)
def test_latency_rubric_threshold_boundaries(seconds: float | None, expected: int | None) -> None:
    assert benchmark.latency_rubric_score(seconds) == expected


def test_runtime_reliability_and_robustness_have_distinct_semantics() -> None:
    success = _model_result(status="success", structure_valid=True, unsupported_count=0)
    validation_failure = _model_result(status="failure", structure_valid=False, unsupported_count=1)
    llm_failure = _model_result(status="failure", structure_valid=False, unsupported_count=0)
    llm_failure["failure_category"] = "llm_generation_failed"
    results = [success, validation_failure, llm_failure]
    for index, result in enumerate(results):
        result.update(model="qwen3:8b", document_id=f"document_{index:03d}")
    model_summary = benchmark.aggregate_model_results(results)

    rubric = benchmark.build_rubric_ready_metrics(
        {"qwen3:8b": results},
        {"qwen3:8b": model_summary},
        {"model_summaries": {}},
    )["qwen3:8b"]

    assert rubric["robustness_rate"] == pytest.approx(1 / 3)
    assert rubric["runtime_reliability_rate"] == pytest.approx(2 / 3)


def test_unsupported_claim_review_pass_rate_uses_zero_flag_outputs() -> None:
    summary = benchmark.aggregate_model_results(
        [
            _model_result(status="success", structure_valid=True, unsupported_count=0),
            _model_result(status="success", structure_valid=True, unsupported_count=2),
            _model_result(status="success", structure_valid=True, unsupported_count=0),
        ]
    )

    finalized = summary["finalized_production_output"]
    assert finalized["outputs_without_unsupported_claim_flags"] == 2
    assert finalized["unsupported_claim_review_pass_rate"] == pytest.approx(2 / 3)


def test_total_latency_median_and_interpolated_p95_are_correct() -> None:
    results = []
    for total_seconds in (1.0, 2.0, 3.0, 4.0):
        result = _model_result(status="success", structure_valid=True, unsupported_count=0)
        result["performance"]["total_seconds"] = total_seconds
        results.append(result)

    distribution = benchmark.aggregate_model_results(results)["performance_distribution"]

    assert distribution["median_total_seconds"] == 2.5
    assert distribution["p95_total_seconds"] == pytest.approx(3.85)


def test_missing_telemetry_remains_null_and_unavailable() -> None:
    telemetry = summarize_samples([])
    result = _model_result(status="success", structure_valid=True, unsupported_count=0)
    result["performance"]["tokens_per_second"] = None
    result["performance"]["resource_telemetry"] = telemetry

    resources = benchmark.aggregate_model_results([result])["resource_metrics"]

    assert resources["mean_cpu_percent"] is None
    assert resources["mean_gpu_percent"] is None
    assert resources["resource_efficiency_score_status"] == "unavailable"


def test_gpu_failure_does_not_disable_psutil_cpu_and_ram_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def cpu_percent(self, interval: float | None = None) -> float:
            del interval
            return 17.5

        def memory_info(self) -> object:
            return type("MemoryInfo", (), {"rss": 64 * telemetry.MEBIBYTE})()

    class FakePsutil:
        @staticmethod
        def Process() -> FakeProcess:
            return FakeProcess()

        @staticmethod
        def cpu_percent(interval: float | None = None) -> float:
            del interval
            return 42.0

        @staticmethod
        def virtual_memory() -> object:
            return type("VirtualMemory", (), {"used": 8_192 * telemetry.MEBIBYTE})()

    def failed_gpu_read() -> tuple[float | None, float | None]:
        raise RuntimeError("GPU telemetry unavailable")

    monkeypatch.setattr(telemetry, "psutil", FakePsutil())
    monkeypatch.setattr(
        telemetry,
        "detect_gpu_reader",
        lambda: ("amd-smi", failed_gpu_read),
    )
    sampler = telemetry.ResourceTelemetrySampler(interval_seconds=60)

    sampler.start()
    sampler._sample()
    measured = sampler.stop()

    assert measured["system_cpu_mean_percent"] == 42.0
    assert measured["process_cpu_mean_percent"] == 17.5
    assert measured["ram_mean_mb"] == 8_192
    assert measured["process_rss_mean_mb"] == 64
    assert measured["gpu_mean_percent"] is None
    assert measured["vram_mean_mb"] is None
    assert measured["status"] == "cpu_ram_available_gpu_unavailable"

    result = _model_result(status="success", structure_valid=True, unsupported_count=0)
    result["performance"]["resource_telemetry"] = measured
    resources = benchmark.aggregate_model_results([result])["resource_metrics"]
    assert resources["mean_cpu_percent"] == 42.0
    assert resources["mean_ram_mb"] == 8_192
    assert resources["resource_efficiency_score_status"] == "partial"


def test_human_csv_rejects_scores_outside_integer_one_to_four(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_human_evaluation.csv"
    path.write_text(
        ",".join(benchmark.HUMAN_EVALUATION_FIELDS) + "\nqwen3:8b,document_001,5,,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be an integer 1-4"):
        benchmark.read_human_evaluations(path)


def test_human_csv_preserves_existing_ratings_on_subsequent_runs(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_human_evaluation.csv"
    existing = {
        "model": "qwen3:8b",
        "document_id": "document_001",
        "content_relevance_score": "4",
        "content_relevance_notes": "Relevant",
        "fluency_readability_score": "3",
        "fluency_readability_notes": "Readable",
        "summary_quality_consistency_score": "",
        "summary_quality_consistency_notes": "Pending",
    }
    benchmark._write_csv(path, [existing])
    result = _model_result(status="success", structure_valid=True, unsupported_count=0)
    result.update(model="qwen3:8b", document_id="document_001")

    rows = benchmark.write_human_evaluation_csv(path, [result])

    assert rows == [existing]
    assert benchmark.read_human_evaluations(path) == [existing]


def test_human_evaluation_aggregation_reports_means_and_missing_counts() -> None:
    rows = [
        {
            "model": "qwen3:8b",
            "document_id": "document_001",
            "content_relevance_score": "4",
            "content_relevance_notes": "",
            "fluency_readability_score": "3",
            "fluency_readability_notes": "",
            "summary_quality_consistency_score": "",
            "summary_quality_consistency_notes": "",
        },
        {
            "model": "qwen3:8b",
            "document_id": "document_002",
            "content_relevance_score": "2",
            "content_relevance_notes": "",
            "fluency_readability_score": "",
            "fluency_readability_notes": "",
            "summary_quality_consistency_score": "1",
            "summary_quality_consistency_notes": "",
        },
    ]

    summary = benchmark.aggregate_human_evaluations(rows)["model_summaries"]["qwen3:8b"]

    assert summary["criteria"]["content_relevance"] == {
        "mean": 3.0,
        "rated_documents": 2,
        "missing_ratings": 0,
    }
    assert summary["criteria"]["fluency_readability"]["missing_ratings"] == 1
    assert summary["overall_mean_human_score"] == 2.5


def test_empty_results_do_not_divide_by_zero_or_fabricate_scores() -> None:
    model_summary = benchmark.aggregate_model_results([])
    rubric = benchmark.build_rubric_ready_metrics(
        {"qwen3:8b": []},
        {"qwen3:8b": model_summary},
        {"model_summaries": {}},
    )["qwen3:8b"]

    assert rubric["instruction_following_rate"] is None
    assert rubric["robustness_rate"] is None
    assert rubric["runtime_reliability_rate"] is None
    assert rubric["latency_rubric_score"] is None


def test_benchmark_summary_contains_complete_rubric_schema(tmp_path: Path) -> None:
    class FakeClient:
        def generate_result(
            self, prompt: str, *, think: bool | None = None
        ) -> OllamaGenerationResult:
            del prompt
            assert think is False
            return OllamaGenerationResult(response=_valid_markdown())

    report = benchmark.benchmark_prepared_documents(
        [_prepared("document_001", "prompt")],
        ["qwen3:8b"],
        tmp_path,
        Settings(_env_file=None),
        client_factory=lambda settings: FakeClient(),
    )

    summary = report["summary"]
    rubric = summary["rubric_ready_metrics"]["qwen3:8b"]
    assert summary["rubric_metadata"]["human_criteria"] == [4, 5, 6]
    assert rubric["instruction_following_rubric_score"] == 4
    assert rubric["structure_compliance_rubric_score"] == 4
    assert rubric["content_relevance_mean"] is None
    assert "resource_metrics" in rubric
    assert (tmp_path / "benchmark_human_evaluation.csv").is_file()
    with (tmp_path / "benchmark_results.csv").open(encoding="utf-8", newline="") as source:
        fieldnames = csv.DictReader(source).fieldnames or []
    assert {
        "raw_structure_valid",
        "finalized_structure_valid",
        "unsupported_claim_review_pass",
        "llm_generation_failed",
        "validation_failed",
        "cpu_mean_percent",
        "gpu_mean_percent",
        "vram_peak_mb",
    }.issubset(fieldnames)
