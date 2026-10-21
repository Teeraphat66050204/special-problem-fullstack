"""Project-summary smoke benchmark tests with fake generation only."""

import json
from dataclasses import replace

import pytest

from app.config import Settings
from app.datasets.groundtruth import DatasetEntry, GroundTruthRecord
from app.services.llm_service import (
    OllamaErrorDiagnostics,
    OllamaGenerationResult,
    OllamaRuntimeError,
)
from app.services.pdf_extractor import TextProvenance
from app.services.wiki_evidence import WikiSourceEvidence
from app.services.wiki_source import SourceDetection, WikiSourcePage, WikiSourceSelection
from scripts import benchmark_abstract_models as benchmark


def valid_summary() -> str:
    sentence = (
        "โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยเพื่อแก้ปัญหาการค้นหาข้อมูล "
        "โดยศึกษาการสกัดข้อความและใช้ Python สร้างต้นแบบสำหรับเอกสารโครงงาน"
    )
    return " ".join(sentence for _ in range(4))


def prepared() -> benchmark.PreparedProjectDocument:
    selection = WikiSourceSelection(
        source_text="Source page 4 (Thai abstract):\nโครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทย",
        pages=(WikiSourcePage(4, "Thai abstract", "บทคัดย่อ", TextProvenance.PYMUPDF),),
        detection=SourceDetection(abstract_pages=(4,), chapter_method="native"),
    )
    return benchmark.PreparedProjectDocument(
        entry=DatasetEntry(
            document_id="document_071",
            pdf="sample/document_071.pdf",
            groundtruth="groundtruth/document_071.json",
        ),
        ground_truth=GroundTruthRecord(
            document_id="document_071",
            abstract_th="โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทย",
        ),
        selection=selection,
        evidence=WikiSourceEvidence(
            title="ระบบค้นคืนเอกสารภาษาไทย",
            title_en="Thai Document Retrieval System",
            students=("สมชาย ใจดี",),
            student_ids=("63050001",),
            advisor="ดร. ตัวอย่าง อาจารย์",
            academic_year="2566",
            keywords=(),
            department="เทคโนโลยีสารสนเทศ",
        ),
        chapter_ocr=benchmark.ChapterOcrDiagnostics("already_detected", "native", 16, 19),
        prompt="project summary prompt sentinel",
        preparation_seconds=0.1,
    )


def settings() -> Settings:
    return Settings(
        _env_file=None,
        ocr_provider="disabled",
        ollama_base_url="http://ollama.invalid:11434",
    )


def prepared_for(document_id: str) -> benchmark.PreparedProjectDocument:
    base = prepared()
    return replace(
        base,
        entry=base.entry.model_copy(
            update={
                "document_id": document_id,
                "pdf": f"sample/{document_id}.pdf",
                "groundtruth": f"groundtruth/{document_id}.json",
            }
        ),
        ground_truth=base.ground_truth.model_copy(update={"document_id": document_id}),
    )


class ValidClient:
    def __init__(self, configured=None):
        self.configured = configured

    def generate_result(self, prompt, *, think=None, num_predict=None):
        return OllamaGenerationResult(
            response=valid_summary(),
            done=True,
            done_reason="stop",
            prompt_eval_count=100,
            eval_count=50,
            eval_duration=2_000_000_000,
        )


def test_fixed_model_set_and_thinking_policy():
    assert benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS == (
        "qwen3:8b",
        "hf.co/mradermacher/typhoon2-qwen2.5-7b-instruct-GGUF:Q4_K_M",
        "hf.co/openthaigpt/openthaigpt1.5-7b-instruct-GGUF:Q4_K_M",
    )
    assert benchmark._think_value(benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[0]) is False
    assert all(
        benchmark._think_value(model) is None
        for model in benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[1:]
    )
    assert (
        tuple(
            f"document_{identifier}"
            for identifier in (
                "007",
                "009",
                "022",
                "064",
                "071",
                "114",
                "123",
                "127",
                "137",
                "191",
                "213",
                "222",
                "251",
                "253",
                "254",
                "266",
                "283",
                "295",
                "335",
                "339",
            )
        )
        == benchmark.FULL_BENCHMARK_DOCUMENTS
    )


def test_smoke_runs_exactly_one_document_by_three_models(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "prepare_smoke_document", lambda *args: prepared())
    calls = []

    class FakeClient:
        def __init__(self, configured):
            self.configured = configured

        def generate_result(self, prompt, *, think=None, num_predict=None):
            calls.append(
                (
                    self.configured.ollama_model,
                    prompt,
                    think,
                    self.configured.ollama_temperature,
                    num_predict,
                )
            )
            return OllamaGenerationResult(
                response=valid_summary(),
                done=True,
                done_reason="stop",
                total_duration=3_000_000_000,
                load_duration=100_000_000,
                eval_count=20,
                eval_duration=2_000_000_000,
                context=(7, 8, 9),
            )

    output = tmp_path / "new-abstract-smoke"
    summary = benchmark.run_smoke_benchmark(
        tmp_path / "data",
        output,
        settings=settings(),
        client_factory=FakeClient,
    )
    assert len(calls) == 3
    assert {call[0] for call in calls} == set(benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS)
    assert [call[2] for call in calls] == [False, None, None]
    assert all(call[1] == "project summary prompt sentinel" for call in calls)
    assert all(call[3] == 0 for call in calls)
    assert all(call[4] == 800 for call in calls)
    assert summary["mode"] == "smoke_1_document_x_3_models"
    assert summary["total_runs"] == summary["successful_runs"] == 3
    assert len(list((output / "results").glob("*.json"))) == 3
    source = json.loads((output / "source.json").read_text(encoding="utf-8"))
    assert source["reference_usage"] == "evaluation_only_never_added_to_prompt"
    assert source["prompt"] == "project summary prompt sentinel"
    assert source["metadata"]["students"] == ["สมชาย ใจดี"]
    first_result = json.loads(next((output / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert first_result["raw_model_output"] == valid_summary()
    assert first_result["generated_summary"] == valid_summary()
    assert first_result["metadata"] == source["metadata"]
    assert first_result["performance"]["done"] is True
    assert first_result["performance"]["done_reason"] == "stop"
    assert first_result["performance"]["total_duration"] == 3_000_000_000
    assert first_result["performance"]["load_duration"] == 100_000_000
    assert first_result["performance"]["context_present"] is True
    assert first_result["performance"]["context_length"] == 3
    assert "context" not in first_result["performance"]


def test_source_artifact_deduplicates_summary_pages_without_dropping_sections():
    base = prepared()
    pages = (
        WikiSourcePage(4, "Thai abstract", "บทคัดย่อ", TextProvenance.PYMUPDF),
        WikiSourcePage(5, "Thai abstract", "ผลการทดลอง", TextProvenance.PYMUPDF),
        WikiSourcePage(16, "Chapter 1", "1.1 ความเป็นมา", TextProvenance.PYMUPDF),
        WikiSourcePage(17, "Chapter 1", "1.2 วัตถุประสงค์", TextProvenance.PYMUPDF),
        WikiSourcePage(17, "Chapter 1", "1.3 ขอบเขต", TextProvenance.PYMUPDF),
        WikiSourcePage(17, "Chapter 1", "1.4 ประโยชน์", TextProvenance.PYMUPDF),
        WikiSourcePage(18, "Chapter 1", "เนื้อหาต่อเนื่อง", TextProvenance.PYMUPDF),
    )
    selection = WikiSourceSelection(
        source_text="unused metadata evidence",
        pages=pages,
        detection=base.selection.detection,
    )

    artifact = benchmark._source_artifact(replace(base, selection=selection))

    assert artifact["summary_selected_pages"] == [4, 5, 16, 17, 18]
    assert "1.2 วัตถุประสงค์" in artifact["focused_source"]
    assert "1.3 ขอบเขต" in artifact["focused_source"]
    assert "1.4 ประโยชน์" in artifact["focused_source"]


def test_invalid_project_summary_and_model_failure_are_recorded():
    class InvalidClient:
        def generate_result(self, prompt, *, think=None, num_predict=None):
            return OllamaGenerationResult(response="ชื่อโครงงาน")

    invalid = benchmark.run_model(prepared(), "qwen3:8b", settings(), InvalidClient())
    assert invalid["status"] == "failure"
    assert invalid["failure_category"] == "project_summary_validation_failed"
    assert invalid["validation"]["is_valid"] is False
    assert invalid["raw_model_output"] == "ชื่อโครงงาน"

    class FailedClient:
        def generate_result(self, prompt, *, think=None, num_predict=None):
            raise OllamaRuntimeError(
                "offline",
                diagnostics=OllamaErrorDiagnostics(
                    model="qwen3:8b",
                    elapsed_seconds=3.5,
                    status_code=None,
                    response_body=None,
                    exception_type="ConnectError",
                    exception_message="connection reset by peer",
                    failure_kind="reset",
                    reset_related=True,
                ),
            )

    failed = benchmark.run_model(prepared(), "qwen3:8b", settings(), FailedClient())
    assert failed["failure_category"] == "llm_generation_failed"
    assert failed["error_type"] == "OllamaRuntimeError"
    assert failed["error_diagnostics"]["model"] == "qwen3:8b"
    assert failed["error_diagnostics"]["failure_kind"] == "reset"


@pytest.mark.parametrize(
    ("done_reason", "expected_status", "expected_category"),
    [
        ("stop", "success", None),
        (None, "success", None),
        ("length", "failure", "project_summary_truncated"),
    ],
)
def test_completion_reason_controls_truncation_status(
    done_reason, expected_status, expected_category
):
    raw = valid_summary()

    class FakeClient:
        def generate_result(self, prompt, *, think=None, num_predict=None):
            return OllamaGenerationResult(
                response=raw,
                done=True,
                done_reason=done_reason,
                eval_count=800,
            )

    result = benchmark.run_model(prepared(), "qwen3:8b", settings(), FakeClient())

    assert result["status"] == expected_status
    assert result["failure_category"] == expected_category
    assert result["raw_model_output"] == raw
    assert result["performance"]["done_reason"] == done_reason
    if done_reason == "length":
        assert result["validation"]["is_valid"] is False
        assert "summary_truncated" in result["validation"]["issues"]


def test_smoke_summary_counts_length_completion_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "prepare_smoke_document", lambda *args: prepared())

    class FakeClient:
        def __init__(self, configured):
            self.model = configured.ollama_model

        def generate_result(self, prompt, *, think=None, num_predict=None):
            reason = (
                "length" if self.model == benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[0] else "stop"
            )
            return OllamaGenerationResult(response=valid_summary(), done=True, done_reason=reason)

    summary = benchmark.run_smoke_benchmark(
        tmp_path / "data",
        tmp_path / "length-summary",
        settings=settings(),
        client_factory=FakeClient,
    )

    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1


def test_summary_handles_model_result_with_no_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "prepare_smoke_document", lambda *args: prepared())

    def fake_run_model(prepared_document, model, configured, client):
        if model == benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[0]:
            return {
                "model": model,
                "status": "failure",
                "failure_category": "llm_generation_failed",
                "validation": None,
                "performance": None,
            }
        return {
            "model": model,
            "status": "success",
            "failure_category": None,
            "validation": {"is_valid": True, "issues": []},
            "performance": {"generation_seconds": 1.0},
        }

    monkeypatch.setattr(benchmark, "run_model", fake_run_model)
    output = tmp_path / "nullable-validation-smoke"

    summary = benchmark.run_smoke_benchmark(
        tmp_path / "data",
        output,
        settings=settings(),
        client_factory=lambda configured: object(),
    )

    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["format_valid_runs"] == 2
    assert summary["results"][0]["validation"] is None
    assert summary["results"][0]["performance"] is None
    written = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert written == summary


def test_output_directory_must_be_new_before_preparation(tmp_path, monkeypatch):
    output = tmp_path / "historical-results"
    output.mkdir()
    marker = output / "keep.json"
    marker.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "prepare_smoke_document",
        lambda *args: pytest.fail("Preparation must not run for an existing output"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark.run_smoke_benchmark(tmp_path / "data", output, settings=settings())
    assert marker.read_text(encoding="utf-8") == "historical"


def test_full_resume_skips_completed_success_and_reconstructs_summary(tmp_path, monkeypatch):
    document = "document_007"
    models = benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[:2]
    monkeypatch.setattr(
        benchmark,
        "prepare_smoke_document",
        lambda data_dir, document_id, configured: prepared_for(document_id),
    )
    output = tmp_path / "full-resume"
    first = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        settings=settings(),
        client_factory=ValidClient,
        document_ids=(document,),
        models=(models[0],),
    )
    assert first["completed_runs"] == first["successful_runs"] == 1

    calls = 0

    class CountingClient(ValidClient):
        def __init__(self, configured=None):
            nonlocal calls
            calls += 1
            super().__init__(configured)

    resumed = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        resume=True,
        settings=settings(),
        client_factory=CountingClient,
        document_ids=(document,),
        models=models,
    )

    assert calls == 1
    assert resumed["completed_runs"] == resumed["successful_runs"] == 2
    assert resumed["skipped_runs"] == 1
    assert resumed["new_runs"] == 1
    assert len(resumed["runs"]) == 2


def test_full_resume_skips_failure_unless_rerun_is_requested(tmp_path, monkeypatch):
    document = "document_007"
    model = benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[0]
    monkeypatch.setattr(
        benchmark,
        "prepare_smoke_document",
        lambda data_dir, document_id, configured: prepared_for(document_id),
    )

    class InvalidClient:
        def __init__(self, configured=None):
            pass

        def generate_result(self, prompt, *, think=None, num_predict=None):
            return OllamaGenerationResult(response="สรุปสั้น")

    output = tmp_path / "failure-resume"
    failed = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        settings=settings(),
        client_factory=InvalidClient,
        document_ids=(document,),
        models=(model,),
    )
    assert failed["failed_runs"] == 1

    skipped = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        resume=True,
        settings=settings(),
        client_factory=lambda configured: pytest.fail("Completed failure must be skipped"),
        document_ids=(document,),
        models=(model,),
    )
    assert skipped["failed_runs"] == 1
    assert skipped["skipped_runs"] == 1

    rerun = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        resume=True,
        rerun_failures=True,
        settings=settings(),
        client_factory=ValidClient,
        document_ids=(document,),
        models=(model,),
    )
    assert rerun["successful_runs"] == 1
    assert rerun["failed_runs"] == 0
    assert rerun["new_runs"] == 1


def test_resume_reruns_corrupt_wrong_and_missing_artifacts(tmp_path, monkeypatch):
    document = "document_007"
    models = benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS
    monkeypatch.setattr(
        benchmark,
        "prepare_smoke_document",
        lambda data_dir, document_id, configured: prepared_for(document_id),
    )
    output = tmp_path / "repair-resume"
    for model in models:
        benchmark._result_path(output, document, model).parent.mkdir(parents=True, exist_ok=True)
    benchmark._result_path(output, document, models[0]).write_text("{broken", encoding="utf-8")
    wrong = benchmark.run_model(prepared_for(document), models[1], settings(), ValidClient())
    wrong["pipeline"] = "wrong_pipeline"
    benchmark._write_json_atomic(
        benchmark._result_path(output, document, models[1]),
        wrong,
        replace_existing=False,
    )

    calls = 0

    class CountingClient(ValidClient):
        def __init__(self, configured=None):
            nonlocal calls
            calls += 1
            super().__init__(configured)

    summary = benchmark.run_full_benchmark(
        tmp_path / "data",
        output,
        resume=True,
        settings=settings(),
        client_factory=CountingClient,
        document_ids=(document,),
        models=models,
    )

    assert calls == 3
    assert summary["completed_runs"] == summary["successful_runs"] == 3
    assert summary["pending_runs"] == 0
    assert summary["prompt_eval_count"] == 300
    assert summary["eval_count"] == 150
    assert summary["done_reason_counts"] == {"stop": 3}


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("artifact_type", "wrong_type"),
        ("document_id", "document_999"),
        ("model", "wrong-model"),
        ("pipeline", "wrong_pipeline"),
    ],
)
def test_completed_artifact_identity_must_match_planned_run(field, wrong_value):
    document = "document_007"
    model = benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[0]
    artifact = benchmark.run_model(prepared_for(document), model, settings(), ValidClient())
    artifact[field] = wrong_value

    assert not benchmark._complete_result_artifact(artifact, document, model)


def test_checkpoint_survives_later_client_factory_exception(tmp_path, monkeypatch):
    document = "document_007"
    models = benchmark.PROJECT_SUMMARY_BENCHMARK_MODELS[:2]
    monkeypatch.setattr(
        benchmark,
        "prepare_smoke_document",
        lambda data_dir, document_id, configured: prepared_for(document_id),
    )
    calls = 0

    def client_factory(configured):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated later interruption")
        return ValidClient(configured)

    output = tmp_path / "interrupted-full"
    with pytest.raises(RuntimeError, match="later interruption"):
        benchmark.run_full_benchmark(
            tmp_path / "data",
            output,
            settings=settings(),
            client_factory=client_factory,
            document_ids=(document,),
            models=models,
        )

    first_path = benchmark._result_path(output, document, models[0])
    assert benchmark._complete_result_artifact(
        benchmark._read_json_object(first_path), document, models[0]
    )
    checkpoint = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert checkpoint["completed_runs"] == 1
    assert checkpoint["pending_runs"] == 1


def test_cli_exposes_no_full_benchmark_or_model_override(capsys):
    with pytest.raises(SystemExit) as failure:
        benchmark.main(["--output-dir", "new", "--models", "other", "--limit", "20"])
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "unrecognized arguments" in error


def test_reference_similarity_is_explicitly_only_a_heuristic():
    result = benchmark._normalized_similarity("ระบบ ค้นคืน เอกสาร", "ระบบค้นคืนเอกสาร")
    assert result == 1.0
    assert benchmark._normalized_similarity("ข้อความ", None) is None
