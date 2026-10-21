"""Ollama Wiki generation tests using fake HTTP responses only."""

import httpx
import pytest

from app.config import Settings
from app.prompts.project_summary import build_project_summary_prompt
from app.services import llm_service


def valid_thai_markdown() -> str:
    return "โครงงานนี้พัฒนาระบบค้นคืนข้อมูลภาษาไทยเพื่อสนับสนุนการค้นหาเอกสาร"


def valid_project_summary() -> str:
    return " ".join(valid_thai_markdown() for _ in range(6))


def fake_response(status_code: int = 200, body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body if body is not None else {"response": valid_thai_markdown()},
        request=httpx.Request("POST", "http://ollama.test:11434/api/generate"),
    )


@pytest.fixture
def settings(monkeypatch) -> Settings:
    configured = Settings(
        _env_file=None,
        ollama_base_url="http://ollama.test:11434/",
        ollama_model="qwen2.5:7b-instruct",
        ollama_timeout_seconds=12.5,
        ollama_temperature=0.3,
    )
    monkeypatch.setattr(llm_service, "get_settings", lambda: configured)
    return configured


def test_successful_generation_sends_configured_model_and_existing_prompt(
    settings, monkeypatch
) -> None:
    captured: dict = {}

    def post(url: str, *, json: dict, timeout: float | None) -> httpx.Response:
        captured.update(url=url, payload=json, timeout=timeout)
        return fake_response(body={"response": valid_project_summary()})

    monkeypatch.setattr(llm_service.httpx, "post", post)
    source_text = "ชื่อโครงการ: ระบบค้นคืนข้อมูล"

    markdown = llm_service.generate_wiki(source_text)

    assert markdown == valid_project_summary()
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["payload"] == {
        "model": settings.ollama_model,
        "prompt": build_project_summary_prompt(source_text),
        "stream": False,
        "options": {
            "temperature": settings.ollama_temperature,
            "num_predict": settings.ollama_num_predict,
        },
    }
    assert captured["timeout"] == settings.ollama_timeout_seconds


def test_zero_ollama_timeout_disables_http_timeout(settings, monkeypatch) -> None:
    configured = Settings(
        _env_file=None,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        ollama_timeout_seconds=0,
        ollama_temperature=settings.ollama_temperature,
    )
    captured: dict = {}

    def post(url: str, *, json: dict, timeout: float | None) -> httpx.Response:
        captured.update(url=url, payload=json, timeout=timeout)
        return fake_response()

    monkeypatch.setattr(llm_service.httpx, "post", post)

    result = llm_service.OllamaClient(configured).generate("prompt")

    assert result == valid_thai_markdown()
    assert configured.ollama_timeout_seconds == 0
    assert captured["timeout"] is None


def test_generate_wiki_calls_the_project_summary_prompt_builder(settings, monkeypatch) -> None:
    source_text = "Extracted source text"
    called_with: list[str] = []

    def build_prompt(text: str) -> str:
        called_with.append(text)
        return "sentinel prompt"

    def post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        del url, timeout
        assert json["prompt"] == "sentinel prompt"
        return fake_response(body={"response": valid_project_summary()})

    monkeypatch.setattr(llm_service, "build_project_summary_prompt", build_prompt)
    monkeypatch.setattr(llm_service.httpx, "post", post)

    assert llm_service.generate_wiki(source_text) == valid_project_summary()
    assert called_with == [source_text]


def test_detailed_generation_can_disable_thinking_and_retains_metrics(
    settings, monkeypatch
) -> None:
    captured: dict = {}

    def post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        del url, timeout
        captured.update(json)
        return fake_response(
            body={
                "response": valid_thai_markdown(),
                "done": True,
                "done_reason": "stop",
                "total_duration": 7_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 100,
                "prompt_eval_duration": 2_000_000_000,
                "eval_count": 50,
                "eval_duration": 4_000_000_000,
                "context": [11, 22, 33],
            }
        )

    monkeypatch.setattr(llm_service.httpx, "post", post)

    result = llm_service.OllamaClient(settings).generate_result("prompt", think=False)

    assert captured["think"] is False
    assert result.done is True
    assert result.done_reason == "stop"
    assert result.total_duration == 7_000_000_000
    assert result.load_duration == 500_000_000
    assert result.prompt_eval_count == 100
    assert result.prompt_eval_duration == 2_000_000_000
    assert result.eval_count == 50
    assert result.eval_duration == 4_000_000_000
    assert result.tokens_per_second == 12.5
    assert result.context == (11, 22, 33)


def test_detailed_generation_tolerates_missing_completion_fields(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(
            body={"response": valid_thai_markdown(), "done": True}
        ),
    )

    result = llm_service.OllamaClient(settings).generate_result("prompt")

    assert result.done is True
    assert result.done_reason is None
    assert result.total_duration is None
    assert result.context is None


def test_length_completion_is_rejected_without_retry(settings, monkeypatch) -> None:
    calls = 0
    raw = valid_project_summary()

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_response(body={"response": raw, "done": True, "done_reason": "length"})

    monkeypatch.setattr(llm_service.httpx, "post", post)

    with pytest.raises(llm_service.InvalidProjectSummaryError) as failure:
        llm_service.OllamaClient(settings).generate("prompt", num_predict=800)

    assert failure.value.issues == ("summary_truncated",)
    assert failure.value.raw_text == raw
    assert calls == 1


def test_ollama_timeout_is_mapped(settings, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        del args, kwargs
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(llm_service.httpx, "post", timeout)

    with pytest.raises(llm_service.OllamaTimeoutError, match="timed out") as failure:
        llm_service.generate_wiki("Source text")
    assert failure.value.diagnostics.model == settings.ollama_model
    assert failure.value.diagnostics.failure_kind == "timeout"
    assert failure.value.diagnostics.exception_type == "ReadTimeout"
    assert failure.value.diagnostics.elapsed_seconds >= 0


def test_ollama_connection_failure_is_mapped(settings, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        del args, kwargs
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm_service.httpx, "post", unavailable)

    with pytest.raises(llm_service.OllamaUnavailableError, match="not reachable") as failure:
        llm_service.generate_wiki("Source text")
    diagnostics = failure.value.diagnostics
    assert diagnostics.model == settings.ollama_model
    assert diagnostics.failure_kind == "connection"
    assert diagnostics.connection_related is True
    assert diagnostics.exception_type == "ConnectError"
    assert diagnostics.exception_message == "connection refused"


def test_missing_ollama_model_is_mapped(settings, monkeypatch) -> None:
    monkeypatch.setattr(llm_service.httpx, "post", lambda *args, **kwargs: fake_response(404))

    with pytest.raises(llm_service.OllamaModelNotFoundError, match=settings.ollama_model):
        llm_service.generate_wiki("Source text")


@pytest.mark.parametrize("response_text", ["", "  \n"])
def test_empty_model_response_is_rejected(settings, monkeypatch, response_text: str) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(body={"response": response_text}),
    )

    with pytest.raises(llm_service.EmptyModelResponseError, match="no generated text"):
        llm_service.generate_wiki("Source text")


def test_markdown_heavy_summary_is_rejected(settings, monkeypatch) -> None:
    invalid_summary = "# Project\n## Methods\n" + valid_project_summary()
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(body={"response": invalid_summary}),
    )

    with pytest.raises(llm_service.InvalidAbstractError) as failure:
        llm_service.generate_wiki("Source text")

    assert "markdown_heavy_summary" in set(failure.value.issues)


def test_valid_thai_summary_is_returned_without_rewriting(settings, monkeypatch) -> None:
    thai_summary = valid_project_summary().replace("เอกสาร", "เอกสาร\u200b", 1)
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(body={"response": thai_summary}),
    )

    assert llm_service.generate_wiki("ข้อความต้นฉบับ") == thai_summary


@pytest.mark.parametrize("status_code", [400, 503])
def test_other_http_errors_are_mapped(settings, monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(status_code),
    )

    with pytest.raises(llm_service.OllamaRuntimeError, match=f"HTTP {status_code}"):
        llm_service.generate_wiki("Source text")


def test_http_error_diagnostics_preserve_status_and_response_body(settings, monkeypatch) -> None:
    response = httpx.Response(
        503,
        text="ollama worker crashed",
        request=httpx.Request("POST", "http://ollama.test:11434/api/generate"),
    )
    monkeypatch.setattr(llm_service.httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(llm_service.OllamaRuntimeError) as failure:
        llm_service.generate_wiki("Source text")

    diagnostics = failure.value.diagnostics
    assert diagnostics.model == settings.ollama_model
    assert diagnostics.status_code == 503
    assert diagnostics.response_body == "ollama worker crashed"
    assert diagnostics.exception_type == "HTTPStatusError"
    assert diagnostics.failure_kind == "http"
    assert diagnostics.runtime_related is True


@pytest.mark.parametrize(
    ("error", "kind", "protocol_related", "reset_related"),
    [
        (httpx.RemoteProtocolError("peer closed connection"), "protocol", True, False),
        (httpx.ReadError("connection reset by peer"), "reset", False, True),
        (RuntimeError("local runtime failure"), "runtime", False, False),
    ],
)
def test_protocol_reset_and_runtime_diagnostics(
    settings, monkeypatch, error, kind, protocol_related, reset_related
) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(llm_service.OllamaRuntimeError) as failure:
        llm_service.generate_wiki("Source text")

    diagnostics = failure.value.diagnostics
    assert diagnostics.model == settings.ollama_model
    assert diagnostics.failure_kind == kind
    assert diagnostics.protocol_related is protocol_related
    assert diagnostics.reset_related is reset_related
    assert diagnostics.exception_type == type(error).__name__
    assert diagnostics.exception_message == str(error)


def test_invalid_json_is_mapped(settings, monkeypatch) -> None:
    response = httpx.Response(
        200,
        content=b"not json",
        request=httpx.Request("POST", "http://ollama.test:11434/api/generate"),
    )
    monkeypatch.setattr(llm_service.httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(llm_service.OllamaRuntimeError, match="invalid JSON"):
        llm_service.generate_wiki("Source text")


def test_source_text_must_not_be_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HTTP should not be called")),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        llm_service.generate_wiki("  \n")


def test_ollama_settings_are_environment_backed(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11500")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.1")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "640")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11500"
    assert settings.ollama_model == "qwen2.5:3b-instruct"
    assert settings.ollama_timeout_seconds == 42
    assert settings.ollama_temperature == 0.1
    assert settings.ollama_num_predict == 640


def test_project_summary_cleanup_is_auditable_and_never_calls_legacy_finalizer(
    settings, monkeypatch
):
    raw = " \n" + valid_project_summary() + " \n"
    monkeypatch.setattr(
        llm_service.OllamaClient,
        "generate",
        lambda self, prompt, **kwargs: raw,
    )
    monkeypatch.setattr(
        llm_service,
        "refine_wiki_markdown",
        lambda *args: pytest.fail("Legacy finalizer must not run"),
    )
    result = llm_service.generate_project_summary_result("source")
    assert result.generated_summary == valid_project_summary()
    assert result.raw_text == raw
    assert result.cleanup_changes == ("boundary_whitespace_removed",)
