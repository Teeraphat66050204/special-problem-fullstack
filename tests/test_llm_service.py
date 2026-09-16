"""Ollama Wiki generation tests using fake HTTP responses only."""

import httpx
import pytest

from app.config import Settings
from app.prompts import REQUIRED_WIKI_HEADINGS, WikiStructureIssueCode, build_wiki_generation_prompt
from app.services import llm_service


def valid_thai_markdown() -> str:
    sections = "\n\n".join(f"{heading}\nเนื้อหาจากเอกสารต้นฉบับ" for heading in REQUIRED_WIKI_HEADINGS)
    return f"# ระบบค้นคืนข้อมูล\n\n{sections}"


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

    def post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        captured.update(url=url, payload=json, timeout=timeout)
        return fake_response()

    monkeypatch.setattr(llm_service.httpx, "post", post)
    source_text = "ชื่อโครงการ: ระบบค้นคืนข้อมูล"

    markdown = llm_service.generate_wiki(source_text)

    assert markdown == valid_thai_markdown()
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["payload"] == {
        "model": settings.ollama_model,
        "prompt": build_wiki_generation_prompt(source_text),
        "stream": False,
        "options": {"temperature": settings.ollama_temperature},
    }
    assert captured["timeout"] == settings.ollama_timeout_seconds


def test_generate_wiki_calls_the_existing_prompt_builder(settings, monkeypatch) -> None:
    source_text = "Extracted source text"
    called_with: list[str] = []

    def build_prompt(text: str) -> str:
        called_with.append(text)
        return "sentinel prompt"

    def post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        del url, timeout
        assert json["prompt"] == "sentinel prompt"
        return fake_response()

    monkeypatch.setattr(llm_service, "build_wiki_generation_prompt", build_prompt)
    monkeypatch.setattr(llm_service.httpx, "post", post)

    assert llm_service.generate_wiki(source_text) == valid_thai_markdown()
    assert called_with == [source_text]


def test_ollama_timeout_is_mapped(settings, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        del args, kwargs
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(llm_service.httpx, "post", timeout)

    with pytest.raises(llm_service.OllamaTimeoutError, match="timed out"):
        llm_service.generate_wiki("Source text")


def test_ollama_connection_failure_is_mapped(settings, monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        del args, kwargs
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm_service.httpx, "post", unavailable)

    with pytest.raises(llm_service.OllamaUnavailableError, match="not reachable"):
        llm_service.generate_wiki("Source text")


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


def test_invalid_markdown_structure_is_rejected(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(body={"response": "# Project\n\n## Wrong section"}),
    )

    with pytest.raises(llm_service.InvalidWikiMarkdownError) as failure:
        llm_service.generate_wiki("Source text")

    assert WikiStructureIssueCode.UNEXPECTED_LEVEL_TWO_HEADING in {
        issue.code for issue in failure.value.issues
    }


def test_valid_thai_markdown_is_returned_without_rewriting(settings, monkeypatch) -> None:
    thai_markdown = valid_thai_markdown().replace("เอกสารต้นฉบับ", "เอกสาร\u200bต้นฉบับ")
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(body={"response": thai_markdown}),
    )

    assert llm_service.generate_wiki("ข้อความต้นฉบับ") == thai_markdown


@pytest.mark.parametrize("status_code", [400, 503])
def test_other_http_errors_are_mapped(settings, monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(
        llm_service.httpx,
        "post",
        lambda *args, **kwargs: fake_response(status_code),
    )

    with pytest.raises(llm_service.OllamaRuntimeError, match=f"HTTP {status_code}"):
        llm_service.generate_wiki("Source text")


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

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11500"
    assert settings.ollama_model == "qwen2.5:3b-instruct"
    assert settings.ollama_timeout_seconds == 42
    assert settings.ollama_temperature == 0.1
