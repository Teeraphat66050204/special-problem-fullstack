"""Wiki generation through a replaceable local Ollama client."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings, get_settings
from app.prompts import (
    WikiStructureIssue,
    build_wiki_generation_prompt,
    validate_wiki_markdown,
)
from app.services.wiki_output import WikiOutputError, refine_wiki_markdown
from app.services.wiki_timing import time_wiki_stage


class LLMServiceError(RuntimeError):
    """Base class for generation failures callers may handle."""


class OllamaUnavailableError(LLMServiceError):
    """Ollama could not be reached."""


class OllamaModelNotFoundError(LLMServiceError):
    """The configured model is not installed or available."""


class OllamaTimeoutError(LLMServiceError):
    """Ollama did not answer within the configured timeout."""


class OllamaRuntimeError(LLMServiceError):
    """Ollama returned an HTTP, transport, or malformed-response error."""


class EmptyModelResponseError(LLMServiceError):
    """Ollama returned no generated text."""


class InvalidWikiMarkdownError(LLMServiceError):
    """Generated Markdown failed the Task 2.1 structure contract."""

    def __init__(self, issues: tuple[WikiStructureIssue, ...]) -> None:
        self.issues = issues
        codes = ", ".join(issue.code for issue in issues)
        super().__init__(f"Generated Wiki Markdown has invalid structure: {codes}")


class InvalidWikiOutputError(LLMServiceError):
    """Generated Wiki text has content that cannot be finalized safely."""

    def __init__(self, message: str, raw_markdown: str) -> None:
        self.raw_markdown = raw_markdown
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WikiGenerationResult:
    """Final Markdown plus raw model output for evaluation diagnostics."""

    markdown: str
    raw_markdown: str
    refinement_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OllamaGenerationResult:
    """Generated text plus optional native Ollama performance counters."""

    response: str
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.eval_count is None or not self.eval_duration:
            return None
        return self.eval_count / (self.eval_duration / 1_000_000_000)


def _optional_nonnegative_int(body: dict, field: str) -> int | None:
    value = body.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class OllamaClient:
    """Own only the provider-specific HTTP request and response parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        return self.generate_result(prompt).response

    def generate_result(self, prompt: str, *, think: bool | None = None) -> OllamaGenerationResult:
        """Generate text and retain provider metrics; optionally set native thinking mode."""

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.settings.ollama_temperature},
        }
        if think is not None:
            payload["think"] = think

        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama generation timed out") from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError("Ollama is not reachable") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Ollama model is not available: {self.settings.ollama_model}"
                ) from exc
            raise OllamaRuntimeError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise OllamaRuntimeError("Ollama request failed") from exc
        except Exception as exc:
            raise OllamaRuntimeError("Unexpected Ollama runtime failure") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaRuntimeError("Ollama returned invalid JSON") from exc

        if not isinstance(body, dict):
            raise OllamaRuntimeError("Ollama returned an invalid response")
        if body.get("error"):
            raise OllamaRuntimeError("Ollama reported a generation error")

        markdown = body.get("response")
        if markdown is None or markdown == "":
            raise EmptyModelResponseError("Ollama returned no generated text")
        if not isinstance(markdown, str):
            raise OllamaRuntimeError("Ollama returned non-text content")
        if not markdown.strip():
            raise EmptyModelResponseError("Ollama returned no generated text")
        return OllamaGenerationResult(
            response=markdown,
            prompt_eval_count=_optional_nonnegative_int(body, "prompt_eval_count"),
            prompt_eval_duration=_optional_nonnegative_int(body, "prompt_eval_duration"),
            eval_count=_optional_nonnegative_int(body, "eval_count"),
            eval_duration=_optional_nonnegative_int(body, "eval_duration"),
        )


def generate_wiki_result(source_text: str) -> WikiGenerationResult:
    """Generate and finalize Markdown, retaining the raw model reply for reviews."""

    prompt = build_wiki_generation_prompt(source_text)
    with time_wiki_stage("llm_generation"):
        raw_markdown = OllamaClient(get_settings()).generate(prompt)
    try:
        with time_wiki_stage("output_finalization"):
            refinement = refine_wiki_markdown(source_text, raw_markdown)
    except WikiOutputError as exc:
        raise InvalidWikiOutputError(str(exc), raw_markdown) from exc
    markdown = refinement.markdown
    with time_wiki_stage("validation"):
        validation = validate_wiki_markdown(markdown)
    if not validation.is_valid:
        raise InvalidWikiMarkdownError(validation.issues)
    return WikiGenerationResult(markdown, raw_markdown, refinement.changes)


def generate_wiki(source_text: str) -> str:
    """Return source-backed Markdown through the existing service interface."""

    return generate_wiki_result(source_text).markdown
