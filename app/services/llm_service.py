"""Wiki generation through a replaceable local Ollama client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.prompts import (
    WikiStructureIssue,
    build_wiki_generation_prompt,
    validate_wiki_markdown,
)
from app.prompts.project_summary import (
    build_project_summary_prompt,
    normalize_project_summary,
    validate_project_summary,
)
from app.services.wiki_output import WikiOutputError, refine_wiki_markdown
from app.services.wiki_timing import time_wiki_stage


class LLMServiceError(RuntimeError):
    """Base class for generation failures callers may handle."""


@dataclass(frozen=True, slots=True)
class OllamaErrorDiagnostics:
    model: str
    elapsed_seconds: float
    status_code: int | None
    response_body: str | None
    exception_type: str
    exception_message: str
    failure_kind: str
    connection_related: bool = False
    reset_related: bool = False
    protocol_related: bool = False
    runtime_related: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "status_code": self.status_code,
            "response_body": self.response_body,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "failure_kind": self.failure_kind,
            "connection_related": self.connection_related,
            "reset_related": self.reset_related,
            "protocol_related": self.protocol_related,
            "runtime_related": self.runtime_related,
        }


class OllamaRequestError(LLMServiceError):
    """An Ollama failure with optional diagnostics for logs and artifacts."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: OllamaErrorDiagnostics | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        super().__init__(message)


class OllamaUnavailableError(OllamaRequestError):
    """Ollama could not be reached."""


class OllamaModelNotFoundError(OllamaRequestError):
    """The configured model is not installed or available."""


class OllamaTimeoutError(OllamaRequestError):
    """Ollama did not answer within the configured timeout."""


class OllamaRuntimeError(OllamaRequestError):
    """Ollama returned an HTTP, transport, or malformed-response error."""


class EmptyModelResponseError(OllamaRequestError):
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


class InvalidProjectSummaryError(LLMServiceError):
    """The model reply violates the plain-text project summary contract."""

    def __init__(self, issues: tuple[str, ...], raw_text: str) -> None:
        self.issues = issues
        self.raw_text = raw_text
        super().__init__(f"Generated project summary is invalid: {', '.join(issues)}")


# Deprecated exception name retained for callers of generate_abstract_result.
InvalidAbstractError = InvalidProjectSummaryError


@dataclass(frozen=True, slots=True)
class ProjectSummaryGenerationResult:
    generated_summary: str
    raw_text: str
    cleanup_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AbstractGenerationResult:
    abstract: str
    raw_text: str
    cleanup_changes: tuple[str, ...]


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
    done: bool | None = None
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    context: tuple[int, ...] | None = None

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


def _optional_bool(body: dict, field: str) -> bool | None:
    value = body.get(field)
    return value if isinstance(value, bool) else None


def _optional_string(body: dict, field: str) -> str | None:
    value = body.get(field)
    return value if isinstance(value, str) else None


def _optional_context(body: dict) -> tuple[int, ...] | None:
    value = body.get("context")
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in value
    ):
        return None
    return tuple(value)


def _response_body(response: httpx.Response | None) -> str | None:
    if response is None:
        return None
    try:
        return response.text
    except Exception:
        return None


def _ollama_diagnostics(
    settings: Settings,
    started: float,
    exception: BaseException,
    *,
    response: httpx.Response | None = None,
    failure_kind: str,
) -> OllamaErrorDiagnostics:
    folded = f"{type(exception).__name__} {exception}".casefold()
    reset_related = "reset" in folded or "broken pipe" in folded
    protocol_related = isinstance(exception, httpx.ProtocolError)
    connection_related = isinstance(exception, httpx.ConnectError)
    runtime_related = failure_kind in {"http", "response", "runtime"}
    return OllamaErrorDiagnostics(
        model=settings.ollama_model,
        elapsed_seconds=time.perf_counter() - started,
        status_code=response.status_code if response is not None else None,
        response_body=_response_body(response),
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        failure_kind=(
            "reset"
            if reset_related
            else "protocol"
            if protocol_related
            else "connection"
            if connection_related
            else failure_kind
        ),
        connection_related=connection_related,
        reset_related=reset_related,
        protocol_related=protocol_related,
        runtime_related=runtime_related,
    )


class OllamaClient:
    """Own only the provider-specific HTTP request and response parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        prompt: str,
        *,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> str:
        result = self.generate_result(prompt, think=think, num_predict=num_predict)
        if result.done_reason and result.done_reason.casefold() == "length":
            raise InvalidProjectSummaryError(("summary_truncated",), result.response)
        return result.response

    def generate_result(
        self,
        prompt: str,
        *,
        think: bool | None = None,
        num_predict: int | None = None,
    ) -> OllamaGenerationResult:
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
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict

        timeout = self.settings.ollama_timeout_seconds
        started = time.perf_counter()
        response: httpx.Response | None = None
        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=timeout if timeout > 0 else None,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                "Ollama generation timed out",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, exc, response=response, failure_kind="timeout"
                ),
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                "Ollama is not reachable",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, exc, response=response, failure_kind="connection"
                ),
            ) from exc
        except httpx.HTTPStatusError as exc:
            diagnostics = _ollama_diagnostics(
                self.settings,
                started,
                exc,
                response=exc.response,
                failure_kind="http",
            )
            if exc.response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Ollama model is not available: {self.settings.ollama_model}",
                    diagnostics=diagnostics,
                ) from exc
            raise OllamaRuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}",
                diagnostics=diagnostics,
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaRuntimeError(
                "Ollama request failed",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, exc, response=response, failure_kind="runtime"
                ),
            ) from exc
        except Exception as exc:
            raise OllamaRuntimeError(
                "Unexpected Ollama runtime failure",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, exc, response=response, failure_kind="runtime"
                ),
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaRuntimeError(
                "Ollama returned invalid JSON",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, exc, response=response, failure_kind="response"
                ),
            ) from exc

        if not isinstance(body, dict):
            error = TypeError("Ollama response JSON is not an object")
            raise OllamaRuntimeError(
                "Ollama returned an invalid response",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, error, response=response, failure_kind="response"
                ),
            ) from error
        if body.get("error"):
            error = RuntimeError(str(body["error"]))
            raise OllamaRuntimeError(
                "Ollama reported a generation error",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, error, response=response, failure_kind="runtime"
                ),
            ) from error

        markdown = body.get("response")
        if markdown is None or markdown == "":
            error = ValueError("response is empty")
            raise EmptyModelResponseError(
                "Ollama returned no generated text",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, error, response=response, failure_kind="response"
                ),
            ) from error
        if not isinstance(markdown, str):
            error = TypeError("response is not a string")
            raise OllamaRuntimeError(
                "Ollama returned non-text content",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, error, response=response, failure_kind="response"
                ),
            ) from error
        if not markdown.strip():
            error = ValueError("response contains only whitespace")
            raise EmptyModelResponseError(
                "Ollama returned no generated text",
                diagnostics=_ollama_diagnostics(
                    self.settings, started, error, response=response, failure_kind="response"
                ),
            ) from error
        return OllamaGenerationResult(
            response=markdown,
            done=_optional_bool(body, "done"),
            done_reason=_optional_string(body, "done_reason"),
            total_duration=_optional_nonnegative_int(body, "total_duration"),
            load_duration=_optional_nonnegative_int(body, "load_duration"),
            prompt_eval_count=_optional_nonnegative_int(body, "prompt_eval_count"),
            prompt_eval_duration=_optional_nonnegative_int(body, "prompt_eval_duration"),
            eval_count=_optional_nonnegative_int(body, "eval_count"),
            eval_duration=_optional_nonnegative_int(body, "eval_duration"),
            context=_optional_context(body),
        )


def generate_legacy_wiki_result(source_text: str) -> WikiGenerationResult:
    """Legacy seven-section evaluation only; never used for production generation."""

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


def generate_abstract_result(source_text: str) -> AbstractGenerationResult:
    """Deprecated view of the plain-text project summary result."""
    result = generate_project_summary_result(source_text)
    return AbstractGenerationResult(
        result.generated_summary,
        result.raw_text,
        result.cleanup_changes,
    )


def generate_project_summary_result(source_text: str) -> ProjectSummaryGenerationResult:
    """Generate and validate a plain-text Thai project summary."""
    prompt = build_project_summary_prompt(source_text)
    settings = get_settings()
    with time_wiki_stage("llm_generation"):
        raw_text = OllamaClient(settings).generate(
            prompt,
            num_predict=settings.ollama_num_predict,
        )
    with time_wiki_stage("output_finalization"):
        summary = normalize_project_summary(raw_text)
        changes = ("boundary_whitespace_removed",) if summary != raw_text else ()
    with time_wiki_stage("validation"):
        validation = validate_project_summary(summary)
    if not validation.is_valid:
        raise InvalidProjectSummaryError(validation.issues, raw_text)
    return ProjectSummaryGenerationResult(
        summary,
        raw_text,
        changes,
    )


def generate_project_summary(source_text: str) -> ProjectSummaryGenerationResult:
    return generate_project_summary_result(source_text)


def generate_abstract(source_text: str) -> str:
    return generate_abstract_result(source_text).abstract


def generate_wiki_result(source_text: str) -> WikiGenerationResult:
    """Compatibility wrapper: markdown fields contain the project summary text."""
    result = generate_abstract_result(source_text)
    return WikiGenerationResult(result.abstract, result.raw_text, result.cleanup_changes)


def generate_wiki(source_text: str) -> str:
    """Deprecated compatibility name for production project-summary generation."""

    return generate_abstract(source_text)
