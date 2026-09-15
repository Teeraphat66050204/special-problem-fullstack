"""Wiki generation through a replaceable local Ollama client."""

from __future__ import annotations

import httpx

from app.config import Settings, get_settings
from app.prompts import (
    WikiStructureIssue,
    build_wiki_generation_prompt,
    validate_wiki_markdown,
)


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


class OllamaClient:
    """Own only the provider-specific HTTP request and response parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.settings.ollama_temperature},
        }

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
        return markdown


def generate_wiki(source_text: str) -> str:
    """Build the existing prompt, generate Markdown, and validate its structure."""

    prompt = build_wiki_generation_prompt(source_text)
    markdown = OllamaClient(get_settings()).generate(prompt)
    validation = validate_wiki_markdown(markdown)
    if not validation.is_valid:
        raise InvalidWikiMarkdownError(validation.issues)
    return markdown
