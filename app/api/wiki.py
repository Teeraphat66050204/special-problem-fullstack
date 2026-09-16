"""Generate an in-memory Draft Wiki from a focused digital-PDF source."""

from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.upload import display_filename, extract_uploaded_pdf
from app.prompts import validate_wiki_markdown
from app.services.llm_service import (
    EmptyModelResponseError,
    InvalidWikiMarkdownError,
    InvalidWikiOutputError,
    LLMServiceError,
    OllamaModelNotFoundError,
    OllamaRuntimeError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_wiki,
)
from app.services.pdf_extractor import PdfExtractionWarning
from app.services.wiki_evidence import collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source

router = APIRouter()


class DraftWikiResponse(BaseModel):
    """Generated Markdown and provenance, with no database identity or publication."""

    status: Literal["draft"] = "draft"
    original_filename: str
    page_count: int
    selected_pages: tuple[int, ...]
    generated_markdown: str
    structure_valid: bool
    warnings: tuple[PdfExtractionWarning, ...]
    title: str | None
    english_title: str | None
    students: tuple[str, ...]
    student_ids: tuple[str, ...]
    advisor: str | None
    academic_year: str | None
    keywords: tuple[str, ...]


@router.post(
    "/api/wiki/generate",
    response_model=DraftWikiResponse,
    summary="Generate a Draft Wiki from a digital PDF",
    description=(
        "Upload one PDF in the `file` multipart field. Only selected title, metadata, "
        "abstract, and keyword pages are sent to the configured Ollama model. "
        "The generated draft is validated and returned without persistence."
    ),
    responses={
        400: {"description": "Empty upload"},
        413: {"description": "Upload exceeds the configured size limit"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Invalid PDF or no usable focused source"},
        502: {"description": "Ollama returned an invalid Wiki or runtime error"},
        503: {"description": "Ollama or the configured model is unavailable"},
        504: {"description": "Ollama generation timed out"},
    },
)
def generate_draft_wiki(
    file: Annotated[UploadFile, File(description="Digital PDF to turn into a Draft Wiki")],
) -> DraftWikiResponse:
    """Extract, select focused text, generate, validate, and return a draft."""

    extraction = extract_uploaded_pdf(file)
    try:
        selection = prepare_wiki_source(extraction)
        if not selection.source_text.strip():
            raise ValueError("No usable front-matter text was extracted for Wiki generation")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not prepare Wiki source") from exc

    try:
        markdown = generate_wiki(selection.source_text)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Configured Ollama model is unavailable"
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Wiki generation timed out") from exc
    except (InvalidWikiMarkdownError, InvalidWikiOutputError, EmptyModelResponseError) as exc:
        raise HTTPException(status_code=502, detail="Generated Wiki is invalid") from exc
    except OllamaRuntimeError as exc:
        raise HTTPException(status_code=502, detail="Ollama generation failed") from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail="Wiki generation failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not generate Draft Wiki") from exc

    validation = validate_wiki_markdown(markdown)
    if not validation.is_valid:
        raise HTTPException(status_code=502, detail="Generated Wiki has invalid structure")

    evidence = collect_wiki_source_evidence(selection.source_text)
    return DraftWikiResponse(
        original_filename=display_filename(file.filename),
        page_count=extraction.page_count,
        selected_pages=selection.selected_pages,
        generated_markdown=markdown,
        structure_valid=validation.is_valid,
        warnings=extraction.warnings,
        title=evidence.title,
        english_title=evidence.title_en,
        students=evidence.students,
        student_ids=evidence.student_ids,
        advisor=evidence.advisor,
        academic_year=evidence.academic_year,
        keywords=evidence.keywords,
    )
