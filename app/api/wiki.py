"""Generate an in-memory Draft Wiki from a persisted PDF extraction."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.db import get_session
from app.models import Document, ExtractionStatus
from app.prompts import validate_wiki_markdown
from app.services.document_extraction import StoredExtractionError, extraction_from_document
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
from app.services.wiki_timing import time_wiki_stage, wiki_request_timing

router = APIRouter()


class DraftWikiRequest(BaseModel):
    """Reference an existing successfully extracted Document."""

    document_id: int = Field(gt=0)


class DraftWikiResponse(BaseModel):
    """Generated Markdown tied to a Document, without Wiki persistence or publication."""

    status: Literal["draft"] = "draft"
    document_id: int
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
    summary="Generate a Draft Wiki from an uploaded Document",
    description=(
        "Send the `document_id` returned by `POST /api/upload`. Only selected title, "
        "metadata, abstract, and keyword pages from its persisted extraction are sent "
        "to the configured Ollama model. The generated draft is validated and returned "
        "without persistence."
    ),
    responses={
        404: {"description": "Document not found"},
        409: {"description": "Document extraction is incomplete or unavailable"},
        422: {"description": "No usable focused source"},
        502: {"description": "Ollama returned an invalid Wiki or runtime error"},
        503: {"description": "Ollama or the configured model is unavailable"},
        504: {"description": "Ollama generation timed out"},
    },
)
def generate_draft_wiki(
    request: DraftWikiRequest,
    session: Annotated[Session, Depends(get_session)],
) -> DraftWikiResponse:
    """Load, select focused text, generate, validate, and return a draft."""

    with wiki_request_timing(request.document_id):
        document = session.get(Document, request.document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.extraction_status is not ExtractionStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="Document extraction is not complete")
        try:
            extraction = extraction_from_document(document)
        except StoredExtractionError as exc:
            raise HTTPException(
                status_code=409, detail="Document extraction data is unavailable"
            ) from exc

        try:
            with time_wiki_stage("source_selection"):
                selection = prepare_wiki_source(extraction)
                if not selection.source_text.strip():
                    raise ValueError(
                        "No usable front-matter text was extracted for Wiki generation"
                    )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not prepare Wiki source") from exc

        with time_wiki_stage("evidence_extraction"):
            evidence = collect_wiki_source_evidence(selection.source_text)

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

        with time_wiki_stage("validation"):
            validation = validate_wiki_markdown(markdown)
        if not validation.is_valid:
            raise HTTPException(status_code=502, detail="Generated Wiki has invalid structure")

        return DraftWikiResponse(
            document_id=request.document_id,
            original_filename=document.original_filename,
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
