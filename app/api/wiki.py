"""Generate Thai abstract drafts and publish separate human-reviewed Wiki versions."""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.db import get_session
from app.models import Document, ExtractionStatus, WikiPage, WikiStatus
from app.prompts import validate_wiki_markdown
from app.prompts.abstract_generation import validate_abstract
from app.services.document_extraction import StoredExtractionError, extraction_from_document
from app.services.llm_service import (
    EmptyModelResponseError,
    InvalidAbstractError,
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
from app.services.wiki_evidence import WikiSourceEvidence, collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source
from app.services.wiki_timing import time_wiki_stage, wiki_request_timing

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


class DraftWikiRequest(BaseModel):
    """Reference an existing successfully extracted Document."""

    document_id: int = Field(gt=0)


class DraftWikiResponse(BaseModel):
    """Abstract draft tied to a Document, without persistence or publication."""

    status: Literal["draft"] = "draft"
    document_id: int
    original_filename: str
    page_count: int
    selected_pages: tuple[int, ...]
    generated_abstract: str
    generated_markdown: str = Field(
        description=("Deprecated alias of generated_abstract; contains plain text."),
        deprecated=True,
    )
    structure_valid: bool
    warnings: tuple[PdfExtractionWarning, ...]
    title: str | None
    english_title: str | None
    students: tuple[str, ...]
    student_ids: tuple[str, ...]
    advisor: str | None
    academic_year: str | None
    keywords: tuple[str, ...]


class PublishWikiRequest(BaseModel):
    """Human-reviewed Markdown to publish for an existing Document."""

    document_id: int = Field(gt=0)
    markdown_content: str = Field(min_length=1)


class PublishedWikiResponse(BaseModel):
    """The newly persisted, currently published Wiki version."""

    wiki_page_id: int
    document_id: int
    version: int
    status: WikiStatus
    is_published: bool
    markdown_content: str


@router.post(
    "/api/wiki/publish",
    response_model=PublishedWikiResponse,
    summary="Publish a human-reviewed Wiki",
    description=(
        "Validate already-reviewed Markdown against the strict canonical Wiki schema, "
        "archive any currently published version, and persist the Markdown as the next "
        "published version. This endpoint does not run generation or finalization."
    ),
    responses={
        404: {"description": "Document not found"},
        422: {"description": "Submitted Markdown violates the canonical Wiki schema"},
        500: {"description": "Wiki publication could not be persisted"},
    },
)
def publish_reviewed_wiki(
    request: PublishWikiRequest,
    session: Annotated[Session, Depends(get_session)],
) -> PublishedWikiResponse:
    """Validate and atomically publish the next Wiki version for a Document."""

    try:
        document = session.exec(
            select(Document).where(Document.id == request.document_id).with_for_update()
        ).one_or_none()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("wiki.publish document lookup failed document_id=%s", request.document_id)
        raise HTTPException(status_code=500, detail="Could not publish Wiki") from exc

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    validation = validate_wiki_markdown(request.markdown_content)
    if not validation.is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Submitted Wiki Markdown has invalid structure",
                "issues": [
                    {"code": issue.code.value, "heading": issue.heading}
                    for issue in validation.issues
                ],
            },
        )

    try:
        existing_pages = session.exec(
            select(WikiPage).where(WikiPage.document_id == request.document_id)
        ).all()
        next_version = max((page.version for page in existing_pages), default=0) + 1

        for page in existing_pages:
            if page.status is WikiStatus.PUBLISHED or page.is_published:
                page.status = WikiStatus.ARCHIVED
                page.is_published = False

        wiki_page = WikiPage(
            document_id=request.document_id,
            markdown_content=request.markdown_content,
            version=next_version,
            status=WikiStatus.PUBLISHED,
            is_published=True,
        )
        session.add(wiki_page)
        session.flush()
        if wiki_page.id is None:
            raise SQLAlchemyError("Published WikiPage did not receive an ID")

        response = PublishedWikiResponse(
            wiki_page_id=wiki_page.id,
            document_id=request.document_id,
            version=next_version,
            status=wiki_page.status,
            is_published=wiki_page.is_published,
            markdown_content=wiki_page.markdown_content,
        )
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("wiki.publish persistence failed document_id=%s", request.document_id)
        raise HTTPException(status_code=500, detail="Could not publish Wiki") from exc

    return response


@router.post(
    "/api/wiki/generate",
    response_model=DraftWikiResponse,
    summary="Generate a Thai abstract draft from an uploaded Document",
    description=(
        "Send the `document_id` returned by `POST /api/upload`. Document-wide abstract "
        "and real Chapter 1 detection selects focused excerpts and optional metadata "
        "from persisted extraction. The model returns one concise Thai abstract draft, "
        "validated and returned without persistence or automatic publication."
    ),
    responses={
        404: {"description": "Document not found"},
        409: {"description": "Document extraction is incomplete or unavailable"},
        422: {"description": "No usable focused source"},
        502: {"description": "Ollama returned an invalid abstract or runtime error"},
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
                    raise ValueError("No usable abstract or real Chapter 1 source was extracted")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not prepare Wiki source") from exc

        with time_wiki_stage("evidence_extraction"):
            try:
                evidence = collect_wiki_source_evidence(selection.source_text)
            except Exception:
                logger.exception("Optional abstract metadata extraction failed")
                evidence = WikiSourceEvidence(None, None, (), (), None, None, ())

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
        except (
            InvalidAbstractError,
            InvalidWikiMarkdownError,
            InvalidWikiOutputError,
            EmptyModelResponseError,
        ) as exc:
            raise HTTPException(status_code=502, detail="Generated abstract is invalid") from exc
        except OllamaRuntimeError as exc:
            raise HTTPException(status_code=502, detail="Ollama generation failed") from exc
        except LLMServiceError as exc:
            raise HTTPException(status_code=502, detail="Wiki generation failed") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not generate Draft Wiki") from exc

        with time_wiki_stage("validation"):
            validation = validate_abstract(markdown)
        if not validation.is_valid:
            raise HTTPException(status_code=502, detail="Generated abstract has invalid structure")

        return DraftWikiResponse(
            document_id=request.document_id,
            original_filename=document.original_filename,
            page_count=extraction.page_count,
            selected_pages=selection.selected_pages,
            generated_abstract=markdown,
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
