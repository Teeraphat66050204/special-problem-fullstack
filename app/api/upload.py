"""Persist page-aware extraction from a temporary digital-PDF upload."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.config import get_settings
from app.db import get_session
from app.models import Document, ExtractionStatus
from app.services.document_extraction import serialize_extraction
from app.services.ocr_service import (
    OcrServiceError,
    add_ocr_failure_warning,
    apply_ocr_fallback,
    load_ocr_provider,
)
from app.services.pdf_extractor import (
    EmptyPdfError,
    EncryptedPdfError,
    InvalidPdfError,
    PdfExtractionError,
    PdfExtractionResult,
    PdfExtractionWarning,
    WarningCode,
    extract_pdf,
)
from app.services.wiki_evidence import collect_wiki_source_evidence
from app.services.wiki_source import prepare_wiki_source
from app.services.wiki_timing import time_wiki_stage, upload_request_timing

router = APIRouter()
logger = logging.getLogger("uvicorn.error")
_COPY_CHUNK_BYTES = 1024 * 1024


class UploadResponse(BaseModel):
    """A persisted extraction that can be used for later Wiki generation."""

    document_id: int
    filename: str
    page_count: int
    extraction_status: ExtractionStatus
    warnings: tuple[PdfExtractionWarning, ...]


def display_filename(filename: str | None) -> str:
    """Keep only a display basename; uploaded names never form filesystem paths."""

    return (filename or "").replace("\\", "/").rsplit("/", 1)[-1] or "upload.pdf"


def _copy_upload(source: UploadFile, destination: Path, max_bytes: int) -> None:
    """Copy at most the configured limit from the spooled multipart file."""

    source.file.seek(0)
    total_bytes = 0
    with destination.open("xb") as output:
        while chunk := source.file.read(_COPY_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(status_code=413, detail="PDF exceeds the maximum upload size")
            if total_bytes == len(chunk) and not chunk.startswith(b"%PDF-"):
                raise HTTPException(status_code=422, detail="The uploaded file is not a valid PDF")
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")


def extract_uploaded_pdf(file: UploadFile) -> PdfExtractionResult:
    """Apply upload checks and extract a safely stored temporary PDF."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only application/pdf uploads are supported")

    settings = get_settings()
    try:
        with TemporaryDirectory(prefix="pdf-upload-") as temporary_directory:
            with time_wiki_stage("upload"):
                pdf_path = Path(temporary_directory) / "upload.pdf"
                _copy_upload(file, pdf_path, settings.max_upload_bytes)
            with time_wiki_stage("extraction"), pdf_path.open("rb") as saved_pdf:
                native_result = extract_pdf(saved_pdf)
                try:
                    result = apply_ocr_fallback(
                        saved_pdf,
                        native_result,
                        load_ocr_provider(settings),
                        front_matter_page_limit=settings.ocr_max_pages,
                    )
                except OcrServiceError as error:
                    result = add_ocr_failure_warning(native_result, error)
                for warning in result.warnings[len(native_result.warnings) :]:
                    if warning.code not in {
                        WarningCode.OCR_CONFIGURATION,
                        WarningCode.OCR_FAILED,
                    }:
                        continue
                    logger.warning(
                        "wiki.upload OCR fallback warning filename=%s code=%s",
                        display_filename(file.filename),
                        warning.code.value,
                    )
    except InvalidPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EncryptedPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmptyPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PdfExtractionError as exc:
        raise HTTPException(status_code=500, detail="Could not extract PDF text") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not process uploaded PDF") from exc

    return result


@router.post(
    "/api/upload",
    response_model=UploadResponse,
    summary="Upload and extract a PDF once",
    description=(
        "Upload one digital PDF in the `file` multipart field. The page-aware text "
        "extraction is persisted as a Document; use the returned `document_id` with "
        "`POST /api/wiki/generate`. This endpoint does not generate a Wiki."
    ),
    responses={
        400: {"description": "Empty upload"},
        413: {"description": "Upload exceeds the configured size limit"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Invalid, encrypted, or empty PDF"},
        500: {"description": "Extraction or persistence failed"},
    },
)
def upload_pdf(
    file: Annotated[UploadFile, File(description="Digital PDF to extract")],
    session: Annotated[Session, Depends(get_session)],
) -> UploadResponse:
    """Extract one digital PDF and persist its page-aware text once."""

    filename = display_filename(file.filename)
    with upload_request_timing(filename):
        result = extract_uploaded_pdf(file)
        title: str | None = None
        author: str | None = None
        student_id: str | None = None
        academic_year: int | None = None
        try:
            selection = prepare_wiki_source(result)
            evidence = collect_wiki_source_evidence(selection.source_text)
            title = evidence.title
            author = evidence.students[0] if evidence.students else None
            student_id = evidence.student_ids[0] if evidence.student_ids else None
            academic_year = int(evidence.academic_year) if evidence.academic_year else None
        except ValueError:
            pass

        document = Document(
            original_filename=filename,
            title=title,
            author=author,
            student_id=student_id,
            academic_year=academic_year,
            storage_key=f"extractions/{uuid4()}.json",
            raw_text=result.full_text,
            extraction_data=serialize_extraction(result),
            page_count=result.page_count,
            extraction_status=ExtractionStatus.COMPLETED,
        )
        try:
            with time_wiki_stage("persistence"):
                session.add(document)
                session.commit()
                session.refresh(document)
        except SQLAlchemyError as exc:
            session.rollback()
            logger.exception("wiki.upload persistence failed filename=%s", filename)
            raise HTTPException(
                status_code=500, detail="Could not persist extracted document"
            ) from exc

        if document.id is None:
            raise HTTPException(status_code=500, detail="Could not persist extracted document")

    return UploadResponse(
        document_id=document.id,
        filename=filename,
        page_count=result.page_count,
        extraction_status=document.extraction_status,
        warnings=result.warnings,
    )
