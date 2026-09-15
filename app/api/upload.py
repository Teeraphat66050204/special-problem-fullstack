"""Temporary digital-PDF upload and extraction endpoint."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.services.pdf_extractor import (
    EmptyPdfError,
    EncryptedPdfError,
    InvalidPdfError,
    PdfExtractionError,
    PdfExtractionWarning,
    PdfPageText,
    extract_pdf,
)

router = APIRouter()
_COPY_CHUNK_BYTES = 1024 * 1024


class UploadResponse(BaseModel):
    """The existing extractor data with a display filename and API text key."""

    filename: str
    page_count: int
    raw_text: str
    pages: tuple[PdfPageText, ...]
    warnings: tuple[PdfExtractionWarning, ...]


def _display_filename(filename: str | None) -> str:
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


@router.post("/api/upload", response_model=UploadResponse)
def upload_pdf(
    file: Annotated[UploadFile, File(description="Digital PDF to extract")],
) -> UploadResponse:
    """Extract a digital PDF without keeping the upload or persisting data."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only application/pdf uploads are supported")

    try:
        with TemporaryDirectory(prefix="pdf-upload-") as temporary_directory:
            pdf_path = Path(temporary_directory) / "upload.pdf"
            _copy_upload(file, pdf_path, get_settings().max_upload_bytes)
            with pdf_path.open("rb") as saved_pdf:
                result = extract_pdf(saved_pdf)
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

    return UploadResponse(
        filename=_display_filename(file.filename),
        page_count=result.page_count,
        raw_text=result.full_text,
        pages=result.pages,
        warnings=result.warnings,
    )
