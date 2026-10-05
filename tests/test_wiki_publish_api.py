"""Human-reviewed Wiki publication API tests."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.api import wiki
from app.db import get_session
from app.main import app
from app.models import Chunk, Document, ExtractionStatus, WikiPage, WikiStatus
from app.prompts import MISSING_INFORMATION_MARKER, REQUIRED_WIKI_HEADINGS
from app.services.document_extraction import serialize_extraction
from app.services.pdf_extractor import PdfExtractionResult, PdfPageText

pytestmark = pytest.mark.usefixtures("api_database")


def reviewed_wiki(title: str, *, overview: str = MISSING_INFORMATION_MARKER) -> str:
    """Return valid canonical Markdown suitable for publication tests."""

    sections = "\n\n".join(
        f"{heading}\n{overview if index == 0 else MISSING_INFORMATION_MARKER}"
        for index, heading in enumerate(REQUIRED_WIKI_HEADINGS)
    )
    return f"# {title}\n\n{sections}"


def persist_document(engine: Engine, *, completed: bool = False) -> int:
    """Create the Document referenced by a publish request."""

    fields: dict[str, object] = {}
    if completed:
        source = "Reviewed Project\nAbstract\nSource text"
        extraction = PdfExtractionResult(
            page_count=1,
            full_text=source,
            pages=(PdfPageText(page_number=1, text=source),),
            warnings=(),
        )
        fields = {
            "raw_text": source,
            "extraction_data": serialize_extraction(extraction),
            "page_count": 1,
            "extraction_status": ExtractionStatus.COMPLETED,
        }

    with Session(engine) as session:
        document = Document(
            original_filename="reviewed-project.pdf",
            storage_key="documents/reviewed-project.pdf",
            **fields,
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        assert document.id is not None
        return document.id


def publish(document_id: int, markdown: str):
    return TestClient(app).post(
        "/api/wiki/publish",
        json={"document_id": document_id, "markdown_content": markdown},
    )


def test_publish_valid_reviewed_markdown_persists_version_one_exactly(
    api_database: Engine,
) -> None:
    document_id = persist_document(api_database)
    markdown = reviewed_wiki(
        "ระบบคลังความรู้ / Knowledge Repository",
        overview="เนื้อหาภาษาไทยและ English text preserved exactly.",
    )

    response = publish(document_id, markdown)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data == {
        "wiki_page_id": data["wiki_page_id"],
        "document_id": document_id,
        "version": 1,
        "status": "published",
        "is_published": True,
        "markdown_content": markdown,
    }
    with Session(api_database) as session:
        pages = session.exec(select(WikiPage)).all()
        assert len(pages) == 1
        assert pages[0].id == data["wiki_page_id"]
        assert pages[0].markdown_content == markdown
        assert pages[0].version == 1
        assert pages[0].status is WikiStatus.PUBLISHED
        assert pages[0].is_published is True
        assert session.exec(select(Chunk)).all() == []


def test_second_publish_archives_first_and_publishes_version_two(
    api_database: Engine,
) -> None:
    document_id = persist_document(api_database)
    first_markdown = reviewed_wiki("First reviewed version")
    second_markdown = reviewed_wiki(
        "Second reviewed version",
        overview="แก้ไขโดยผู้ตรวจทาน / Human-reviewed update",
    )

    first_response = publish(document_id, first_markdown)
    second_response = publish(document_id, second_markdown)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["version"] == 2
    assert second_response.json()["markdown_content"] == second_markdown
    with Session(api_database) as session:
        pages = session.exec(
            select(WikiPage).where(WikiPage.document_id == document_id).order_by(WikiPage.version)
        ).all()

    assert [(page.version, page.status, page.is_published) for page in pages] == [
        (1, WikiStatus.ARCHIVED, False),
        (2, WikiStatus.PUBLISHED, True),
    ]
    assert pages[0].markdown_content == first_markdown
    assert pages[1].markdown_content == second_markdown
    assert sum(page.status is WikiStatus.PUBLISHED for page in pages) == 1


def test_publish_unknown_document_returns_404_and_creates_nothing(
    api_database: Engine,
) -> None:
    response = publish(999_999, reviewed_wiki("Unknown document"))

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
    with Session(api_database) as session:
        assert session.exec(select(WikiPage)).all() == []


def test_publish_rejects_noncanonical_markdown_without_finalizing_it(
    api_database: Engine,
) -> None:
    document_id = persist_document(api_database)
    noncanonical = reviewed_wiki("Needs finalization").replace(
        "## ภาพรวมโครงงาน", "## ภาพรวมโครงการ"
    )

    response = publish(document_id, noncanonical)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "Submitted Wiki Markdown has invalid structure"
    assert {issue["code"] for issue in detail["issues"]} == {
        "missing_required_heading",
        "unexpected_level_two_heading",
    }
    with Session(api_database) as session:
        assert session.exec(select(WikiPage)).all() == []


def test_generate_endpoint_still_does_not_persist_wiki_page(
    api_database: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = persist_document(api_database, completed=True)
    markdown = "โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยเพื่อสนับสนุนการเข้าถึงข้อมูล"
    monkeypatch.setattr(wiki, "generate_wiki", lambda source: markdown)

    response = TestClient(app).post("/api/wiki/generate", json={"document_id": document_id})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "draft"
    assert response.json()["generated_markdown"] == markdown
    with Session(api_database) as session:
        assert session.exec(select(WikiPage)).all() == []


def test_database_failure_rolls_back_archival_and_new_version(
    api_database: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = persist_document(api_database)
    first_markdown = reviewed_wiki("Published before failure")
    first_response = publish(document_id, first_markdown)
    assert first_response.status_code == 200
    first_page_id = first_response.json()["wiki_page_id"]

    failing_session = Session(api_database)
    rollback_called = False

    def fail_commit() -> None:
        raise SQLAlchemyError("forced publication failure")

    real_rollback = failing_session.rollback

    def track_rollback() -> None:
        nonlocal rollback_called
        rollback_called = True
        real_rollback()

    monkeypatch.setattr(failing_session, "commit", fail_commit)
    monkeypatch.setattr(failing_session, "rollback", track_rollback)

    def override_session() -> Generator[Session]:
        yield failing_session

    previous_override = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = override_session
    try:
        response = publish(document_id, reviewed_wiki("Publication that fails"))
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_session, None)
        else:
            app.dependency_overrides[get_session] = previous_override
        failing_session.close()

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not publish Wiki"}
    assert "forced publication failure" not in response.text
    assert rollback_called is True
    with Session(api_database) as session:
        pages = session.exec(select(WikiPage)).all()

    assert len(pages) == 1
    assert pages[0].id == first_page_id
    assert pages[0].version == 1
    assert pages[0].status is WikiStatus.PUBLISHED
    assert pages[0].is_published is True
    assert pages[0].markdown_content == first_markdown


def test_publish_route_is_documented_in_openapi() -> None:
    operation = TestClient(app).get("/openapi.json").json()["paths"]["/api/wiki/publish"]["post"]

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["$ref"].endswith("/PublishWikiRequest")
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/PublishedWikiResponse"
    )
    assert {"404", "422", "500"} <= operation["responses"].keys()
