"""Tests for the Phase 1 SQLModel entities."""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Chunk, Document, ExtractionStatus, WikiPage, WikiStatus


@pytest.fixture
def engine():
    database = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(database, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(database)
    yield database
    database.dispose()


def test_document_wiki_and_chunk_relationships(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        title="A Special Project",
        author="Student Name",
        student_id="66050000",
        academic_year=2569,
        storage_key="documents/project.pdf",
        raw_text="Source page text",
        extraction_data=(
            '{"version":1,"pages":[{"page_number":1,"text":"Source page text"}],"warnings":[]}'
        ),
        page_count=1,
        extraction_status=ExtractionStatus.COMPLETED,
    )
    wiki_page = WikiPage(
        markdown_content="# A Special Project",
        version=1,
        status=WikiStatus.DRAFT,
    )
    chunk = Chunk(
        chunk_index=0,
        content="Source page text",
        source_page=1,
        token_count=3,
    )
    document.wiki_pages.append(wiki_page)
    document.chunks.append(chunk)
    wiki_page.chunks.append(chunk)

    with Session(engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)

        stored = session.exec(select(Document)).one()
        assert stored.extraction_status is ExtractionStatus.COMPLETED
        assert stored.wiki_pages[0].document_id == stored.id
        assert stored.chunks[0].document_id == stored.id
        assert stored.chunks[0].wiki_page_id == stored.wiki_pages[0].id
        assert stored.chunks[0].source_page == 1


def test_wiki_relationship_populates_chunk_document_provenance(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(markdown_content="# Project")
    chunk = Chunk(chunk_index=0, content="Wiki content")
    document.wiki_pages.append(wiki_page)
    wiki_page.chunks.append(chunk)

    with Session(engine) as session:
        session.add(document)
        session.commit()

        assert chunk.wiki_page_id == wiki_page.id
        assert chunk.document_id == document.id
        assert chunk in document.chunks


def test_deleting_wiki_page_deletes_its_chunks(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(markdown_content="# Project")
    chunk = Chunk(chunk_index=0, content="Wiki content")
    document.wiki_pages.append(wiki_page)
    wiki_page.chunks.append(chunk)

    with Session(engine) as session:
        session.add(document)
        session.commit()
        chunk_id = chunk.id

        session.delete(wiki_page)
        session.commit()

        assert session.get(Chunk, chunk_id) is None


def test_default_model_lifecycle_values() -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(document_id=1, markdown_content="")

    assert document.extraction_status is ExtractionStatus.PENDING
    assert document.page_count == 0
    assert wiki_page.status is WikiStatus.DRAFT
    assert wiki_page.version == 1
    assert wiki_page.is_published is False
    assert document.created_at.tzinfo is not None


def test_completed_document_requires_persisted_extraction_data(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
        raw_text="Source text",
        extraction_status=ExtractionStatus.COMPLETED,
    )

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_wiki_version_is_unique_per_document(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.wiki_pages.extend(
        [
            WikiPage(markdown_content="# First", version=1),
            WikiPage(markdown_content="# Duplicate", version=1),
        ]
    )

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_chunk_rejects_invalid_page_number(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.chunks.append(Chunk(chunk_index=0, content="Text", source_page=0))

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_chunk_rejects_negative_token_count(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.chunks.append(Chunk(chunk_index=0, content="Text", token_count=-1))

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


@pytest.mark.parametrize("content", ["", "   "])
def test_chunk_rejects_empty_content(engine, content: str) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.chunks.append(Chunk(chunk_index=0, content=content))

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_publication_status_and_flag_must_agree(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.wiki_pages.append(
        WikiPage(
            markdown_content="# Project",
            status=WikiStatus.PUBLISHED,
            is_published=False,
        )
    )

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_chunk_rejects_wiki_page_from_another_document(engine) -> None:
    document_a = Document(original_filename="a.pdf", storage_key="documents/a.pdf")
    document_b = Document(original_filename="b.pdf", storage_key="documents/b.pdf")
    wiki_page_a = WikiPage(markdown_content="# Document A")
    document_a.wiki_pages.append(wiki_page_a)

    with Session(engine) as session:
        session.add(document_a)
        session.add(document_b)
        session.commit()

        mismatched_chunk = Chunk(
            document_id=document_b.id,
            wiki_page_id=wiki_page_a.id,
            chunk_index=0,
            content="Invalid provenance",
        )
        session.add(mismatched_chunk)

        with pytest.raises(IntegrityError):
            session.commit()


def test_chunk_accepts_wiki_page_from_same_document(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(markdown_content="# Project")
    document.wiki_pages.append(wiki_page)

    with Session(engine) as session:
        session.add(document)
        session.commit()

        chunk = Chunk(
            document_id=document.id,
            wiki_page_id=wiki_page.id,
            chunk_index=0,
            content="Valid provenance",
        )
        session.add(chunk)
        session.commit()

        assert chunk.id is not None


def test_document_level_chunk_index_is_unique_within_document(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    document.chunks.extend(
        [
            Chunk(chunk_index=0, content="First"),
            Chunk(chunk_index=0, content="Duplicate"),
        ]
    )

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_document_level_chunk_indexes_are_scoped_to_document(engine) -> None:
    document_a = Document(original_filename="a.pdf", storage_key="documents/a.pdf")
    document_b = Document(original_filename="b.pdf", storage_key="documents/b.pdf")
    document_a.chunks.append(Chunk(chunk_index=0, content="Document A"))
    document_b.chunks.append(Chunk(chunk_index=0, content="Document B"))

    with Session(engine) as session:
        session.add(document_a)
        session.add(document_b)
        session.commit()

        assert document_a.chunks[0].id is not None
        assert document_b.chunks[0].id is not None


def test_chunk_indexes_are_scoped_to_wiki_page(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    first_wiki = WikiPage(markdown_content="# First", version=1)
    second_wiki = WikiPage(markdown_content="# Second", version=2)
    document.wiki_pages.extend([first_wiki, second_wiki])

    with Session(engine) as session:
        session.add(document)
        session.commit()

        first_chunk = Chunk(
            document_id=document.id,
            wiki_page_id=first_wiki.id,
            chunk_index=0,
            content="First Wiki",
        )
        second_chunk = Chunk(
            document_id=document.id,
            wiki_page_id=second_wiki.id,
            chunk_index=0,
            content="Second Wiki",
        )
        session.add(first_chunk)
        session.add(second_chunk)
        session.commit()

        assert first_chunk.id is not None
        assert second_chunk.id is not None


def test_chunk_index_is_unique_within_wiki_page(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(markdown_content="# Project")
    wiki_page.chunks.extend(
        [
            Chunk(chunk_index=0, content="First"),
            Chunk(chunk_index=0, content="Duplicate"),
        ]
    )
    document.wiki_pages.append(wiki_page)

    with Session(engine) as session:
        session.add(document)
        with pytest.raises(IntegrityError):
            session.commit()


def test_deleting_document_deletes_related_wiki_pages_and_chunks(engine) -> None:
    document = Document(
        original_filename="project.pdf",
        storage_key="documents/project.pdf",
    )
    wiki_page = WikiPage(markdown_content="# Project")
    chunk = Chunk(chunk_index=0, content="Wiki content")
    wiki_page.chunks.append(chunk)
    document.wiki_pages.append(wiki_page)

    with Session(engine) as session:
        session.add(document)
        session.commit()
        document_id = document.id
        wiki_page_id = wiki_page.id
        chunk_id = chunk.id

        session.delete(document)
        session.commit()

        assert session.get(Document, document_id) is None
        assert session.get(WikiPage, wiki_page_id) is None
        assert session.get(Chunk, chunk_id) is None
