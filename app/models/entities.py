"""SQLModel table definitions for the document-to-wiki pipeline."""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, Relationship, SQLModel


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for model defaults."""

    return datetime.now(UTC)


def enum_column(enum_type: type[StrEnum], name: str) -> Column:
    """Build a portable string-backed SQL enum column.

    Native PostgreSQL enum types can be introduced later through a migration. A
    string-backed enum keeps the first schema equally testable on SQLite.
    """

    return Column(
        SAEnum(
            enum_type,
            name=name,
            native_enum=False,
            validate_strings=True,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        index=True,
    )


class ExtractionStatus(StrEnum):
    """Lifecycle of text extraction for an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class WikiStatus(StrEnum):
    """Editorial lifecycle of a generated or manually written wiki page."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Document(SQLModel, table=True):
    """An uploaded source PDF and its extracted text and metadata."""

    __tablename__ = "document"
    __table_args__ = (
        CheckConstraint("length(trim(original_filename)) > 0", name="ck_document_filename"),
        CheckConstraint("page_count >= 0", name="ck_document_page_count"),
        CheckConstraint(
            "extraction_status != 'completed' OR "
            "(raw_text IS NOT NULL AND extraction_data IS NOT NULL)",
            name="ck_document_completed_extraction_data",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    original_filename: str = Field(max_length=255, index=True)
    title: str | None = Field(default=None, max_length=500, index=True)
    author: str | None = Field(default=None, max_length=255)
    student_id: str | None = Field(default=None, max_length=100, index=True)
    academic_year: int | None = Field(default=None, index=True)
    storage_key: str = Field(max_length=1024, unique=True)
    raw_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    extraction_data: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    page_count: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.PENDING,
        sa_column=enum_column(ExtractionStatus, "extraction_status"),
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=utc_now),
    )

    wiki_pages: list["WikiPage"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    chunks: list["Chunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "overlaps": "chunks,wiki_page",
        },
    )


class WikiPage(SQLModel, table=True):
    """A versioned, reviewable Markdown representation of a document."""

    __tablename__ = "wiki_page"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_wiki_page_version"),
        CheckConstraint(
            "is_published = (status = 'published')",
            name="ck_wiki_page_publication_state",
        ),
        UniqueConstraint("document_id", "version", name="uq_wiki_page_document_version"),
        UniqueConstraint("id", "document_id", name="uq_wiki_page_id_document"),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    markdown_content: str = Field(sa_column=Column(Text, nullable=False))
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False))
    status: WikiStatus = Field(
        default=WikiStatus.DRAFT,
        sa_column=enum_column(WikiStatus, "wiki_status"),
    )
    generated_by_model: str | None = Field(default=None, max_length=255)
    is_published: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False, index=True),
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=utc_now),
    )

    document: Document = Relationship(back_populates="wiki_pages")
    chunks: list["Chunk"] = Relationship(
        back_populates="wiki_page",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "foreign_keys": "[Chunk.wiki_page_id, Chunk.document_id]",
            "overlaps": "chunks,document",
        },
    )


class Chunk(SQLModel, table=True):
    """A provenance-bearing text segment prepared for future embedding."""

    __tablename__ = "chunk"
    __table_args__ = (
        CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
        CheckConstraint(
            "source_page IS NULL OR source_page >= 1",
            name="ck_chunk_source_page",
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_chunk_token_count",
        ),
        CheckConstraint("length(trim(content)) > 0", name="ck_chunk_content"),
        UniqueConstraint("wiki_page_id", "chunk_index", name="uq_chunk_wiki_page_index"),
        ForeignKeyConstraint(
            ["wiki_page_id", "document_id"],
            ["wiki_page.id", "wiki_page.document_id"],
            name="fk_chunk_wiki_page_document",
            ondelete="CASCADE",
        ),
        Index(
            "uq_chunk_document_index_without_wiki",
            "document_id",
            "chunk_index",
            unique=True,
            sqlite_where=text("wiki_page_id IS NULL"),
            postgresql_where=text("wiki_page_id IS NULL"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("document.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    wiki_page_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            nullable=True,
            index=True,
        ),
    )
    chunk_index: int = Field(sa_column=Column(Integer, nullable=False))
    content: str = Field(sa_column=Column(Text, nullable=False))
    source_page: int | None = Field(default=None)
    token_count: int | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    document: Document = Relationship(
        back_populates="chunks",
        sa_relationship_kwargs={"overlaps": "chunks,wiki_page"},
    )
    wiki_page: WikiPage | None = Relationship(
        back_populates="chunks",
        sa_relationship_kwargs={
            "foreign_keys": "[Chunk.wiki_page_id, Chunk.document_id]",
            "overlaps": "chunks,document",
        },
    )
