"""Schema and loaders for curated sample-document ground truth."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GroundTruthRecord(BaseModel):
    """Title-page metadata and abstracts for one sample document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=r"^document_\d{3}$")
    title_th: str | None = None
    title_en: str | None = None
    students: list[str] = Field(default_factory=list)
    advisor: str | None = None
    academic_year: int | None = None
    abstract_th: str | None = None
    abstract_en: str | None = None
    keywords: list[str] = Field(default_factory=list)
    raw_text: str | None = None


class DatasetEntry(BaseModel):
    """Relative paths joining a sample PDF to its ground-truth record."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=r"^document_\d{3}$")
    pdf: str
    groundtruth: str

    @model_validator(mode="after")
    def paths_match_document_id(self) -> "DatasetEntry":
        if self.pdf != f"sample/{self.document_id}.pdf":
            raise ValueError("PDF path must match document ID")
        if self.groundtruth != f"groundtruth/{self.document_id}.json":
            raise ValueError("Ground-truth path must match document ID")
        return self


class DatasetIndex(BaseModel):
    """Versioned list of sample documents available for local evaluation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    documents: list[DatasetEntry]

    @model_validator(mode="after")
    def document_ids_are_unique(self) -> "DatasetIndex":
        identifiers = [entry.document_id for entry in self.documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Dataset index contains duplicate document IDs")
        return self


def load_groundtruth(path: str | Path) -> GroundTruthRecord:
    """Read UTF-8 JSON and ensure its document ID matches the filename."""

    source = Path(path)
    record = GroundTruthRecord.model_validate_json(source.read_text(encoding="utf-8"))
    if source.stem != record.document_id:
        raise ValueError(f"Ground-truth document ID does not match filename: {source.name}")
    return record


def load_dataset_index(path: str | Path) -> DatasetIndex:
    """Read the small index without requiring ignored PDF files to be present."""

    return DatasetIndex.model_validate_json(Path(path).read_text(encoding="utf-8"))
