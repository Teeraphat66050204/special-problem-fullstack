"""Deterministic Markdown-aware Wiki chunking tests."""

import pytest

from app.services.wiki_chunker import WikiChunk, chunk_wiki_markdown


def test_short_markdown_returns_one_exact_chunk_with_provenance() -> None:
    markdown = "# โครงงาน\n\n## ภาพรวมโครงงาน\nเนื้อหาสั้น"

    chunks = chunk_wiki_markdown(markdown, document_id="document_071", wiki_page_id=12)

    assert chunks == (
        WikiChunk(
            chunk_index=0,
            content=markdown,
            section_title=None,
            headings=("โครงงาน", "ภาพรวมโครงงาน"),
            char_count=len(markdown),
            start_offset=0,
            end_offset=len(markdown),
            document_id="document_071",
            wiki_page_id=12,
        ),
    )


def test_multiple_h2_sections_keep_original_order_and_indexes() -> None:
    markdown = (
        "# Wiki\n\n"
        "## ภาพรวมโครงงาน\nภาพรวมสั้น\n\n"
        "## วิธีดำเนินงาน\nรายละเอียดวิธีดำเนินงาน\n\n"
        "## ผลลัพธ์\nผลลัพธ์จากเอกสาร"
    )

    chunks = chunk_wiki_markdown(markdown, chunk_size=70, chunk_overlap=0)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.start_offset for chunk in chunks] == sorted(
        chunk.start_offset for chunk in chunks
    )
    assert "".join(chunk.content for chunk in chunks) == markdown
    assert chunks[0].headings == ("Wiki", "ภาพรวมโครงงาน")
    assert tuple(heading for chunk in chunks for heading in chunk.headings) == (
        "Wiki",
        "ภาพรวมโครงงาน",
        "วิธีดำเนินงาน",
        "ผลลัพธ์",
    )


def test_long_section_splits_recursively_and_keeps_heading_context() -> None:
    paragraphs = [f"ย่อหน้าที่ {index} อธิบายข้อมูลจากเอกสารต้นฉบับอย่างครบถ้วน" for index in range(8)]
    markdown = "## วิธีดำเนินงาน\n" + "\n\n".join(paragraphs)

    chunks = chunk_wiki_markdown(markdown, chunk_size=115, chunk_overlap=0)

    assert len(chunks) > 1
    assert chunks[0].content.startswith("## วิธีดำเนินงาน\n")
    assert all(chunk.section_title == "วิธีดำเนินงาน" for chunk in chunks)
    assert all(chunk.headings == ("วิธีดำเนินงาน",) for chunk in chunks)
    assert all(chunk.char_count <= 115 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == markdown


def test_overlap_is_boundary_aligned_bounded_and_not_a_duplicate_chunk() -> None:
    markdown = "## Results\n" + " ".join(f"measurement-{index}" for index in range(35))

    chunks = chunk_wiki_markdown(markdown, chunk_size=100, chunk_overlap=24)

    assert len(chunks) > 2
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous.start_offset < following.start_offset
        assert following.start_offset <= previous.end_offset
        overlap = previous.end_offset - following.start_offset
        assert 0 < overlap <= 24
        assert previous.content[-overlap:] == following.content[:overlap]
        assert (previous.start_offset, previous.end_offset) != (
            following.start_offset,
            following.end_offset,
        )


def test_thai_english_terms_numbers_and_punctuation_are_never_rewritten() -> None:
    markdown = (
        "## ผลลัพธ์\n"
        "สูตรมี POE 10%, SLS 1%, EDTA 0.1% และ Wetting agent 0.5% ที่ pH 11. "
        "ประสิทธิภาพ (%CEF) สูงกว่าผลิตภัณฑ์ 2 ชนิด! "
    ) * 5

    chunks = chunk_wiki_markdown(markdown, chunk_size=145, chunk_overlap=0)

    assert "".join(chunk.content for chunk in chunks) == markdown
    reconstructed = "".join(chunk.content for chunk in chunks)
    for exact_value in ("POE 10%", "SLS 1%", "EDTA 0.1%", "0.5%", "pH 11", "%CEF"):
        assert exact_value in reconstructed


def test_markdown_list_items_prefer_line_boundaries() -> None:
    items = [f"- ขั้นตอนที่ {index}: ใช้สารตัวอย่าง {index}%" for index in range(1, 13)]
    markdown = "## วิธีดำเนินงาน\n" + "\n".join(items)

    chunks = chunk_wiki_markdown(markdown, chunk_size=95, chunk_overlap=0)

    assert "".join(chunk.content for chunk in chunks) == markdown
    assert all(chunk.content.endswith("\n") for chunk in chunks[:-1])
    assert all(item in "".join(chunk.content for chunk in chunks) for item in items)


@pytest.mark.parametrize("markdown", ["", "   ", "\n\t\n"])
def test_empty_or_whitespace_markdown_returns_empty_tuple(markdown: str) -> None:
    assert chunk_wiki_markdown(markdown) == ()


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_wiki_markdown("# Valid", chunk_size=chunk_size, chunk_overlap=overlap)


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        chunk_wiki_markdown(None)  # type: ignore[arg-type]


def test_repeated_runs_are_deterministic() -> None:
    markdown = (
        "# Wiki\n\n## ภาพรวมโครงงาน\nข้อมูลภาษาไทย and English technical terms.\n\n"
        "## ผลลัพธ์\n" + "ผลลัพธ์ 98.75% จาก Model-X. " * 20
    )
    arguments = {
        "document_id": 71,
        "wiki_page_id": 9,
        "chunk_size": 130,
        "chunk_overlap": 20,
    }

    assert chunk_wiki_markdown(markdown, **arguments) == chunk_wiki_markdown(markdown, **arguments)
