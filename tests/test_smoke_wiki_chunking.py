from pathlib import Path

from scripts import smoke_wiki_chunking


def test_smoke_prints_chunk_metadata_and_exact_content(tmp_path: Path, capsys, monkeypatch) -> None:
    markdown = "# โครงงาน Test\n\n## ภาพรวมโครงงาน\nเนื้อหา 50% with FastAPI"
    markdown_path = tmp_path / "wiki.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    received: list[str] = []
    real_chunker = smoke_wiki_chunking.chunk_wiki_markdown

    def recording_chunker(content: str):
        received.append(content)
        return real_chunker(content)

    monkeypatch.setattr(smoke_wiki_chunking, "chunk_wiki_markdown", recording_chunker)

    assert smoke_wiki_chunking.run(markdown_path) == 0

    captured = capsys.readouterr()
    assert received == [markdown]
    assert "chunk_index: 0" in captured.out
    assert "section_title: <none>" in captured.out
    assert "covered_headings: โครงงาน Test, ภาพรวมโครงงาน" in captured.out
    assert f"char_count: {len(markdown)}" in captured.out
    assert "start_offset: 0" in captured.out
    assert f"end_offset: {len(markdown)}" in captured.out
    assert markdown in captured.out
    assert "Total chunks: 1" in captured.out
    assert captured.err == ""


def test_smoke_reports_missing_markdown_file(tmp_path: Path, capsys) -> None:
    missing_path = tmp_path / "missing.md"

    assert smoke_wiki_chunking.run(missing_path) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Markdown file not found: {missing_path}" in captured.err


def test_smoke_reports_invalid_utf8(tmp_path: Path, capsys) -> None:
    markdown_path = tmp_path / "invalid.md"
    markdown_path.write_bytes(b"\xff\xfe")

    assert smoke_wiki_chunking.run(markdown_path) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Could not read Markdown file as UTF-8:" in captured.err


def test_smoke_prints_zero_chunks_for_whitespace(tmp_path: Path, capsys) -> None:
    markdown_path = tmp_path / "empty.md"
    markdown_path.write_text(" \n\t", encoding="utf-8")

    assert smoke_wiki_chunking.run(markdown_path) == 0

    captured = capsys.readouterr()
    assert captured.out == "Total chunks: 0\n"
    assert captured.err == ""
