"""Inspect deterministic chunks from a local UTF-8 Wiki Markdown file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.services.wiki_chunker import chunk_wiki_markdown


def run(markdown_path: Path) -> int:
    """Print every Wiki chunk and return a shell exit code."""

    if not markdown_path.is_file():
        print(f"Markdown file not found: {markdown_path}", file=sys.stderr)
        return 1

    try:
        markdown_content = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"Could not read Markdown file as UTF-8: {exc}", file=sys.stderr)
        return 1

    chunks = chunk_wiki_markdown(markdown_content)
    for chunk in chunks:
        covered_headings = ", ".join(chunk.headings) if chunk.headings else "<none>"
        section_title = chunk.section_title or "<none>"

        print(f"=== Chunk {chunk.chunk_index} ===")
        print(f"chunk_index: {chunk.chunk_index}")
        print(f"section_title: {section_title}")
        print(f"covered_headings: {covered_headings}")
        print(f"char_count: {chunk.char_count}")
        print(f"start_offset: {chunk.start_offset}")
        print(f"end_offset: {chunk.end_offset}")
        print("content:")
        sys.stdout.write(chunk.content)
        if not chunk.content.endswith("\n"):
            print()
        print(f"=== End chunk {chunk.chunk_index} ===")

    print(f"Total chunks: {len(chunks)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect chunks produced from a UTF-8 Wiki Markdown file"
    )
    parser.add_argument("markdown", type=Path, help="Path to a Wiki Markdown file")
    arguments = parser.parse_args(argv)
    return run(arguments.markdown)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
