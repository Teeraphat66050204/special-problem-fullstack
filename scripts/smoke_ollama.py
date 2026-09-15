"""Manually run the standalone Wiki service against a local Ollama instance."""

import argparse
import sys
from pathlib import Path

from app.services.llm_service import LLMServiceError, generate_wiki


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Wiki Markdown from a UTF-8 text file")
    parser.add_argument("source", type=Path, help="File containing extracted document text")
    arguments = parser.parse_args()

    try:
        source_text = arguments.source.read_text(encoding="utf-8")
        markdown = generate_wiki(source_text)
    except (OSError, UnicodeError, ValueError, LLMServiceError) as exc:
        print(f"Wiki generation failed: {exc}", file=sys.stderr)
        return 1

    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
