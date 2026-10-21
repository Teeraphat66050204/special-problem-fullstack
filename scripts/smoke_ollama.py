"""Manually generate project metadata and a summary with configured Ollama."""

import argparse
import json
import sys
from pathlib import Path

from app.services.llm_service import LLMServiceError, generate_project_summary_result
from app.services.wiki_evidence import collect_wiki_source_evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate project metadata and a Thai summary from focused UTF-8 text"
    )
    parser.add_argument("source", type=Path, help="File containing extracted document text")
    arguments = parser.parse_args()

    try:
        source_text = arguments.source.read_text(encoding="utf-8")
        evidence = collect_wiki_source_evidence(source_text)
        result = generate_project_summary_result(source_text)
    except (OSError, UnicodeError, ValueError, LLMServiceError) as exc:
        print(f"Wiki generation failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "metadata": evidence.project_metadata(),
                "generated_summary": result.generated_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
