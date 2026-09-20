"""Validate and summarize the benchmark human-evaluation worksheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_wiki_models import (
    aggregate_human_evaluations,
    read_human_evaluations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate 1-4 human Wiki benchmark ratings"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("benchmark/benchmark_human_evaluation.csv"),
    )
    arguments = parser.parse_args(argv)
    try:
        rows = read_human_evaluations(arguments.csv_path)
        summary = aggregate_human_evaluations(rows)
    except (OSError, ValueError) as exc:
        print(f"Cannot summarize human evaluations: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
