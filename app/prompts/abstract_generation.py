"""Deprecated compatibility names for the production project-summary contract."""

import re

from app.prompts.project_summary import (
    PROJECT_SUMMARY_GENERATION_INSTRUCTIONS,
    ProjectSummaryValidation,
    build_project_summary_prompt,
    validate_project_summary,
)

ABSTRACT_GENERATION_INSTRUCTIONS = PROJECT_SUMMARY_GENERATION_INSTRUCTIONS
AbstractValidation = ProjectSummaryValidation
build_abstract_generation_prompt = build_project_summary_prompt
validate_abstract = validate_project_summary


def normalize_abstract(text: str) -> str:
    """Deprecated plain-text cleanup retained for callers outside production."""
    text = text.strip()
    return re.sub(r"^บทคัดย่อ\s*[:：]\s*", "", text, count=1).strip()
