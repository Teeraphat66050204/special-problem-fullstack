# Project metadata and summary generation

This file keeps its original name so existing links remain valid. Production now
generates structured project metadata and a moderately detailed Thai project
summary. It no longer asks Ollama for a strict academic abstract.

## Output contract

The generation result has this primary structure:

```json
{
  "metadata": {
    "title_th": null,
    "title_en": null,
    "students": [],
    "student_ids": [],
    "academic_year": null,
    "advisor": null,
    "department": null,
    "faculty": null,
    "institution": null,
    "project_type": null
  },
  "generated_summary": "..."
}
```

Unavailable scalar metadata remains `null`; unavailable students and student IDs
remain empty arrays. Python extracts these fields deterministically from selected
title and project metadata pages. Metadata is never requested from the LLM.

The API temporarily retains `generated_abstract` and `generated_markdown` as
deprecated aliases of `generated_summary`. Existing top-level title, student,
advisor, year, and keyword fields also remain available while clients migrate to
the nested `metadata` object.

## Source selection

`app.services.wiki_source.prepare_wiki_source` remains deterministic and
document-wide. It preserves:

- Thai and English abstract discovery anywhere in the extracted document
- real Chapter 1 discovery with TOC-entry rejection
- the bounded Typhoon OCR fallback when native Chapter 1 text is damaged
- title and project metadata excerpts
- page labels, selected page numbers, and native/OCR/mixed provenance
- the real Chapter 2 boundary when it can be detected

The selector does not return to a fixed first-page window. Its existing character
budgets retain evidence for background, objectives, scope, methods, tools, and
supported results or conclusions when those sections are present.

## Runtime prompt and parsing

`app.prompts.project_summary.build_project_summary_prompt` inserts the summary-only
focused source between `<SOURCE_DOCUMENT>` tags. The source prefers the Thai
abstract, falls back to the English abstract, retains relevant Chapter 1 sections,
and excludes title/student/institution pages used by deterministic metadata
extraction. The prompt asks only for a source-supported Thai summary, with a soft
target of roughly 700–1200 characters. It prioritizes actual results, formulas,
recommended values, and important numbers, and distinguishes them from objectives
and expected benefits. Project-summary requests set `num_predict` from
`OLLAMA_NUM_PREDICT`, which defaults to 800.

The model returns plain text. Production trims boundary whitespace, preserves the
raw response for diagnostics and benchmark artifacts, and never fails merely
because the response is not JSON. Python combines the validated summary with
deterministic metadata when building the API response.

Ollama calls remain non-streaming. Request failures carry model name, elapsed
time, HTTP status and response body when available, underlying exception
class/message, and connection/reset/protocol/runtime classification. The API
keeps its existing public error mapping while logging these details.

## Validation

The summary validator rejects obvious contract failures:

- a non-string or empty summary
- a summary shorter than 300 characters
- a summary with no Thai text
- multiple Markdown headings/list items or a code fence inside the summary
- assistant-style commentary, prompt/source references, or offers to help
- explicit missing-information markers from the old Wiki pipeline

The 700–1200 character range is guidance only. Otherwise useful summaries slightly
outside it remain valid. Multiple natural paragraphs, different sentence shapes,
and omission of nonessential source details are not rejected by themselves.

## Compatibility and historical evaluation

`generate_project_summary_result` is the primary LLM service and returns summary
text plus its raw response and cleanup audit. The old
`generate_abstract`, `generate_abstract_result`, `generate_wiki`, and
`generated_abstract` names remain compatibility adapters that expose only the
summary text. `app.prompts.abstract_generation` likewise re-exports deprecated
prompt and validation names.

The seven-section Wiki evaluator, curated `abstract_th` GroundTruth field, and
previous benchmark artifacts retain their historical meaning. They are not
rewritten as project-summary results.
