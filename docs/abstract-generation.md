# Thai abstract generation

The follow-up [targeted Chapter 1 OCR fallback](chapter-ocr-fallback.md) recovers
all three previously undetected chapters. The native-only baseline and initial
verification below are retained for comparison; see the follow-up report for
current OCR-assisted results and checks.

`POST /api/wiki/generate` now returns a concise Thai academic abstract draft in
`generated_abstract`. `generated_markdown` is a deprecated compatibility alias
containing the same plain text. `structure_valid` reports the abstract format
check. Generation reads an extracted Document but does not persist or publish
its output. The separate human-reviewed Wiki publication and chunking contracts
remain unchanged; they still require canonical Wiki Markdown.

## Source selection

`prepare_wiki_source` scans **all** `extraction.pages`, using each page's final
native/OCR text and provenance. It never uses `extraction.full_text`. The existing
PyMuPDF extraction, front-matter OCR policy, and Document persistence are unchanged.
Upload can additionally use the bounded chapter fallback described above.

- Standalone Thai/English abstract headings support spacing, case, and Markdown
  emphasis. TOC pages and page-number/dotted-leader entries are rejected. Thai
  evidence is retained first, followed by English evidence. Abstract continuation
  stops at a new structural heading, completed keywords, incompatible language,
  or the character budget.
- Chapter headings support `บทที่ 1`, `บท ที่ 1`, `บทที่ ๑`, `บท 1`, `Chapter 1`,
  `CHAPTER 1`, and `Chapter One`. Chapter 2 uses the same rejection rules.
- Real chapter candidates require substantial following text (at least 100
  characters excluding recognized headings/leaders), plus a nearby subsection,
  introduction title, or several following body lines. Chapter 1 must occur in
  the first eight nonempty lines of a page, **not** the first eight PDF pages.
  A Chapter 2 boundary farther down a page requires nearby structural evidence.
- Reject TOC headings, multiple dotted leaders, several chapter entries, dense
  numbered TOC entries, and chapter titles ending with a page reference. A first
  string occurrence alone never establishes a real chapter.
- Parse numbered Chapter 1 sections and prioritize background/1.1, objectives,
  scope, expected benefits, and methods. Split number/title lines are supported.
  Long recognized sections share the chapter budget. With weak subsection
  detection, keep a bounded chapter slice. Without a detected Chapter 2, report
  the end as unknown and retain bounded excerpts.
- Preserve optional labelled metadata throughout the document, plus bounded
  cover-page context. Metadata-only documents are not sufficient evidence for
  generation. If neither an abstract nor a real Chapter 1 is detected, return a
  controlled 422 without calling Ollama.

Default character budgets: 24,000 total including provenance labels, 6,500 per
abstract language, 12,000 Chapter 1, 2,400 per recognized subsection, and 3,500
metadata. Metadata space is reserved within the total. There is no selected-page
count cap. `selected_pages` contains sorted unique physical PDF pages that
actually contribute text; a page can contribute several labelled excerpts.
Duplicate-page evidence is combined rather than overwritten.

## Output contract

`build_abstract_generation_prompt` asks for one cohesive Thai academic paragraph,
conceptually 150–250 words. It covers background, objectives, method/tools,
results, and significance only when source-supported. Missing facts are omitted;
expected benefits cannot become measured outcomes. It prohibits Markdown,
lists, headings, preambles, repetitions, and missing-data markers.

`normalize_abstract` only trims surrounding whitespace and removes an exact
leading `บทคัดย่อ:` label. It never rewrites facts or joins separate paragraphs.
`validate_abstract` rejects empty/non-Thai output, headings, lists, section labels,
markup, obvious meta commentary, old missing-data markers, blank-separated
paragraphs, and output exceeding 4,000 characters. Single soft line wraps are
allowed. The validator checks formatting, not factual accuracy or writing quality.

The Ollama URL, model, temperature, and timeout remain Settings-driven.
`OllamaClient.generate_result(..., think=False)` remains supported; thinking is
never disabled globally. Metadata extraction failures return empty metadata and
do not prevent generation. `generate_wiki` and `generate_wiki_result` remain
compatibility wrappers around abstract generation; new code should use
`generate_abstract` / `generate_abstract_result`.

## Examples

Illustrative focused source (not a live model run):

```text
Source page 12 (Thai abstract):
บทคัดย่อ
โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยโดยใช้การค้นหาคำสำคัญ

Source page 20 (Chapter 1):
1.1 ความเป็นมา
ผู้ใช้ต้องค้นหาเอกสารภาษาไทยที่เกี่ยวข้องกับคำค้น

Source page 21 (Chapter 1):
1.2 วัตถุประสงค์
พัฒนาระบบค้นหาคำสำคัญและจัดแสดงเอกสารที่ตรงกับคำค้น

Source page 1 (project metadata):
ชื่อโครงงาน ระบบค้นคืนเอกสารภาษาไทย
```

Valid concise output supported by that illustrative source:

> โครงงานนี้พัฒนาระบบค้นคืนเอกสารภาษาไทยเพื่อช่วยให้ผู้ใช้เข้าถึงเอกสารที่เกี่ยวข้องกับคำค้น โดยมีวัตถุประสงค์เพื่อรองรับการค้นหาข้อมูลผ่านคำสำคัญและจัดแสดงเอกสารที่ตรงกับความต้องการของผู้ใช้ ระบบใช้การค้นหาคำสำคัญเป็นแนวทางในการค้นคืนและแสดงเอกสารที่สัมพันธ์กับคำค้น

The example omits results because the illustrative source reports none. Short
sources may warrant shorter abstracts than the approximate target length.

## Legacy benchmark boundary

Existing benchmark outputs, evaluations, replay artifacts, and human scores
remain historical seven-section Wiki results. They have not been changed or
reinterpreted. `benchmark_wiki_models.py` and `evaluate_wiki_generation.py` use
the explicitly frozen `legacy_wiki_source`, old prompt, old validator, and legacy
finalizer/generation entry point. No production abstract uses that finalizer.
A new abstract-specific benchmark and human evaluation will be needed later,
with distinct output locations. No model benchmark was run for this change.

## Deterministic fixture diagnostics

Run `python -m scripts.diagnose_abstract_source data/sample` to print diagnostics
without OCR, Ollama, or writes. The checked-in
[JSON report](abstract-source-diagnostics.json) records abstract pages, Chapter 1
start/end, Chapter 2 start, contributing source pages, character counts, and exact
acceptance/rejection signals for all 20 local PDFs.

All 20 supplied usable abstract evidence. Seventeen supplied a detectable real
Chapter 1, including `document_251.pdf` at physical page 25. Documents 064, 123,
and 254 have damaged native Thai headings; Chapter 1 remains undetected and
English abstracts provide fallback evidence. The existing OCR scope is unchanged,
so diagnostics do not promise recovery of damaged later chapter pages.

Other limitations: unusual unnumbered sections, heavily corrupted headings,
chapter openings below the first several nonempty lines, and abstracts without
recognizable boundaries can reduce recall or require bounded truncation. Signals
are heuristic explanations, not calibrated confidence probabilities. Budget
truncation can omit later facts; reviewers must check the draft against its source.
Live model output quality has not been evaluated in this task.

<!-- Diagnostic table is generated from the separate deterministic JSON report. -->

| PDF | Abstract pages | Chapter 1 start–end | Chapter 2 start | Selected source pages |
|---|---|---|---|---|
| document_007.pdf | 5 | 7–8 | 9 | 1, 2, 3, 4, 5, 7, 8, 62 |
| document_009.pdf | 5 | 10–12 | 13 | 1, 2, 3, 4, 5, 10, 11 |
| document_022.pdf | 4, 5 | 12–12 | 13 | 1, 2, 4, 5, 12 |
| document_064.pdf | 5 | undetected | undetected | 1, 2, 5 |
| document_071.pdf | 4, 5, 6, 7 | 16–18 | 19 | 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 70 |
| document_114.pdf | 5, 6 | 16–21 | 22 | 1, 2, 3, 4, 5, 6, 16, 18, 19, 158 |
| document_123.pdf | 6, 7 | undetected | undetected | 1, 2, 6, 7 |
| document_127.pdf | 5 | 12–17 | 18 | 1, 2, 3, 4, 5, 12, 13, 14, 15 |
| document_137.pdf | 4, 5, 6, 7 | 15–23 | 24 | 1, 2, 4, 5, 6, 7, 15, 16, 17, 20, 21 |
| document_191.pdf | 4, 5 | 11–14 | 15 | 1, 2, 4, 5, 11, 12, 13 |
| document_213.pdf | 6, 7 | 14–15 | 16 | 1, 2, 3, 4, 5, 6, 7, 14, 15 |
| document_222.pdf | 4, 5 | 11–13 | 14 | 1, 2, 3, 4, 5, 11, 12, 13 |
| document_251.pdf | 5, 6 | 25–27 | 28 | 1, 2, 5, 6, 25, 26, 27 |
| document_253.pdf | 5 | 15–16 | 17 | 1, 2, 3, 4, 5, 15, 16 |
| document_254.pdf | 5 | undetected | undetected | 1, 2, 5 |
| document_266.pdf | 4, 5 | 16–18 | 19 | 1, 2, 3, 4, 5, 16, 17, 18, 106 |
| document_283.pdf | 4, 5 | 13–14 | 15 | 1, 2, 3, 4, 5, 6, 13, 14 |
| document_295.pdf | 4, 5 | 13–16 | 17 | 1, 2, 3, 4, 5, 13, 14, 15, 96 |
| document_335.pdf | 5 | 18–19 | 20 | 1, 2, 3, 4, 5, 18, 19, 34, 35, 36, 37, 39 |
| document_339.pdf | 5 | 14–19 | 20 | 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19 |

## Verification and changed files

Validation: **357 tests passed**, including all existing extraction/OCR tests,
publication tests, and historical benchmark unit tests. Ruff lint and formatting
checks pass; `git diff --check` passes. Two existing FastAPI/Starlette dependency
deprecation warnings remain. No expensive inference, OCR calls, full LLM
benchmark, commit, or push was performed.

New source-selection tests cover Chapter 1 at pages 5/15/20, all requested
heading variants, TOCs with and without leaders, real Chapter 2 boundaries,
late abstracts, abstract-only/chapter-only/missing-source cases, character
budgets, late relevant subsections, split headings, OCR provenance, and a real
page-25 fixture. Output tests cover valid prose, all requested invalid forms,
minimal cleanup, and prompt grounding instructions. Service/API tests verify
configured HTTP behavior, optional thinking, invalid-output handling, optional
metadata failure, actual contributing pages, and no generation persistence.

Files changed for this task (pre-existing publishing and benchmark import work
was preserved):

- Production: `app/services/wiki_source.py`, `app/prompts/abstract_generation.py`,
  `app/services/llm_service.py`, `app/api/wiki.py`, `app/services/wiki_evidence.py`.
- Legacy boundary: `app/services/legacy_wiki_source.py`,
  `app/services/wiki_output.py`, `app/prompts/wiki_generation.py`,
  `scripts/benchmark_wiki_models.py`, `scripts/evaluate_wiki_generation.py`.
- Diagnostics/smoke tools: `scripts/diagnose_abstract_source.py`,
  `scripts/smoke_pdf_to_wiki.py`, `scripts/smoke_ollama.py`.
- Tests: `tests/test_abstract_source.py`, `tests/test_abstract_output.py`,
  `tests/test_llm_service.py`, `tests/test_wiki_api.py`,
  `tests/test_wiki_publish_api.py`, `tests/test_wiki_evidence.py`,
  `tests/test_wiki_output.py`, `tests/test_dataset_and_wiki_source.py`,
  `tests/test_smoke_pdf_to_wiki.py`. Historical selector/finalizer tests explicitly
  exercise legacy entry points; production source tests use the new selector.
- Documentation: `README.md`, `docs/abstract-generation.md`,
  `docs/abstract-source-diagnostics.json`, `docs/wiki-generation-prompt.md`,
  `docs/wiki-generation-evaluation.md`, `docs/wiki-model-benchmark.md`.
