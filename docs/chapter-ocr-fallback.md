# Targeted Chapter 1 OCR fallback

Native/existing extracted text remains the primary path. A successful Chapter 1
scan returns immediately without loading the provider, ranking OCR candidates,
rendering pages, or making OCR requests.

When Chapter 1 is missing, `recover_chapter_one` ranks plausible pages using
native text only: 1.1 patterns, short chapter/introduction fragments (including
known corrupted font glyphs), printed body page 1, degraded text, proximity to
detected abstract/TOC regions, and neighbours of strong candidates. These signals
never repair text or establish a chapter by themselves. Known TOC/list-of-figures
pages are excluded before OCR.

`CHAPTER_OCR_MAX_PAGES` defaults to **6** and accepts **0–6**. Zero disables the
new fallback. It is an additional budget separate from the unchanged front-matter
`OCR_MAX_PAGES` scope. `OCR_PROVIDER=disabled` disables both. Each distinct page
attempt consumes one slot, including errors, Chapter 2 boundary discovery, and
degraded Chapter 1 continuation pages. There are no automatic retries or
unbounded expansion. The existing Typhoon adapter accepts later physical pages
but still caps each request at six pages. No second OCR provider was introduced.

After each candidate is OCRed, the same native-path chapter heuristics run on
the trial page text. OCR TOCs and insufficient chapter evidence are rejected.
Once Chapter 1 is found, remaining slots may recover a plausible Chapter 2
boundary, then degraded continuation pages inside that verified span. Only OCR
inside the verified span and at its Chapter 2 boundary is retained; failed
front-matter candidates cannot replace abstracts. If the end is unknown, only
the opening and its immediate supporting page may be retained. The final
retained extraction is validated again.

Recovered pages retain exact normalized OCR text, original native text, and
`ocr` provenance using the existing `PdfPageText` model. No persistence schema
change is required. Provider failure stops further requests, adds a safe existing
OCR warning, and preserves usable native/abstract evidence or an already verified
opening. If recovery fails, abstract-only source selection continues normally.

## Integration and diagnostics

The upload path runs this fallback after the existing front-matter OCR stage,
while its temporary PDF is still available. It persists recovered page text using
the existing serializer, then deletes the temporary file as before. Subsequent
generation reuses stored OCR text; it performs no OCR or publication.

Already-persisted documents whose original PDFs have been discarded require
re-upload to benefit. This change does not silently locate a different PDF or
modify old stored extractions. The standalone native-text selector remains pure.

Upload logs contain structured diagnostics: status, detection method, candidate
page/score/reasons, attempted pages, retained pages, final opening/boundary, and
rejection reasons. Source-selection diagnostics also expose `chapter_method`.
Neither credentials nor PDF text are included in those diagnostics.

```powershell
# Native scan and candidate planning only; no OCR or generation.
python -m scripts.diagnose_abstract_source data/sample

# Native scan, then bounded Typhoon OCR only for missing Chapter 1.
python -m scripts.diagnose_abstract_source data/sample --chapter-ocr --output docs/new-chapter-diagnostics.json
```

The output path must be new. Existing reports cannot be overwritten by this
command. The tool never calls Ollama or the abstract-generation service. Candidate
ranking and chapter selection are deterministic; remote OCR transcription is not
guaranteed deterministic.

## Measured result

The [new 20-PDF report](chapter-ocr-diagnostics.json) preserves the earlier
[native-only baseline](abstract-source-diagnostics.json). All **20/20** documents
now have a detected Chapter 1 and Chapter 2 boundary. Seventeen used native text
and made **zero** fallback OCR calls. The three affected documents used **11 OCR
pages total**, with no abstract/LLM generation:

- **064:** Chapter 1 pages **11–12**, Chapter 2 starts **13**. OCR request order:
  **11, 13, 12** (3 pages).
- **123:** Chapter 1 pages **20–23**, Chapter 2 starts **24**. OCR request order:
  **20, 24, 21, 22, 23** (5 pages).
- **254:** Chapter 1 pages **12–13**, Chapter 2 starts **14**. OCR request order:
  **12, 14, 13** (3 pages).

Chapter 2 pages provide boundaries and are excluded from the chapter source.
The character budget and relevant-section filter still control which recovered
Chapter 1 pages actually contribute to `selected_pages`.

## Remaining limits

- Candidate ranking can miss headings with no useful text/layout transition
  signals, unusual chapter formatting, or a necessary page outside the budget.
- OCR remains dependent on Typhoon availability and transcription accuracy.
  Failures are non-fatal; no unlimited retry/recovery guarantee is made.
- The six-page budget is shared with boundary and continuation recovery. A long
  damaged chapter can remain partly native, or its Chapter 2 end can remain unknown.
- Existing stored documents need access to the original PDF through re-upload.
- Detection scores explain ordering; they are not calibrated confidence values.

## Changed files in this follow-up

- `.env.example`, `app/config.py`: configurable hard-bounded chapter OCR budget.
- `app/services/chapter_ocr.py`: candidate ranking, adaptive bounded recovery,
  provenance retention, and diagnostics.
- `app/services/wiki_source.py`: shared unchanged heading heuristics and detection method.
- `app/services/ocr_service.py`: reusable validated OCR-page invocation with stream restoration.
- `app/services/typhoon_ocr.py`: allow targeted later pages while limiting request count.
- `app/api/upload.py`: recovery while the temporary PDF is available; structured logging.
- `scripts/diagnose_abstract_source.py`: optional chapter OCR and non-overwriting reports.
- `tests/test_chapter_ocr.py`, `tests/test_typhoon_ocr.py`, `tests/test_upload_api.py`:
  recovery, TOCs, budgets, failure paths, native bypass, persistence reuse, and fixtures.
- `README.md`, `docs/ocr.md`, `docs/abstract-generation.md`, this guide, and
  `docs/chapter-ocr-diagnostics.json`: current behavior and measured results.

Historical benchmark artifacts/results, PDF extraction, and persistence schema
remain untouched. No commit or push was performed.

Final validation: **379 tests passed** (two existing FastAPI/Starlette deprecation
warnings), Ruff lint passed, Ruff format check passed (63 Python files), and
`git diff --check` passed. Regression coverage includes damaged native headings,
OCR TOCs, total budget exhaustion, non-fatal provider failure, no OCR on native
success, title-only openings needing next-page OCR evidence, later-page Typhoon
requests, upload-to-generation persistence reuse, and native fixture rankings for
064/123/254.

<!-- Full table and final checks follow. -->

| Document | Chapter 1 pages | Chapter 2 start | Detection | OCR pages (request order) |
|---|---|---|---|---|
| document_007.pdf | 7–8 | 9 | native | none |
| document_009.pdf | 10–12 | 13 | native | none |
| document_022.pdf | 12–12 | 13 | native | none |
| document_064.pdf | 11–12 | 13 | OCR fallback | 11, 13, 12 |
| document_071.pdf | 16–18 | 19 | native | none |
| document_114.pdf | 16–21 | 22 | native | none |
| document_123.pdf | 20–23 | 24 | OCR fallback | 20, 24, 21, 22, 23 |
| document_127.pdf | 12–17 | 18 | native | none |
| document_137.pdf | 15–23 | 24 | native | none |
| document_191.pdf | 11–14 | 15 | native | none |
| document_213.pdf | 14–15 | 16 | native | none |
| document_222.pdf | 11–13 | 14 | native | none |
| document_251.pdf | 25–27 | 28 | native | none |
| document_253.pdf | 15–16 | 17 | native | none |
| document_254.pdf | 12–13 | 14 | OCR fallback | 12, 14, 13 |
| document_266.pdf | 16–18 | 19 | native | none |
| document_283.pdf | 13–14 | 15 | native | none |
| document_295.pdf | 13–16 | 17 | native | none |
| document_335.pdf | 18–19 | 20 | native | none |
| document_339.pdf | 14–19 | 20 | native | none |
