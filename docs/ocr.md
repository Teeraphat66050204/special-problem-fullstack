# Conditional Typhoon OCR

PyMuPDF remains the primary PDF extractor. The application assesses native text
quality before loading front-matter evidence. It calls Typhoon only for degraded
pages selected from pages 1–6; `OCR_MAX_PAGES` is validated with a hard maximum
of 6 in both configuration and the OCR service. If Chapter 1 is still missing,
uploads may additionally OCR at most `CHAPTER_OCR_MAX_PAGES` targeted pages
anywhere in the PDF. See [chapter recovery](chapter-ocr-fallback.md).

## Setup

Copy `.env.example` to `.env` and configure:

```dotenv
OCR_PROVIDER=typhoon
TYPHOON_API_KEY=
TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1
TYPHOON_OCR_MODEL=typhoon-ocr
OCR_TIMEOUT_SECONDS=120
OCR_MAX_PAGES=6
CHAPTER_OCR_MAX_PAGES=6
```

Put the API key only in the local `.env` or the deployment secret store. `.env`
is ignored by Git. Do not place a real key in `.env.example`, source code, logs,
or documentation. Set `OCR_PROVIDER=disabled` when OCR is not wanted; no key is
then required.

The adapter renders each requested PDF page with PyMuPDF and calls Typhoon's
OpenAI-compatible `chat/completions` endpoint. The base URL should normally end
in `/v1`. The official model identifier is `typhoon-ocr`.

## Failure and provenance behavior

If a key is missing when degraded text needs OCR, the upload succeeds with an
`ocr_configuration` warning. Network, timeout, HTTP, rendering, and invalid
response failures produce an `ocr_failed` warning. These warnings contain no API
response bodies, credentials, PDF text, or stack traces. In both cases, native
PyMuPDF text remains selected.

Successful front-matter OCR is compared with native text using deterministic quality
checks. Better OCR is selected with `ocr` provenance; complementary text may be
kept as `mixed`; worse OCR is stored for inspection but the selected text and
provenance remain `pymupdf`. The production upload and benchmark harness both use
this same front-matter fallback service. Chapter recovery additionally requires
the shared real-chapter heuristics, uses its own total page budget, and runs only
on upload/local diagnostics. Historical benchmarks do not use chapter recovery.
