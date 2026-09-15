# Sample dataset and focused Wiki input

The reusable local dataset has 20 matching document IDs:

```text
data/
  index.json
  sample/                 # local PDFs, ignored by Git
    document_007.pdf
    ...
  groundtruth/            # small UTF-8 JSON records
    document_007.json
    ...
```

`data/index.json` maps each `document_id` to a relative PDF path and a
ground-truth JSON path. The index and JSON records can be loaded without the
PDFs, so normal unit tests remain small. The PDFs are local fixtures only:
`.gitignore` excludes `data/sample/*.pdf` and ZIP files. To restore them in a
fresh checkout, copy the `Sample/document_*.pdf` entries from the supplied
Sample ZIP into `data/sample/`. Do not add those PDFs to Git.

## GroundTruth JSON schema

`app.datasets.groundtruth.GroundTruthRecord` loads the JSON records. The
machine-readable schema is `data/groundtruth.schema.json`. Every record contains
these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `document_id` | string, required | Filename stem such as `document_007`. |
| `title_th` | string or null | Thai project title. |
| `title_en` | string or null | English title when present in the supplied TXT. |
| `students` | array of strings | Student names; IDs remain in `raw_text`. |
| `advisor` | string or null | Advisor named in the front matter. |
| `academic_year` | integer or null | Buddhist Era year printed in the document. |
| `abstract_th` | string or null | Thai abstract text. |
| `abstract_en` | string or null | English abstract only after it is curated. |
| `keywords` | array of strings | Keywords conservatively separated on commas. |

The records also retain optional `raw_text`: the exact decoded text from the
supplied `Sample/GroudTruth/document_*.txt` files. It is provenance, not the
default Wiki prompt input. Missing optional values are `null` or `[]` rather
than guesses. The supplied TXT files contain no English abstracts, so the
current `abstract_en` values are null even when an English abstract page exists
in a PDF. Those fields can be curated later from verified source pages.

Run `python scripts/curate_sample_groundtruth.py` to fill still-missing fields
from the supplied TXT text and rebuild the index and JSON Schema; existing manually
corrected values are preserved. The parser keeps a keyword phrase together when
the TXT has no comma separator. These records are source references for future
evaluation; no final evaluation metrics are implemented yet.

## Wiki input selection

The Task 1.3 extractor still returns all page text and `full_text`. The separate
`app.services.wiki_source.prepare_wiki_source(extraction)` layer selects a small
front-matter input and returns `source_text` plus `selected_pages`. It prioritizes
the first title pages, project metadata, Thai and English abstract pages, and
keyword lines. It checks headings in the first eight pages, uses the page before
an English abstract as a possible Thai abstract when its heading was extracted
poorly, and can use page four as a bounded fallback. At most six source pages
are included by default. The policy and detector are isolated so page detection
can be refined without changing the full-document extractor.

Wiki generation should pass this focused `source_text` to the existing Task 2.2
LLM service. It should not pass `PdfExtractionResult.full_text` or a dataset
record's `raw_text` by default. The GroundTruth records are evaluation references
and are not automatically fed to the generator. The current upload response
still returns full extraction results; PDF-to-Wiki orchestration remains a later
task.
