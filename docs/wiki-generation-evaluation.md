# Task 2.3 local Wiki generation evaluation

Run the batch evaluator from the repository root after restoring the ignored
sample PDFs and starting Ollama with the configured model:

```bash
python scripts/evaluate_wiki_generation.py --output data/evaluation/wiki-report.json
python scripts/evaluate_wiki_generation.py --ids document_007 document_123 --output data/evaluation/small-report.json
```

`data/index.json` is the pairing authority. Each document uses the Task 1.3 PDF
extractor and `prepare_wiki_source`; only `selection.source_text` is passed to
`generate_wiki()`. GroundTruth title, metadata, abstract, and keywords are read
after pairing for evaluation only. The evaluator never edits GroundTruth. Missing
PDFs, missing/invalid GroundTruth, extraction failures, Ollama failures, and
invalid Markdown are recorded per document so later pairs still run.

The terminal summary shows document count, generation successes/failures,
structure pass rate, common categories, and notes for every document. The
optional UTF-8 JSON report adds generated Markdown, selected source pages,
reference coverage details, and a prompt hash for reproducibility. Files under
`data/evaluation/` are ignored by Git. The structure rate uses only outputs
whose structure could be assessed, including outputs rejected by the LLM service
for invalid headings; an outage is not counted as a structure failure.

The content checks compare GroundTruth values to the generated Markdown **only
when the same value is intact in the focused PDF source**. They report exact
title preservation, student/advisor/year and English-title coverage, keyword
coverage, and numeric/technical anchors from GroundTruth finding passages.
Finding excerpts without exact anchors are marked for manual review. The checks
also list generated numbers, named technical terms, and strong-claim words absent
from the focused source, plus likely missing-information marker omissions.

These checks are conservative lexical indicators, not semantic proof of factual
accuracy or completeness. Thai PDF font encodings can damage glyphs even when
the curated GroundTruth text is correct. Such references are reported as source
gaps and are not counted as model omissions. Reviewers should compare each
flagged claim with the selected PDF pages. In particular, generic prose and
paraphrased findings still need a human grounding check. The prompt now asks
the model to use only legible source evidence, keep aims distinct from measured
results, name tools/materials only when explicit, preserve technical terms and
numbers, and use `ไม่พบข้อมูลในเอกสารต้นฉบับ` for unsupported sections.

These three-document local runs are prompt diagnostics, not a benchmark.
Ollama generation uses the configured temperature, so a change in the observed
structure rate alone does not prove the prompt caused the improvement. The
JSON prompt hash and model/timeout fields identify each run; the generated
Markdown and selected-page numbers support later human review.

## Observed local Qwen run and prompt revision

A local run of `document_007`, `document_123`, and `document_254` with
`qwen2.5:7b-instruct` returned two pages and one invalid-structure error. The
first-pass structure rate was 2/3 (67%) among assessed outputs. The returned
pages had almost empty required sections; one appended a period to the exact
missing-information marker, and another changed a Thai title into Chinese.
The third output repeated all seven required headings. The focused PDF text
also had damaged Thai glyphs, so exact GroundTruth titles and some names were
unavailable in the prompt source. One output omitted source-supported metadata,
and another omitted numeric finding anchors. These are separate source and
generation problems; GroundTruth was not rewritten.

The revised prompt explicitly requires a body under every heading, a copied
legible title, the exact marker with no punctuation, and a final check for one
title and seven unique headings. The evaluator now reports empty bodies,
non-source script, ungrounded titles, and marker-format errors alongside the
original reference checks. The first-pass local JSON artifact is
`data/evaluation/task-2.3-real.json` and is ignored by Git.

The first tuned run completed all three generations with valid headings (3/3),
and none had empty section bodies. It still exposed source-grounding problems:
`document_123` used Chinese characters absent from its PDF source, while
`document_254` paraphrased a corrupted Thai title despite an intact English
title on the title page. Two pages omitted source-supported metadata, and
`document_007` and `document_254` still added a period to the exact marker.
That report is `data/evaluation/task-2.3-tuned.json`. The prompt was tightened
again to prefer a verbatim English title when the Thai text layer is damaged,
ban scripts absent from the source, and show the exact marker without a period.

The next three-document run again returned 3/3 pages with valid headings. It
used readable English titles for `document_007` and `document_123` and had no
Chinese-script indicator. `document_254` still rearranged its intact English
title, and `document_123` still appended periods to marker lines. The report is
`data/evaluation/task-2.3-final.json`. Manual source review found two claims
that require caution: `document_007` inferred that an earlier iOS service could
not work, and `document_254` described earlier assessment as inaccurate. Neither
phrase is intact in the focused source. The final prompt revision calls out this
aim-to-failure inference, asks for legible metadata once in the overview, and
illustrates the marker's exact spelling. The lexical checker now flags these
unsupported-problem phrases as review indicators. Even with those checks,
unsupported paraphrases can escape automated detection and require review of
the selected pages.

The final two-document follow-up (`document_007`, `document_254`) returned two
pages with valid heading structure (2/2). Neither output contained the two
previously flagged aim-to-failure phrases, but both still added periods to the
exact missing-information marker. `document_254` also used a Thai title that
does not occur intact in its focused PDF source despite a readable English
title. Both pages omitted some source-supported metadata. The local report is
`data/evaluation/task-2.3-revised.json`. This shows why the structure pass rate
must be read alongside the content flags and manual source review; the prompt
does not guarantee that this local Qwen model follows every grounding rule.

## Focused title and metadata refinement

The prompt builder now copies a conservative title and metadata checklist from
the focused source into `FOCUSED_SOURCE_FACTS`. It prefers a readable Thai title;
when the Thai text layer is damaged and an English title is legible on the title
page, it uses that exact English spelling. Readable English student names,
student IDs, advisor, and academic year are copied when detected. This checklist
comes only from selected PDF pages, never from GroundTruth. The original focused
source is still sent unchanged inside `SOURCE_DOCUMENT`.

The evaluator now tests the generated title against the focused source with
case, spelling, digits, and punctuation preserved. Layout line wraps may be
joined. `# ไม่พบข้อมูลในเอกสารต้นฉบับ` is the sole allowed title fallback when
no reliable title exists. Metadata results identify omissions by field and
value, including readable English student names and IDs that GroundTruth's
curated Thai names cannot match in a damaged text layer. Marker correctness is
reported separately: any marker line with a period, colon, same-line explanation,
or extra section text is invalid even when heading structure passes.

The reusable LLM service now finalizes only fields that can be copied without
interpretation. It selects an exact source title (or the title marker), rebuilds
the overview metadata list from readable selected-page facts, and removes
trivial punctuation from standalone marker lines. Metadata bullets that the
model added without those source facts are removed; extra marker explanations
cause a service error rather than silent deletion. The evaluator saves the raw
Ollama Markdown and its checks alongside the returned final Markdown and
refinement changes. Final checks describe the actual service output, while raw
checks show what the model got wrong. This cannot prove that every prose claim
is supported; reviewers must inspect the selected pages for that.

The final targeted check uses only `document_007` and `document_254` at Ollama
temperature 0. Both generations completed, both passed the required heading
structure, and both returned titles matched the exact readable English title in
their focused sources. All metadata legible to the source-evidence checker was
present in the finalized pages (4/4 items for `document_007`; 8/8 for
`document_254`). Every missing-information marker was an exact standalone line.
The raw Qwen replies still had marker punctuation in both documents and omitted
one readable student name in `document_007`; source-backed finalization repaired
those fields. `document_007` also had a redundant marker alongside source-backed
Swift/iOS tools, which was removed. The JSON report is
`data/evaluation/task-2.3-final-check.json` and remains ignored by Git.

GroundTruth Thai titles, Thai student names, and advisor names are damaged or
absent in the focused PDF text layer, so the exact English titles and readable
English names are used without reconstruction. The evaluator's lexical checks
do not prove the overview prose is grounded; claims about application features
and assessment methods still need comparison with the selected PDF pages.
