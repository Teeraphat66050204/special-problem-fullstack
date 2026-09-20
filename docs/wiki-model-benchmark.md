# Wiki model benchmark

The benchmark runs every pair listed in `data/index.json` through the existing
PDF extractor, focused source selector, evidence extractor, Wiki prompt builder,
source-backed output finalizer, and strict validator. Each PDF is prepared once,
then every selected model receives the identical focused source, evidence-derived
prompt, temperature, and validation path. Only the Ollama model name changes.

Qwen3 thinking is disabled with Ollama's native JSON request field
`"think": false`; the prompt is not modified with `/no_think`.

## Commands

Start Ollama and install both initial models:

```bash
ollama pull qwen3:8b
ollama pull qwen3:14b
```

Run all 20 indexed PDF/Ground Truth pairs from the repository root:

```bash
python scripts/benchmark_wiki_models.py --models qwen3:8b qwen3:14b
```

Run one indexed document as a smoke test:

```bash
python scripts/benchmark_wiki_models.py --models qwen3:8b qwen3:14b --limit 1
```

Run an explicit subset in the supplied order:

```bash
python scripts/benchmark_wiki_models.py --models qwen3:8b qwen3:14b --documents document_064 document_114 document_127 document_213 document_251 document_253
```

`--documents` and `--limit` are mutually exclusive. Unknown or duplicate document
IDs are rejected before PDF preparation begins.

Run progress is printed and flushed immediately for PowerShell and other
terminals. The default heartbeat interval is 10 seconds. To print a heartbeat
every 30 seconds instead:

```bash
python scripts/benchmark_wiki_models.py --models qwen3:8b qwen3:14b --progress-interval 30
```

Each completed run reports generation and total time plus tokens/second when
Ollama supplies token metrics. Benchmark-wide progress reports elapsed time,
average completed-run duration, ETA, and estimated finish time. ETA remains
unknown until at least two runs have completed.

Use non-default locations when needed:

```bash
python scripts/benchmark_wiki_models.py --models qwen3:8b --data-dir data --output-dir benchmark
```

The script uses `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT_SECONDS`, and
`OLLAMA_TEMPERATURE` from `.env` or the environment. The temperature and all
other request settings are held constant across models.

## Outputs and metrics

Generated files are local and ignored by Git:

```text
benchmark/
  extraction/document_xxx.json
  qwen3-8b/document_xxx.json
  qwen3-14b/document_xxx.json
  benchmark_results.csv
  benchmark_human_evaluation.csv
  benchmark_summary.json
```

Metrics are separated into two layers. `extraction/document_xxx.json` compares
the shared deterministic evidence with Ground Truth once per document under the
`extraction_pipeline` scope. Metadata accuracy is never copied into a model
summary. Metadata uses Unicode NFC, trimmed/collapsed whitespace, and
case-insensitive comparison for English-oriented fields only. No fuzzy matcher
or LLM judge is used. Because the current Ground Truth schema has no separate
`student_ids` field, expected IDs are conservatively read as standalone
eight-digit values from its curated `raw_text`.

Each model artifact preserves `raw_model_output` and `finalized_markdown`
separately. Their metrics use the explicit `raw_model_output` and
`finalized_production_output` scopes and include strict structure validity,
canonical-section counts, missing-marker section counts, possible unsupported
claim flags, and output size. Unsupported-claim flags are lexical review aids,
not proof: they cover source-absent numbers/technical terms/strong claims and
likely unsupported sections that omit the exact missing-information marker.

The CSV contains one extraction row per document plus separate raw and finalized
rows per model/document, distinguished by `record_type` and `metric_scope`.
The summary reports extraction accuracy once, then model-specific generation
quality, failure, and performance/resource trade-offs without mixing the layers.
`prompt_eval_duration` and `eval_duration` preserve Ollama's native nanoseconds;
wall-clock generation and total pipeline durations are recorded in seconds.

## Rubric-aware reporting

`benchmark_summary.json` includes `rubric_ready_metrics` for each model. It
reports automatic scores for raw instruction following, finalized structure,
robustness, runtime reliability, and latency. Grounding is reported as an
`unsupported_claim_review_pass_rate`: this is a deterministic screening
heuristic for prioritizing human review, not proof of factual correctness.
The latency score uses researcher-defined operational thresholds documented in
`rubric_metadata`; they are not thresholds taken directly from prior literature.

The benchmark samples system/process CPU and memory with `psutil` during model
generation. GPU measurements are optional: `nvidia-smi`, AMD SMI, and ROCm SMI
are detected when available. Missing GPU support remains null and never fails a
run. Per-run telemetry distinguishes `cpu_ram_available_gpu_available`,
`cpu_ram_available_gpu_unavailable`, and `unavailable`. Rubric-facing resource
reporting is `measured`, `partial`, or `unavailable`; no resource-efficiency
rubric score is fabricated.

Criteria 4-6 remain human-only. Each successful output gets a row in
`benchmark_human_evaluation.csv`; enter integer scores from 1 through 4 and
optional notes. Existing rows and ratings are retained by later benchmark runs.
Validate and aggregate the worksheet with:

```bash
python scripts/summarize_human_evaluation.py benchmark/benchmark_human_evaluation.csv
```

Blank ratings remain missing rather than becoming zero. Invalid scores cause a
clear validation error. The next benchmark run incorporates valid worksheet
means and rating counts into `benchmark_summary.json`.
