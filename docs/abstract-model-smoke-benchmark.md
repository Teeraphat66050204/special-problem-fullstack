# Project-summary model smoke benchmark

The `production_project_summary_v1` generation contract is frozen for the full
benchmark: prompt, source priority, OCR/chapter detection, model settings,
validation thresholds, and retry behavior must remain unchanged during the run.

The filename is retained for compatibility with the previously prepared abstract
smoke harness. New runs use the production project metadata and summary contract
and remain limited to **one dataset document by three fixed models**:

1. `qwen3:8b`
2. `hf.co/mradermacher/typhoon2-qwen2.5-7b-instruct-GGUF:Q4_K_M`
3. `hf.co/openthaigpt/openthaigpt1.5-7b-instruct-GGUF:Q4_K_M`

The harness extracts the chosen PDF once, applies the existing front-matter OCR
policy and targeted Chapter 1 recovery, selects the same summary-only focused
source for every model, and builds `build_project_summary_prompt`. Metadata uses
the separate deterministic evidence source. GroundTruth `abstract_th` is
loaded only for a labelled reference-similarity diagnostic and never enters the
prompt.

Every model runs with temperature `0`. The Ollama `think` field is `false` for
`qwen3:8b` and omitted for both Qwen2.5-derived models. Model output remains plain
text and is validated with `validate_project_summary`; JSON is neither requested
nor required.

Per-model artifacts retain the raw response, deterministic metadata,
`generated_summary`, validation issues, Ollama error diagnostics, ordered unique
summary page provenance, OCR and source diagnostics, token counters, speed, and
elapsed time. When Ollama returns them, artifacts also retain `done`, `done_reason`,
`total_duration` and `load_duration`; absent fields remain `null`. The raw Ollama
token-ID context is not written. Artifacts retain only `context_present` and
`context_length` to avoid multi-thousand-line result files. Token counts are
recorded without inferring a context-limit failure. A response whose explicit
`done_reason` is `length` is recorded as `project_summary_truncated`, retains its
raw output and performance diagnostics, and is never retried.
The normalized sequence-match ratio against the historical Thai abstract is only
a reference heuristic; it is not a factuality or writing-quality score.

Generation remains non-streaming (`stream: false`). Streaming is intentionally
not enabled by this harness.

An output directory is required and must not already exist. This protects previous
Wiki benchmark reports and abstract smoke runs. The harness uses
`OLLAMA_TIMEOUT_SECONDS`; the default is 120 seconds, while `0` or a negative value
waits indefinitely. It also uses `OLLAMA_NUM_PREDICT`, defaulting to 800 generated
tokens per project summary.

## Commands

Pull the fixed models once:

```powershell
ollama pull qwen3:8b
ollama pull hf.co/mradermacher/typhoon2-qwen2.5-7b-instruct-GGUF:Q4_K_M
ollama pull hf.co/openthaigpt/openthaigpt1.5-7b-instruct-GGUF:Q4_K_M
```

Run one document into a new directory:

```powershell
python -m scripts.benchmark_abstract_models --document document_071 --output-dir benchmark/project-summary-smoke/document-071-run-001
```

To exercise targeted Chapter 1 OCR, choose an affected document and configure
Typhoon first:

```powershell
python -m scripts.benchmark_abstract_models --document document_064 --output-dir benchmark/project-summary-smoke/document-064-run-001
```

Each command performs exactly three Ollama calls. The implementation and tests do
not invoke either command automatically.

## Resumable full benchmark

The full runner uses the fixed 20-document dataset index and the same three models,
executing all 60 runs sequentially. It atomically checkpoints each model result
before continuing and regenerates `summary.json` from complete per-run artifacts.

Start a new full benchmark in a new output directory:

```powershell
python -m scripts.benchmark_abstract_models --full --output-dir benchmark/project-summary-full/run-001
```

Resume after an interruption. Complete successes and complete failures are
skipped; missing, corrupt, incomplete, or identity-mismatched artifacts are rerun:

```powershell
python -m scripts.benchmark_abstract_models --full --resume --output-dir benchmark/project-summary-full/run-001
```

Rerun completed failures while retaining completed successes:

```powershell
python -m scripts.benchmark_abstract_models --full --resume --rerun-failures --output-dir benchmark/project-summary-full/run-001
```

Resume validation checks schema version, artifact type, pipeline, document ID,
model, status, settings, performance data, and success/failure completion fields.
Timestamps are not used to decide completeness.
