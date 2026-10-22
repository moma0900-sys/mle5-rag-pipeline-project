# PROJECT_SOLUTION — Northstar RAG Pipeline

Documentation date: October 22, 2026.

This document describes the custom implementation in the repository root. The `solution/` directory contains a separate reference implementation and is not part of the application workflow described here.

## 1. Purpose and current status

The application searches the fictional Northstar knowledge base and returns relevant text passages with citations. It abstains when the retrieved evidence is insufficient. It runs locally, using an embedding model on the CPU and Qdrant in memory.

Answer generation is deterministic and extractive: existing passages are copied and cited. There is no generative LLM integration, no provider API calls, and no interactive chat interface. The embedding model produces search vectors, not written answers.

Verified status:

- 10 Markdown documents, 22 chunks, vector shape `(22, 384)`.
- 6 calibration cases used to select the abstention rules.
- 16 final evaluation cases with saved individual results and metrics.
- 7 passing automated tests.
- Results: [evaluation_results.json](reports/evaluation_results.json).
- Assessment and limitations: [evaluation_report.md](reports/evaluation_report.md).

## 2. Setup and usage

### Prerequisites

Python 3.13 and `uv` are required. The first embedding-model download also requires an internet connection to Hugging Face. The project configuration allows Python 3.13 or later; `.python-version` selects 3.13. A local model cache enables subsequent offline execution.

This implementation does not require Docker, a Qdrant server, a provider account, a GPU, or API keys. The custom application does not load `.env`. Provider dependencies included in the template are not used in the current workflow.

### Install dependencies

Open a terminal in the repository root: the directory containing `main.py`, `pyproject.toml`, and `uv.lock`. Then run:

```powershell
uv sync --locked --group dev
```

This creates the local environment, including pytest, using the lockfile. Source files are accessed through relative paths, so starting the application from another working directory may fail.

### Run the application and evaluation

```powershell
uv run main.py
```

This command runs the complete workflow:

1. Read and validate the Markdown files.
2. Create chunks with source metadata.
3. Load the model and calculate chunk embeddings.
4. Recreate and populate the `northstar_support` collection in memory.
5. Embed and search all questions from `data/evaluation/evaluation_questions.json`.
6. Apply the safety filter and score thresholds.
7. Return cited evidence passages or an abstention message.
8. Calculate metrics and write `reports/evaluation_results.json`.

The `reports/` directory is created if necessary. Each successful run replaces the previous JSON results file. The Markdown evaluation report is not regenerated automatically.

### Rebuild the index

No separate rebuild command is needed:

```powershell
uv run main.py
```

Every run creates a new index from the current source documents. Qdrant uses `QdrantClient(":memory:")`; the index is lost when the client closes. This avoids a separate service and stale index files, but requires repeated embedding computation. Only the model cache and written results files remain on disk.

### Run tests

```powershell
uv run pytest tests -q
```

Current expected result: `7 passed`. The explicit `tests` path restricts execution to the custom tests; the separate tests under `solution/` are not part of this implementation.

The tests use synthetic texts or vectors. They do not require a model download and do not measure the semantic quality of the embeddings.

### Optional: run offline after downloading the model

In PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
uv run main.py
```

The model and its configured revision must already be cached. Loading fails in offline mode if the required files are missing. These variables apply to the current session; remove them with `Remove-Item Env:HF_HUB_OFFLINE` and `Remove-Item Env:TRANSFORMERS_OFFLINE`.

## 3. Files and responsibilities

| File or directory | Responsibility |
|---|---|
| `helpers.py` | Pydantic models: `Document`, `Chunk`, and `ChunkSize`. |
| `read_md_files.py` | Load Markdown files, prepare paragraphs, and create chunks. |
| `embeddings.py` | Load the pinned model and embed chunks and questions. |
| `vector_store.py` | Populate Qdrant and select evidence using score thresholds. |
| `railguards.py` | Detect known suspicious instructions, exclude chunks, and log warnings. |
| `generate_answer.py` | Cite evidence text and reject missing citations or citations to sources outside the selected evidence. |
| `main.py` | Orchestrate the workflow, load questions, calculate metrics, and write JSON. |
| `data/knowledge_base/` | The 10 Markdown source documents. |
| `data/evaluation/calibration.json` | 6 cases for selecting thresholds. |
| `data/evaluation/evaluation_questions.json` | 16 final cases; originally named `questions.json`. |
| `tests/test_guardrails.py` | Five cases covering the safety filter, abstention, output, and citation validation. |
| `tests/test_chunking_and_search.py` | Two cases covering chunk metadata and Qdrant vector-to-payload mapping. |
| `reports/evaluation_results.json` | Automatically written answers, hits, evidence, metrics, and scoring rules. |
| `reports/evaluation_report.md` | Written analysis of the saved evaluation run. |

## 4. Implementation details

### 4.1 Documents and chunking

`available_mds()` loads files directly inside the knowledge-base directory in sorted order, accepting the `.md` extension regardless of case. It does not search subdirectories recursively. Files are read as UTF-8. Each file must have a first line starting with `# `, a nonempty title, and a nonempty body.

`Document` contains `doc_id`, `title`, `text`, and `path`. The document ID is the filename without its extension.

`chunk_documents()` splits text into paragraphs at blank lines. Line breaks within a paragraph become spaces. Short paragraphs remain separate; they are not combined into larger chunks. Only overlong paragraphs are split further by `chunk_size()`.

The settings in `main()` are:

- At most 100 whitespace-separated words per window.
- An overlap of 15 words within a split paragraph.
- A step size of `100 - 15 = 85`.
- No overlap between different paragraphs or documents.

These values are intended to keep compact paragraphs intact and preserve context at window boundaries in long paragraphs. They are a justified starting point, not a demonstrated optimum. The longest chunk in the current corpus contains only 43 words. Window overlap is therefore unnecessary for the actual corpus; its behavior is tested using a longer synthetic paragraph.

Each `Chunk` retains `chunk_id`, `doc_id`, `title`, `text`, and a zero-based `position`. Example: `account-mfa-0`. The position restarts for each document. The file path remains in the document object but is not copied into the chunk payload.

The code currently requires a positive overlap smaller than the chunk size. An overlap of 0 is not allowed.

### 4.2 Embeddings

The model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision:

```text
1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

`load_embedding_model()` loads it onto the CPU. Model files are cached locally; the loaded instance resides in memory during execution.

Chunks are embedded as `title + "\n" + text`. The title provides additional context for the paragraph. Questions are embedded with the same model. Outputs are normalized and processed as NumPy `float32` arrays.

Each chunk produces one vector containing 384 numbers. For 22 chunks, the shape is `(22, 384)`; there are not 384 vectors per chunk. The word limit used for chunking is also distinct from the model's token limit.

### 4.3 Storage and search

`store_chunks()` validates the number and shape of the vectors and checks that all values are finite. Chunks and vectors are paired by position using `zip(..., strict=True)`. Each Qdrant point receives:

- a sequential numeric point ID,
- its vector as a Python list,
- the complete chunk as a JSON-compatible payload.

The collection is named `northstar_support` and uses 384 dimensions with cosine similarity. The dimension is taken from `vectors.shape[1]`.

For each question, `query_points()` returns at most three ranked chunk hits, including payloads but excluding returned vectors. Three hits may come from fewer than three distinct documents. The model calculates the vectors; Qdrant calculates their similarity during search.

### 4.4 Safety filter and abstention

The sequence is:

```text
Hits → filter_unsafe_hits() → select_evidence() → generate_answer()
```

The safety filter normalizes case and whitespace in the title and text. It checks a small list of known phrases requesting that prior instructions be ignored or that the system prompt or API key be revealed. A matching chunk is excluded completely. A warning identifies the chunk ID and document ID; by default, it is printed to the console rather than saved as a persistent audit file.

The rules do not depend on the source ID. An attack quoted as a safety example is also excluded. Other unflagged chunks from the same source can still be used.

Next, `select_evidence()` applies these rules:

1. No hits: abstain.
2. Best remaining score below 0.45: abstain.
3. Otherwise, retain passages with a positive score of at least 0.40.

Rechecking scores after the safety filter prevents an excluded high-scoring hit from authorizing an answer. These rules do not establish whether a policy actually applies to the question. The phrase filter can miss reworded attacks and exclude harmless quotations.

### 4.5 Answers and citation validation

`generate_answer()` copies selected texts verbatim and appends `[doc_id]` to each passage. Multiple passages are joined with blank lines. Without evidence, it returns `NO_ANSWER`.

`validate_citations()` raises `ValueError` for answers without a citation or with citations to document IDs outside the selected evidence. This validates source-ID membership, not necessarily the factual support for each statement.

### 4.6 Evaluation and JSON structure

`main()` loads the final questions from `evaluation_questions.json`. Expected sources and terms do not influence retrieval or answer generation; they are used only for scoring.

The JSON file contains:

| Field | Contents |
|---|---|
| `question_set` | Name of the question file used. |
| `metrics` | Mean values of the five metrics, each between 0 and 1. |
| `metric_case_counts` | Number of cases actually evaluated for each metric. |
| `metric_definitions` | English descriptions of the scoring rules. |
| `results` | Question, expectations, original hits, selected evidence, answer, abstention, and per-case metrics. |

`null` means “not evaluated” and is excluded from the corresponding mean. Recall and MRR use the original top-3 hits. Relevance is a term-coverage proxy; faithfulness checks exact text-and-citation matches for returned answers. The [evaluation report](reports/evaluation_report.md) defines the denominators and limitations in full.

## 5. Changing settings or calibrating

| Setting | Current value | Location |
|---|---|---|
| Chunk size | 100 words | `chunk_documents()` call in `main.py`. |
| Overlap | 15 words | The same call in `main.py`. |
| Top-k | 3 | `query_points(limit=3)` in `main.py`. |
| Evaluation cutoff | 3 | `retrieved_hits[:3]` and metric names in `main.py`. |
| Minimum best-hit score | 0.45 | Default argument of `select_evidence()` in `vector_store.py`. |
| Minimum individual evidence score | 0.40 | Default argument of `select_evidence()` in `vector_store.py`. |
| Model, revision, and device | MiniLM, pinned revision, CPU | `embeddings.py`. |
| Collection and distance | `northstar_support`, Cosine | `vector_store.py`. |
| Evaluation file | `evaluation_questions.json` | `cases_path` in `main.py`. |

There are currently no command-line arguments for these settings. Changing top-k requires updating the search limit, evaluation cutoff, metric names, and documentation together.

For new experiments, choose development settings using only `calibration.json` or new calibration cases. To do this, temporarily set `cases_path` to `data/evaluation/calibration.json`. Note that the application still overwrites the same JSON output file. Preserve the final results first or record experiments separately.

Settings are fixed before final evaluation. The configuration documented here remains unchanged. Once settings are adjusted using previously seen final questions, that question set is no longer an independent final test.

## 6. Tests, reproducibility, and limitations

The seven tests cover attack removal, score checks after filtering, cited output, missing and fabricated citations, chunk metadata, and Qdrant vector-to-payload mapping. The two citation errors become two test cases through parametrization.

All seven tests were rerun successfully on October 22, 2026. The six calibration cases were also reproduced offline in memory using the cached model. The report values come from the existing final JSON file, which was not regenerated during that check.

`uv sync --locked --group dev` is the documented setup command for a fresh environment. A complete installation into an initially empty environment was not performed during this documentation review.

Additional limitations:

- The small corpus and question sets do not establish production reliability.
- There is no semantic check that a retrieved policy applies to the question.
- The safety filter is a limited heuristic; safety events are not included in the evaluation JSON.
- The JSON file does not currently store a timestamp, source hash, or complete configuration snapshot. This document records the verified configuration.
- `main.py` currently prints answers twice in the console. This does not affect the single saved result per question in the JSON file.
- The custom evaluation is a batch workflow. Individual-query interaction, a persistent index, and a graphical interface are not implemented.

The four submission questions, failure analysis, and three prioritized improvement recommendations are addressed in the [evaluation report](reports/evaluation_report.md).

