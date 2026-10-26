# CMAPSS Technical Q&A — Hybrid RAG

A local technical question-answering assistant built around NASA C-MAPSS simulated turbofan-engine data and a technical document about damage propagation and prognostics.

The system combines three query routes:

- **Document:** retrieves passages from the technical Markdown document and uses a local Ollama model to synthesize an answer with source labels.
- **Structured:** answers engine, cycle, RUL, and sensor questions directly from the CMAPSS dataframe.
- **Hybrid:** combines a technical explanation from documents with a structured CMAPSS result.

> **Current checkpoint:** the reported end-to-end smoke test passed **6/6** cases, and the router logic tests passed. These checks cover representative cases; they are not a guarantee of correctness for arbitrary questions.

## 1. Architecture

```text
User question
    |
    v
src/query_router.py
    |------------------------|------------------------|
    v                        v                        v
 Document                  Structured                Hybrid
    |                        |                        |
    v                        v                        +-- structured lookup
src/rag_pipeline.py       src/cmapss_retriever.py     +-- document pipeline
    |                        ^                        |
    v                        |                        v
src/retriever.py          src/load_data.py        combine results
Dense + BM25 + RRF
    |
    v
Optional CrossEncoder reranking
    |
    v
src/answer_generator.py
Split sentences; preserve context; rank evidence
    |
    v
src/answer_synthesizer.py
Ollama qwen2.5:3b; answer; citation-label validation; abstention
    |
    v
Final answer
```

### Runtime modules

| File | Responsibility |
|---|---|
| `src/query_router.py` | Classifies queries as `document`, `structured`, `hybrid`, or `unknown`; parses structured parameters and combines route results. |
| `src/rag_pipeline.py` | Connects document retrieval, sentence-evidence ranking, and answer synthesis. |
| `src/retriever.py` | Provides Dense retrieval, BM25, weighted Reciprocal Rank Fusion (RRF), optional CrossEncoder reranking, and `load_index()`. |
| `src/answer_generator.py` | Splits retrieved chunks into candidate sentences, preserves nearby context and section metadata, and ranks evidence. Despite its filename, this module primarily prepares/ranks evidence rather than generating the final prose answer. |
| `src/answer_synthesizer.py` | Calls the local Ollama chat API, asks the model to answer from evidence, normalizes citation formats, validates citation labels, and abstains when configured checks fail. |
| `src/load_data.py` | Loads the structured CMAPSS data used by the application. |
| `src/cmapss_retriever.py` | Implements engine data, RUL, and sensor-value lookups. |

### Offline data preparation

| File | Responsibility |
|---|---|
| `src/load_markdown.py` | Loads the technical Markdown source. |
| `src/chunk_markdown.py` | Creates chunks and associated metadata. |
| `src/create_embeddings.py` | Generates and saves embeddings for the processed chunks. |

The processed document index is expected at:

```text
data/processed/chunks.json
data/processed/embeddings.npy
```

The chunks and embedding matrix must correspond one-to-one. The configured embedding model is `all-MiniLM-L6-v2` (384-dimensional embeddings in the current setup). Rebuild the index and embeddings together if the source text, chunking strategy, or embedding model changes.

## 2. Requirements

- Windows and PowerShell (examples below use the project virtual environment).
- A working Python virtual environment with the project dependencies, including `numpy`, `pandas`, `sentence-transformers`, `rank-bm25`, and `requests`.
- Ollama installed and running locally.
- Ollama model `qwen2.5:3b` downloaded.
- The CMAPSS input data and the processed chunk/embedding files.

Check the local Ollama model:

```powershell
ollama list
```

The model used in the current setup is `qwen2.5:3b`. Set it in the same PowerShell session used to run Python:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
```

The default Ollama endpoint is `http://localhost:11434` unless `OLLAMA_BASE_URL` is set. This local setup does **not** require an OpenAI API key.

If Ollama is not responding, test it directly:

```powershell
ollama run qwen2.5:3b
```

## 3. Run from the project root

Open PowerShell at the project root, for example:

```text
D:\New folder\UROP\test\chatgpt rag cmapss
```

Set the Ollama model:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
```

### Check syntax

```powershell
.\.venv\Scripts\python.exe -m py_compile `
    src\retriever.py `
    src\answer_generator.py `
    src\answer_synthesizer.py `
    src\rag_pipeline.py `
    src\query_router.py
```

### Run router logic tests

```powershell
.\.venv\Scripts\python.exe src\test_router_logic.py
```

The current run reported all route, parameter-extraction, and hybrid-query-extraction tests passing.

### Run end-to-end smoke tests

```powershell
.\.venv\Scripts\python.exe src\smoke_test.py
```

Latest reported summary:

```text
SMOKE TEST SUMMARY: 6 passed, 0 failed
```

### Run the application/router

```powershell
.\.venv\Scripts\python.exe src\query_router.py
```

### Run retrieval evaluation

```powershell
.\.venv\Scripts\python.exe src\evaluate_retrieval.py
```

The current `evaluate_retrieval.py` code uses `retrieve(..., top_k=3)` without passing a CrossEncoder, so this result evaluates the Dense + BM25 + RRF ranking before CrossEncoder reranking. The current labeled evaluation uses three query cases.

### Run evidence-label evaluation

The labeled input file is stored in `src/` in the current folder layout. Pass the path explicitly so the script does not depend on the terminal's working directory:

```powershell
.\.venv\Scripts\python.exe src\evaluate_evidence_labels.py src\evidence_review_report_labeled.json
```

The script saves these outputs beside the input file:

```text
src/evidence_metrics_report.json
src/evidence_metrics_per_query.csv
```

### Run RAG baseline versus Multi-query evaluation

```powershell
.\.venv\Scripts\python.exe src\evaluate_rag.py
```

This evaluation writes `data/processed/rag_evaluation_report.json` in the current setup. Check the script's console output for the exact destination if project paths have changed.

## 4. Query routing

### Document route

Examples:

- `What is remaining useful life (RUL)?`
- `What is wear in the challenge problem?`
- `How is damage propagation modeled?`
- `What material is the engine turbine blade made of?`

The document route retrieves chunks, ranks evidence sentences, and asks Ollama to synthesize an answer. If the retrieved evidence is not judged sufficient by the current system checks, it abstains rather than guessing.

### Structured route

Examples:

- `What is the RUL of Engine 24 at cycle 100?`
- `What is the value of Sensor 7 of Engine 24 at cycle 100?`

The answer is read from the CMAPSS dataframe instead of being recalculated by the LLM.

Observed reference values for the current dataset:

| Query | Observed result |
|---|---:|
| Engine 24 RUL at cycle 100 | 47 cycles |
| Sensor 7 of Engine 24 at cycle 100 | 553.51 |

### Hybrid route

Example:

`How is damage propagation modeled, and what is the RUL of Engine 24 at cycle 100?`

The router retrieves the RUL from structured data and separately answers the damage-propagation part using document RAG, then combines the two outputs.

## 5. Evaluation results and interpretation

### A. Baseline versus Multi-query

The user-run comparison reported the following metrics on its current test set:

| Metric | Baseline | Multi-query | Delta |
|---|---:|---:|---:|
| Hit@3 | 1.000 | 1.000 | +0.000 |
| Hit@5 | 1.000 | 1.000 | +0.000 |
| MRR@3 | 1.000 | 0.958 | -0.042 |
| MRR@5 | 1.000 | 0.958 | -0.042 |
| Precision@3 | 0.472 | 0.500 | +0.028 |
| Precision@5 | 0.367 | 0.367 | +0.000 |
| EvidencePhraseRecall@3 | 0.819 | 0.944 | +0.125 |
| EvidencePhraseRecall@5 | 0.972 | 0.944 | -0.028 |

**Decision:** keep Baseline Hybrid Retrieval + CrossEncoder as the default. Keep Multi-query as an experimental option until its improvements are consistent across a larger, carefully reviewed test set.

A notable per-query regression was recorded for `competition_simulation_method`: EvidencePhraseRecall@5 changed from `0.6667` with Baseline to `0.3333` with Multi-query, although both had `VI. COMPETITION DATA` as their top section. This illustrates why matching a section alone does not guarantee that the retrieved chunk contains the required evidence.

### B. Evidence-label evaluation

The current `evidence_metrics_report.json` reports 12 queries in total: 10 labeled answerable and 2 labeled unanswerable. **The labels are provisional assistant-generated labels, not human-verified ground truth.** Treat these figures as preliminary diagnostics rather than final benchmark results.

Metrics for the 10 labeled-answerable queries:

| Metric | K=1 | K=3 | K=5 |
|---|---:|---:|---:|
| Evidence Hit@K | 0.900 | 1.000 | 1.000 |
| Evidence MRR@K | 0.900 | 0.950 | 0.950 |
| Evidence Precision@K | 0.900 | 0.733 | 0.660 |
| Unsupported retrieval rate | 0.100 | 0.267 | 0.340 |
| Section Hit@K | 0.900 | 1.000 | 1.000 |

The unanswerable-query checks found no retrieved chunks labeled evidence-relevant in the top 1, 3, or 5 for the two currently labeled-unanswerable questions. This is a **label-consistency check only**; it is not a measured model abstention accuracy score.

### C. Small retrieval diagnostic

The current three-query `evaluate_retrieval.py` run reported:

| Metric | Value |
|---|---:|
| Hit@3 | 1.000 |
| Precision@3 | 0.333 |
| MRR@3 | 0.833 |

This diagnostic uses only three queries and, with the current script, does not apply CrossEncoder reranking. Do not compare it directly with the larger evidence-label evaluation without noting the differences in test cases, labels, and ranking configuration.

## 6. Known limitations

1. **Citation validation is structural, not semantic.** The validator checks that citation labels are present and within the supplied evidence range. It does not prove that each cited source supports the neighboring claim.
2. **The abstention threshold is heuristic.** A CrossEncoder score threshold from a small set of cases can reject answerable questions or accept weak evidence. Calibrate it with a larger manually reviewed dataset before relying on it.
3. **Evaluation labels are provisional.** The current evidence labels must be checked by a human against the actual source text before being treated as ground truth.
4. **Routing is keyword-based.** Unusual phrasing or complex multi-intent questions may be routed incorrectly.
5. **Document extraction has formatting artifacts.** Some chunks contain Markdown, HTML fragments, or OCR-like text, which can affect ranking and answer quality.
6. **The smoke test is small.** Six passing cases establish basic expected flows, not research-grade correctness or broad coverage.
7. **The local model has limited capacity.** `qwen2.5:3b` may generate incorrect details or citations. Review technical claims and equations, especially when the answer uses multiple sources.
8. **RRF is not a confidence probability.** The retrieval `score` field represents a rank-fusion score, not a calibrated measure of answerability. Print `dense_score`, `bm25_score`, and `reranker_score` separately when diagnosing ranking.

## 7. Troubleshooting

### `OLLAMA_MODEL is not configured`

In the same PowerShell session used to run Python:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
```

Alternatively, use `qwen2.5:3b` as the fallback in `call_llm()` in `src/answer_synthesizer.py`.

### Cannot connect to Ollama

Check that Ollama is running and that the configured endpoint is `http://localhost:11434`. Test the model directly with `ollama run qwen2.5:3b`.

### `ImportError: cannot import name 'load_index' from 'retriever'`

Check that `src/retriever.py` defines `load_index()` and `MODEL_NAME`, and that the router imports the intended module from the project's `src/` folder.

### Chunk/embedding count mismatch

The number of items in `data/processed/chunks.json` must equal the number of vectors in `data/processed/embeddings.npy`. Rebuild the index and embeddings together if they are out of sync.

## 8. Final checklist before sharing

- [x] Document, structured, and hybrid routes demonstrated in the smoke test.
- [x] Router logic tests passed.
- [x] Ollama answer generation exercised with `qwen2.5:3b`.
- [x] At least one unsupported question tested for abstention.
- [x] Baseline/Multi-query comparison recorded; Baseline retained as default.
- [ ] Manually review a representative sample of evidence labels and citations.
- [ ] Expand answerable and unanswerable test questions.
- [ ] Confirm all scripts and paths from a clean PowerShell session.
- [ ] Keep experimental scripts only if still useful; archive older no-result experiments rather than deleting them immediately.
- [ ] Record exact Python/package versions in `requirements.txt`.

---

**Project status:** the core MVP flow is implemented and has passed the current representative smoke test. Retrieval and evidence-label metrics provide preliminary diagnostics, while semantic citation validation and broader manually reviewed evaluation remain open work.
