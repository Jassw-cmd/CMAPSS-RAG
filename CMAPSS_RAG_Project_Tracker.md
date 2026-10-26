# CMAPSS Technical Q&A RAG — Project Tracker

**Last updated:** 26 October 2026  
**Estimated implementation readiness:** 95% for the tested MVP paths  
**Validation maturity:** preliminary; human review of evidence labels and citations remains incomplete.

> The readiness percentage is a practical estimate, not code coverage or proof of production readiness. Results below are copied from user-run console output and files shared during development; they have not been executed against the user's Windows repository from this environment.

## 1. Completed implementation

- [x] Load technical-document chunks from `data/processed/chunks.json`.
- [x] Generate/use `all-MiniLM-L6-v2` chunk embeddings (384 dimensions in current setup).
- [x] Dense retrieval + BM25 + weighted RRF.
- [x] Optional CrossEncoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [x] Sentence-level evidence splitting and contextual ranking.
- [x] Local Ollama answer synthesis using `qwen2.5:3b`.
- [x] Citation-label normalization and structural validation.
- [x] Abstention behavior tested with a turbine-blade-material question lacking direct support in the retrieved documents.
- [x] Structured CMAPSS lookups for engine, RUL, and sensor values.
- [x] Hybrid document + structured routing.
- [x] Router logic tests passed: route classification, parameter extraction, and hybrid query extraction.
- [x] End-to-end smoke test passed: **6 passed, 0 failed**.

## 2. Verified smoke-test reference outcomes

| Test | Observed behavior |
|---|---|
| RUL definition | Answer synthesized with source labels |
| Wear in challenge problem | Wear equation and normalized health-index equation explained separately |
| Turbine blade material | Abstains when retrieved evidence does not support a material answer |
| Engine 24 RUL at cycle 100 | `47` cycles |
| Sensor 7 of Engine 24 at cycle 100 | `553.51` |
| Hybrid damage propagation + Engine 24 RUL | Technical document answer plus `47` cycles |

These are representative cases, not a complete correctness guarantee.

## 3. Retrieval design decision

**Keep Baseline Hybrid Retrieval + CrossEncoder as default.** Keep Multi-query as an experimental alternative.

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

Multi-query improves top-3 evidence phrase coverage but has lower MRR and lower evidence phrase recall at top 5. For `competition_simulation_method`, recall@5 falls from `0.6667` to `0.3333`, despite both methods returning `VI. COMPETITION DATA` as top section. This supports keeping the baseline until broader testing demonstrates consistent benefits.

## 4. Evaluation outputs

### `evaluate_retrieval.py` (three-query diagnostic)

The current script calls `retrieve(..., top_k=3)` without a CrossEncoder. It evaluates Dense + BM25 + RRF ranking before CrossEncoder reranking.

| Metric | Value |
|---|---:|
| Hit@3 | 1.000 |
| Precision@3 | 0.333 |
| MRR@3 | 0.833 |

Interpret carefully: this represents only three test queries.

### `evaluate_evidence_labels.py`

Source report: `src/evidence_review_report_labeled.json`  
Outputs: `src/evidence_metrics_report.json` and `src/evidence_metrics_per_query.csv`.

The input contains 12 queries: 10 labeled answerable and 2 labeled unanswerable. The labels are provisional assistant-generated labels and have not been established as human-verified ground truth.

| Metric on labeled-answerable queries | K=1 | K=3 | K=5 |
|---|---:|---:|---:|
| Evidence Hit@K | 0.900 | 1.000 | 1.000 |
| Evidence MRR@K | 0.900 | 0.950 | 0.950 |
| Evidence Precision@K | 0.900 | 0.733 | 0.660 |
| Unsupported retrieval rate | 0.100 | 0.267 | 0.340 |
| Section Hit@K | 0.900 | 1.000 | 1.000 |

For the two unanswerable queries, no retrieved chunk in the top 1, 3, or 5 was labeled evidence-relevant. This is a label-consistency check, not a measured model abstention-accuracy score.

### `evaluate_rag.py` baseline-versus-Multi-query comparison

- Hit@3 and Hit@5: both methods 1.000.
- Baseline MRR@3/@5: 1.000; Multi-query: 0.958.
- Multi-query EvidencePhraseRecall@3 improves from 0.819 to 0.944.
- Baseline EvidencePhraseRecall@5 is 0.972 versus 0.944 for Multi-query.
- The turbine-blade-material and airline questions remain flagged for manual review because retrieval ranking alone cannot determine answerability.

EvidencePhraseRecall is a lexical proxy, not a full semantic-support or answer-correctness measure.

## 5. Remaining work

### Priority 1 — evidence and citation quality

- [ ] Manually audit evidence labels against source text for a representative set of questions.
- [ ] Check each generated factual claim against the cited passage; structural label validity alone is insufficient.
- [ ] Evaluate abstention on a larger set of answerable and unanswerable questions.
- [ ] Review the two unanswerable questions (turbine blade material; airline) against the actual source document.

### Priority 2 — repository packaging

- [ ] Verify `requirements.txt` and record actual Python/package versions.
- [ ] Confirm the app and evaluation scripts run from a fresh PowerShell session.
- [ ] Add the current routes, setup steps, metrics, and known limits to `README.md`.
- [ ] Archive legacy experimental scripts that give no useful result, only after confirming they are not imported.
- [ ] Keep the offline data-preparation scripts (`load_markdown.py`, `chunk_markdown.py`, `create_embeddings.py`) for reproducibility.

## 6. Known limitations

1. Citation validation currently checks label structure/existence, not semantic entailment.
2. The CrossEncoder abstention threshold is heuristic and based on a small current test set.
3. Routing is keyword-based and may misclassify unusual wording.
4. Some Markdown-converted passages contain HTML/OCR formatting artifacts.
5. The smoke test has six representative cases; passing it does not imply broad coverage.
6. The 12-query evidence-label dataset includes provisional assistant-generated labels and requires human review before reporting its metrics as benchmark results.
7. RRF scores are rank-fusion scores, not calibrated probabilities of answerability.

## Decision summary

- **Default retrieval:** Baseline Hybrid Retrieval + CrossEncoder.
- **Experimental retrieval:** Multi-query, retained for future comparison.
- **LLM:** Ollama `qwen2.5:3b`, currently configured with `temperature=0.0` and `seed=42` for more repeatable outputs.
- **Smoke test:** 6/6 passed in the latest reported run.
- **MVP state:** core tested paths work; citation grounding, broader human-reviewed evaluation, and repository cleanup remain.
