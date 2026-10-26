r"""Compare the baseline hybrid retriever with Multi-query retrieval.

Run from the project root:
    .\.venv\Scripts\python.exe src\evaluate_rag.py

Optional diagnostic output:
    .\.venv\Scripts\python.exe src\evaluate_rag.py --diagnostics

The script intentionally keeps evaluation compact. It runs the same test set
through both retrieval methods, compares ranking and lexical evidence coverage,
and saves the complete results to data/processed/rag_evaluation_report.json.
RRF and CrossEncoder scores are ranking signals, not calibrated probabilities.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from retriever import retrieve, retrieve_multi_query, diagnose_retrieval


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
REPORT_PATH = BASE_DIR / "data" / "processed" / "rag_evaluation_report.json"

MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASELINE_CANDIDATE_K = 25
MULTI_QUERY_CANDIDATE_K = 25
K_VALUES = (3, 5)
TOP_K = max(K_VALUES)


# Query expansions are kept deliberately focused: the original query remains
# the first query, and the additional queries target distinct evidence facets.
SUBQUERIES = {
    "rul_definition": [
        "What does remaining useful life estimate, and in what units is it measured?",
    ],
    "wear_challenge": [
        "How is initial wear represented in the challenge problem?",
        "What degradation behavior is assumed in the challenge model?",
    ],
    "damage_propagation": [
        "How are flow and efficiency changes used to model damage propagation?",
        "What health equation describes damage propagation in the challenge problem?",
    ],
    "cmapss_definition": [
        "What does C-MAPSS simulate?",
        "What is the full name of C-MAPSS?",
    ],
    "initial_wear": [
        "What causes initial wear in the engine model?",
        "How can initial wear be modeled through flow and efficiency variations?",
    ],
    "noise": [
        "What types of noise affect the simulated engine sensor measurements?",
        "How are noise distributions added to the generated data?",
    ],
    "competition_dataset": [
        "How were trajectories generated using C-MAPSS?",
        "What operating conditions and degradation combinations were used?",
        "How was measurement noise added to the generated sensor time series?",
    ],
    "arrhenius": [
        "Which failure mechanisms are described by the Arrhenius model?",
        "What role does temperature or activation energy play in the model?",
    ],
    "health_index": [
        "How are the engine's operative margins combined into a health index?",
        "What health-index value defines the failure criterion?",
    ],
    "system_model": [
        "How does the system model connect health-parameter changes to sensor outputs?",
        "What engine system is simulated by C-MAPSS?",
    ],
    "competition_operating_conditions": [
        "Which combinations of altitude, TRA, and Mach number were simulated?",
        "How was HPC degradation varied across the challenge trajectories?",
    ],
    "competition_simulation_method": [
        "What operating conditions and degradation combinations were used to generate the training, test, and validation trajectories?",
        "How was degradation in the high-pressure compressor (HPC) simulated using flow and efficiency modifiers?",
        "How was measurement noise added to the C-MAPSS sensor output time series?",
    ],
}


# ---------------------------------------------------------------------------
# Test set: retain existing ground-truth labels and evidence phrase groups.
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "rul_definition",
        "query": "What is remaining useful life (RUL)?",
        "answerable": True,
        "relevant_sections": ["II. PROGNOSTICS"],
        "evidence_checks": [
            ["remaining useful life (RUL)", "remaining useful component life"],
            ["units of time", "hours or cycles", "hours", "cycles"],
        ],
    },
    {
        "id": "wear_challenge",
        "query": "What is wear in the challenge problem?",
        "answerable": True,
        "relevant_sections": [
            "A. Initial Wear",
            "D. Damage Propagation Model for the Challenge Problem",
        ],
        "evidence_checks": [
            ["initial wear", "wear threshold"],
            ["manufacturing inefficiencies", "manufacturing and assembly variations"],
            ["exponential behavior", "exponential term"],
        ],
    },
    {
        "id": "damage_propagation",
        "query": "How is damage propagation modeled?",
        "answerable": True,
        "relevant_sections": [
            "D. Damage Propagation Model for the Challenge Problem",
            "A. C-MAPSS",
            "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation",
        ],
        "evidence_checks": [
            ["exponential rate of change", "exponential term"],
            ["flow and efficiency", "flow", "efficiency"],
            ["wear threshold", "operational limit"],
        ],
    },
    {
        "id": "cmapss_definition",
        "query": "What is C-MAPSS?",
        "answerable": True,
        "relevant_sections": ["A. C-MAPSS"],
        "evidence_checks": [
            ["Commercial Modular AeroPropulsion System Simulation"],
            ["simulating a realistic large commercial turbofan engine"],
        ],
    },
    {
        "id": "initial_wear",
        "query": "How is initial wear represented in the engine model?",
        "answerable": True,
        "relevant_sections": ["A. Initial Wear"],
        "evidence_checks": [
            ["initial wear"],
            ["initial values", "random distribution"],
            ["flow and efficiencies", "flow and efficiency"],
        ],
    },
    {
        "id": "noise",
        "query": "What role does noise play in the simulated data?",
        "answerable": True,
        "relevant_sections": ["B. Noise"],
        "evidence_checks": [
            ["process noise"],
            ["measurement noise", "sensor noise"],
            ["sensor measurements"],
        ],
    },
    {
        "id": "competition_dataset",
        "query": "How is the competition dataset generated?",
        "answerable": True,
        "relevant_sections": ["VI. COMPETITION DATA"],
        "evidence_checks": [
            ["train, test, and validation data sets"],
            ["trajectories were created from C-MAPSS"],
            ["six different combinations", "6 different combinations"],
        ],
    },
    {
        "id": "arrhenius",
        "query": "What is the Arrhenius model?",
        "answerable": True,
        "relevant_sections": ["A. Arrhenius"],
        "evidence_checks": [
            ["Arrhenius model"],
            ["chemical reactions", "diffusion processes", "migration processes"],
            ["activation energy"],
        ],
    },
    {
        "id": "health_index",
        "query": "How is the health index calculated?",
        "answerable": True,
        "relevant_sections": ["D. Health Index Calculation"],
        "evidence_checks": [
            ["operability margins", "operational margins"],
            ["normalized to the range [0,1]", "range [0,1]"],
            ["minimum of several", "minimum of all operative margins"],
        ],
    },
    {
        "id": "system_model",
        "query": "What does the system model represent?",
        "answerable": True,
        "relevant_sections": ["III. SYSTEM MODEL"],
        "evidence_checks": [
            ["system model"],
            ["input variations of health related parameters"],
            ["resulting output sensor measurements"],
        ],
    },
    {
        "id": "unknown_blade_material",
        "query": "What material is the engine turbine blade made of?",
        "answerable": False,
        "relevant_sections": [],
        "evidence_checks": [],
        "review_note": "The source may not specify the blade material; do not infer an answer from related passages.",
    },
    {
        "id": "unknown_airline",
        "query": "Which airline operated the aircraft used to collect this data?",
        "answerable": False,
        "relevant_sections": [],
        "evidence_checks": [],
        "review_note": "The paper describes simulated data; verify that no retrieved passage directly identifies an airline before answering.",
    },
    {
        "id": "competition_operating_conditions",
        "query": "What operating conditions were used to produce the PHM challenge trajectories?",
        "answerable": True,
        "relevant_sections": ["VI. COMPETITION DATA"],
        "evidence_checks": [
            ["six different combinations", "6 different combinations"],
            ["Altitude, TRA, and Mach number"],
            ["Simulation of degradation in HPC module"],
        ],
    },
    {
        "id": "competition_simulation_method",
        "query": "How did the authors simulate the competition data?",
        "answerable": True,
        "relevant_sections": [
            "VI. COMPETITION DATA",
            "_C. Data Generation_",
            "_A. C-MAPSS_",
            "_B. Noise_",
        ],
        "evidence_checks": [
            ["trajectories were created from C-MAPSS"],
            ["flow and efficiencies of the HPC module"],
            ["measurement noise"],
        ],
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_chunks() -> list:
    """Load and validate the processed chunk list."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty JSON list.")

    return chunks


def get_text(item: Any) -> str:
    """Return chunk text for either a string or dictionary chunk."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return ""


def get_section(item: Any) -> str:
    """Extract the section label from a chunk or retrieval result."""
    if not isinstance(item, dict):
        return ""

    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        section = metadata.get("section") or metadata.get("heading")
        if section:
            return str(section)

    return str(item.get("section") or item.get("heading") or "")


def normalize_section(section: str) -> str:
    """Normalize Markdown decoration for section-level label comparison."""
    return (
        str(section)
        .casefold()
        .replace("_", "")
        .replace("*", "")
        .strip()
    )


def phrase_group_found(texts: list[str], alternatives: list[str]) -> bool:
    """Lexical proxy: check whether any alternative occurs in retrieved text."""
    joined_text = "\n".join(texts).casefold()
    return any(phrase.casefold() in joined_text for phrase in alternatives)


# ---------------------------------------------------------------------------
# Retrieval and metrics
# ---------------------------------------------------------------------------


def retrieve_for_case(
    test_case: dict,
    method: str,
    model,
    chunks: list,
    embeddings,
    reranker,
) -> list[dict]:
    """Run either baseline retrieval or Multi-query retrieval for one test."""
    query = test_case["query"]

    if method == "baseline":
        return retrieve(
            query=query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=TOP_K,
            reranker_model=reranker,
            candidate_k=BASELINE_CANDIDATE_K,
        ) or []

    if method == "multi_query":
        return retrieve_multi_query(
            query=query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            subqueries=SUBQUERIES.get(test_case["id"], []),
            top_k=TOP_K,
            candidate_k=MULTI_QUERY_CANDIDATE_K,
            reranker_model=reranker,
        ) or []

    raise ValueError(f"Unknown retrieval method: {method}")


def calculate_query_metrics(test_case: dict, results: list[dict]) -> dict:
    """Calculate section ranking and evidence phrase coverage at K=3 and K=5."""
    if not test_case["answerable"]:
        return {}

    gold_sections = {
        normalize_section(section)
        for section in test_case.get("relevant_sections", [])
    }

    # A retrieved chunk is relevant when its normalized section has a gold label.
    relevant_flags = [
        normalize_section(get_section(item)) in gold_sections
        for item in results
    ]

    metrics = {}
    checks = test_case.get("evidence_checks", [])

    for k in K_VALUES:
        flags = relevant_flags[:k]
        metrics[f"Hit@{k}"] = int(any(flags))
        metrics[f"Precision@{k}"] = sum(flags) / k

        reciprocal_rank = 0.0
        for rank, relevant in enumerate(flags, start=1):
            if relevant:
                reciprocal_rank = 1.0 / rank
                break
        metrics[f"MRR@{k}"] = reciprocal_rank

        if checks:
            texts = [get_text(item) for item in results[:k]]
            groups_found = sum(
                phrase_group_found(texts, group) for group in checks
            )
            metrics[f"EvidencePhraseRecall@{k}"] = groups_found / len(checks)
            metrics[f"EvidencePhraseGroupsFound@{k}"] = (
                f"{groups_found}/{len(checks)}"
            )

    return metrics


def result_for_report(
    test_case: dict,
    method: str,
    retrieved: list[dict],
) -> dict:
    """Build a compact report entry, preserving texts for evidence review."""
    metrics = calculate_query_metrics(test_case, retrieved)
    entries = []

    gold_sections = {
        normalize_section(section)
        for section in test_case.get("relevant_sections", [])
    }

    for rank, item in enumerate(retrieved, start=1):
        section = get_section(item)
        entries.append({
            "rank": rank,
            "section": section,
            "section_relevant": (
                normalize_section(section) in gold_sections
                if test_case["answerable"] else None
            ),
            "text": get_text(item),
            "rrf_score": item.get("score"),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("bm25_score"),
            "reranker_score": item.get("reranker_score"),
            "multi_query_score": item.get("multi_query_score"),
            "matched_query": item.get("matched_query"),
            "facet_ranks": item.get("facet_ranks"),
        })

    return {
        "id": test_case["id"],
        "query": test_case["query"],
        "answerable": test_case["answerable"],
        "metrics": metrics,
        "retrieved": entries,
        "review_note": test_case.get("review_note", ""),
        "manual_review_required": not test_case["answerable"],
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Macro-average metrics over answerable queries only."""
    answerable = [result for result in results if result["answerable"]]
    summary = {"answerable_queries": len(answerable), "metrics": {}}

    metric_names = [
        name
        for k in K_VALUES
        for name in (
            f"Hit@{k}",
            f"Precision@{k}",
            f"MRR@{k}",
            f"EvidencePhraseRecall@{k}",
        )
    ]

    for name in metric_names:
        values = [
            result["metrics"][name]
            for result in answerable
            if name in result["metrics"]
            and isinstance(result["metrics"][name], (int, float))
        ]
        if values:
            summary["metrics"][name] = float(np.mean(values))

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(name: str, summary: dict) -> None:
    """Print only the main metrics to avoid overwhelming output."""
    print(f"\n{name} — {summary['answerable_queries']} answerable queries")
    for k in K_VALUES:
        print(f"  K={k}")
        for label in (
            f"Hit@{k}",
            f"MRR@{k}",
            f"Precision@{k}",
            f"EvidencePhraseRecall@{k}",
        ):
            value = summary["metrics"].get(label)
            if value is not None:
                print(f"    {label:<26} {value:.3f}")


def print_comparison(baseline: dict, multi_query: dict) -> None:
    """Compare aggregate metrics and show deltas in percentage points."""
    base_summary = aggregate_metrics(baseline)
    multi_summary = aggregate_metrics(multi_query)

    print("\n" + "=" * 76)
    print("BASELINE vs MULTI-QUERY — SUMMARY")
    print("=" * 76)
    print(f"{'Metric':<30}{'Baseline':>12}{'Multi-query':>14}{'Delta':>12}")
    print("-" * 68)

    names = [
        f"Hit@{k}" for k in K_VALUES
    ] + [
        f"MRR@{k}" for k in K_VALUES
    ] + [
        f"Precision@{k}" for k in K_VALUES
    ] + [
        f"EvidencePhraseRecall@{k}" for k in K_VALUES
    ]

    for name in names:
        base_value = base_summary["metrics"].get(name)
        multi_value = multi_summary["metrics"].get(name)
        if base_value is None or multi_value is None:
            continue
        delta = multi_value - base_value
        print(
            f"{name:<30}{base_value:>12.3f}{multi_value:>14.3f}"
            f"{delta:>+12.3f}"
        )

    # Show only cases whose evidence coverage changed, plus the key challenge query.
    baseline_by_id = {item["id"]: item for item in baseline}
    multi_by_id = {item["id"]: item for item in multi_query}
    changed = []

    for test_case in TEST_CASES:
        if not test_case["answerable"]:
            continue
        before = baseline_by_id[test_case["id"]]["metrics"]
        after = multi_by_id[test_case["id"]]["metrics"]
        before_recall = before.get("EvidencePhraseRecall@5")
        after_recall = after.get("EvidencePhraseRecall@5")
        if (
            before_recall != after_recall
            or test_case["id"] == "competition_simulation_method"
        ):
            changed.append((test_case, baseline_by_id[test_case["id"]], multi_by_id[test_case["id"]]))

    if changed:
        print("\nNotable per-query changes (EvidencePhraseRecall@5):")
        for test_case, before, after in changed:
            base_recall = before["metrics"].get("EvidencePhraseRecall@5")
            multi_recall = after["metrics"].get("EvidencePhraseRecall@5")
            base_top = before["retrieved"][0]["section"] if before["retrieved"] else "(none)"
            multi_top = after["retrieved"][0]["section"] if after["retrieved"] else "(none)"
            print(f"  {test_case['id']}: {base_recall} -> {multi_recall}")
            print(f"    top section: baseline={base_top} | multi-query={multi_top}")

    print("\nUnanswerable queries: retrieval rankings do not determine answerability.")
    for item in multi_query:
        if item["manual_review_required"]:
            top = item["retrieved"][0]["section"] if item["retrieved"] else "(no results)"
            print(f"  MANUAL REVIEW | {item['query']} | multi-query top={top}")

    return base_summary, multi_summary


def run_diagnostics(model, chunks, embeddings) -> None:
    """Optional diagnostic mode; omitted during normal fast evaluation."""
    queries = [
        "How is the competition dataset generated?",
        "How did the authors simulate the competition data?",
    ]
    for query in queries:
        print(f"\nDIAGNOSTIC QUERY: {query}")
        diagnose_retrieval(
            query=query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CMAPSS RAG retrievers.")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print Dense/BM25/RRF diagnostic rankings before evaluation.",
    )
    args = parser.parse_args()

    print("[1/4] Loading processed chunks...")
    chunks = load_chunks()
    print(f"      Loaded {len(chunks)} chunks.")

    print(f"[2/4] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [get_text(chunk) for chunk in chunks]
    if any(not text.strip() for text in texts):
        raise ValueError("At least one chunk has empty text in chunks.json.")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"[3/4] Loading CrossEncoder once: {RERANKER_NAME}")
    reranker = CrossEncoder(RERANKER_NAME)

    if args.diagnostics:
        run_diagnostics(model, chunks, embeddings)

    print(f"[4/4] Evaluating {len(TEST_CASES)} queries with both methods...")
    baseline_results = []
    multi_query_results = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(f"  {index:02d}/{len(TEST_CASES)} {test_case['id']}")

        baseline_hits = retrieve_for_case(
            test_case, "baseline", model, chunks, embeddings, reranker
        )
        multi_hits = retrieve_for_case(
            test_case, "multi_query", model, chunks, embeddings, reranker
        )

        baseline_results.append(
            result_for_report(test_case, "baseline", baseline_hits)
        )
        multi_query_results.append(
            result_for_report(test_case, "multi_query", multi_hits)
        )

    baseline_summary, multi_summary = print_comparison(
        baseline_results,
        multi_query_results,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "models": {
            "embedding": MODEL_NAME,
            "reranker": RERANKER_NAME,
        },
        "configuration": {
            "top_k": TOP_K,
            "baseline_candidate_k": BASELINE_CANDIDATE_K,
            "multi_query_candidate_k": MULTI_QUERY_CANDIDATE_K,
            "k_values": list(K_VALUES),
        },
        "notes": [
            "RRF and CrossEncoder scores are ranking signals, not calibrated probabilities.",
            "EvidencePhraseRecall is a lexical proxy, not a semantic entailment check.",
            "Unanswerable cases remain manual-review cases; no automatic abstention model is evaluated here.",
        ],
        "baseline_summary": baseline_summary,
        "multi_query_summary": multi_summary,
        "baseline_results": baseline_results,
        "multi_query_results": multi_query_results,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nComparison report saved to: {REPORT_PATH}")
    print("Evaluation finished. No interactive pause is required.")


if __name__ == "__main__":
    main()
