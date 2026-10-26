"""
Evaluate chunk-level evidence labels from evidence_review_report_labeled.json.

Run from the project root:
    python evaluate_evidence_labels.py evidence_review_report_labeled.json

This script uses only the Python standard library. Labels are provisional until
reviewed by a human against the source document.
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean

K_VALUES = [1, 3, 5]


def calculate_evidence_metrics(items, k):
    """Compute Hit@K, MRR@K, Precision@K and unsupported rate for answerable queries."""
    hits, reciprocal_ranks, precisions, unsupported_rates = [], [], [], []
    per_query = []

    for item in items:
        retrieved = item.get("retrieved", [])[:k]
        if not retrieved:
            continue

        relevant = [r.get("evidence_relevant") is True for r in retrieved]
        first_rank = next(
            (r.get("rank", i + 1) for i, r in enumerate(retrieved)
             if r.get("evidence_relevant") is True),
            None,
        )
        hit = int(any(relevant))
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        precision = sum(relevant) / len(retrieved)
        unsupported_rate = sum(
            r.get("evidence_relevant") is False for r in retrieved
        ) / len(retrieved)

        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)
        precisions.append(precision)
        unsupported_rates.append(unsupported_rate)
        per_query.append({
            "id": item.get("id"),
            "query": item.get("query"),
            "k": k,
            "evidence_hit": hit,
            "evidence_mrr": round(reciprocal_rank, 6),
            "evidence_precision": round(precision, 6),
            "unsupported_retrieval_rate": round(unsupported_rate, 6),
            "first_evidence_rank": first_rank,
        })

    average = lambda values: round(mean(values), 6) if values else None
    return {
        "question_count": len(hits),
        "evidence_hit_at_k": average(hits),
        "evidence_mrr_at_k": average(reciprocal_ranks),
        "evidence_precision_at_k": average(precisions),
        "unsupported_retrieval_rate_at_k": average(unsupported_rates),
        "per_query": per_query,
    }


def calculate_section_hit(items, k):
    """Compute section-level Hit@K separately; it is not chunk-level evidence."""
    hits = []
    for item in items:
        retrieved = item.get("retrieved", [])[:k]
        if retrieved:
            hits.append(int(any(r.get("section_relevant") is True for r in retrieved)))
    return round(mean(hits), 6) if hits else None


def main():
    project_root = Path(__file__).resolve().parent.parent

    if len(sys.argv) > 1:
        source = Path(sys.argv[1])

        if not source.is_absolute():
            source = Path.cwd() / source
    else:
        source = (
            Path(__file__).resolve().parent
            / "evidence_review_report_labeled.json"
        )

    if not source.is_file():
        raise FileNotFoundError(
            f"Evidence label file not found: {source}"
        )

    with source.open(encoding="utf-8") as file:
        report = json.load(file)
        
    items = report.get("items", [])
    answerable = [item for item in items if item.get("answerable") is True]
    unanswerable = [item for item in items if item.get("answerable") is False]

    output = {
        "source_report": str(source),
        "note": (
            "Metrics use provisional labels and should not be treated as final "
            "until manually verified. The unanswerable check is not abstention accuracy."
        ),
        "counts": {
            "all_queries": len(items),
            "answerable_queries": len(answerable),
            "unanswerable_queries": len(unanswerable),
        },
        "answerable_query_metrics": {},
        "unanswerable_query_checks": {},
    }
    per_query_rows = []

    for k in K_VALUES:
        metrics = calculate_evidence_metrics(answerable, k)
        per_query_rows.extend(metrics.pop("per_query"))
        metrics["section_hit_at_k"] = calculate_section_hit(answerable, k)
        output["answerable_query_metrics"][str(k)] = metrics

        any_relevant = []
        chunk_relevant = []
        for item in unanswerable:
            retrieved = item.get("retrieved", [])[:k]
            if retrieved:
                any_relevant.append(int(any(
                    row.get("evidence_relevant") is True for row in retrieved
                )))
                chunk_relevant.extend(
                    int(row.get("evidence_relevant") is True) for row in retrieved
                )
        output["unanswerable_query_checks"][str(k)] = {
            "queries_with_any_chunk_labeled_evidence_relevant_rate":
                round(mean(any_relevant), 6) if any_relevant else None,
            "chunks_labeled_evidence_relevant_rate":
                round(mean(chunk_relevant), 6) if chunk_relevant else None,
            "interpretation": "Label-consistency check only; not a model abstention score.",
        }

    json_path = source.with_name("evidence_metrics_report.json")
    csv_path = source.with_name("evidence_metrics_per_query.csv")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        columns = [
            "id", "query", "k", "evidence_hit", "evidence_mrr",
            "evidence_precision", "unsupported_retrieval_rate", "first_evidence_rank",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_query_rows)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print("Answerable query metrics:")
    for k, values in output["answerable_query_metrics"].items():
        print(
            f"  K={k}: Hit={values['evidence_hit_at_k']}, "
            f"MRR={values['evidence_mrr_at_k']}, "
            f"Precision={values['evidence_precision_at_k']}, "
            f"UnsupportedRate={values['unsupported_retrieval_rate_at_k']}, "
            f"SectionHit={values['section_hit_at_k']}"
        )


if __name__ == "__main__":
    main()
