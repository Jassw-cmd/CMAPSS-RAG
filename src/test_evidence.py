r"""Generate a focused, human-reviewable evidence audit for CMAPSS RAG.

Run from the project root:
    .\.venv\Scripts\python.exe src\test_evidence.py

The script writes data/processed/evidence_review_report.json. For unanswerable
questions, inspect whether the retrieved passages directly support an answer;
retrieval scores alone cannot establish answerability.
"""

import json
from pathlib import Path

from evaluate_rag import (
    MODEL_NAME,
    TEST_CASES,
    TOP_K,
    BASE_DIR,
    load_chunks,
    load_embeddings,
    evaluate_one_query,
)


OUTPUT_PATH = BASE_DIR / "data" / "processed" / "evidence_review_report.json"


def main() -> None:
    """Run the same test set and save ranked passages for manual review."""
    chunks = load_chunks()
    model, embeddings = load_embeddings(chunks)

    report = {
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "instructions": (
            "For answerable queries, section_relevant is a gold-section proxy, "
            "not a guarantee that the whole passage supports every claim. "
            "For unanswerable queries, section_relevant is null by design; "
            "manually inspect whether the passage directly answers the question."
        ),
        "items": [],
    }

    for case in TEST_CASES:
        result = evaluate_one_query(case, chunks, embeddings, model)
        item = {
            "id": result["id"],
            "query": result["query"],
            "answerable": result["answerable"],
            "review_note": result["review_note"],
            "retrieved": result["retrieved"],
        }
        report["items"].append(item)

        print("\n" + "=" * 88)
        print(f"[{item['id']}] {item['query']}")
        print(f"Ground truth answerable: {item['answerable']}")
        for passage in item["retrieved"]:
            label = passage["section_relevant"]
            if label is True:
                label_text = "GOLD SECTION"
            elif label is False:
                label_text = "OTHER SECTION"
            else:
                label_text = "MANUAL REVIEW"
            print(
                f"\n#{passage['rank']} [{label_text}] {passage['section']}\n"
                f"RRF={passage['rrf_score']:.5f} | "
                f"Dense={passage['dense_score']:.5f} | "
                f"BM25={passage['bm25_score']:.5f}"
            )
            print(" ".join(passage["text"].split())[:900])
        if not item["answerable"]:
            print("\nMANUAL REVIEW: does any retrieved passage directly answer the question?")
            print(item["review_note"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nEvidence review report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
