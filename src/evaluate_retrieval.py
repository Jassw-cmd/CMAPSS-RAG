
from sentence_transformers import SentenceTransformer

from retriever import load_index, retrieve, MODEL_NAME


# Each query has a manually labeled relevant section.
TEST_CASES = [
    {
        "query": "What is remaining useful life (RUL)?",
        "relevant_sections": ["II. PROGNOSTICS"],
    },
    {
        "query": "How is damage propagation modeled?",
        "relevant_sections": [
            "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation",
            "_D. Damage Propagation Model for the Challenge Problem",
        ],
    },
    {
        "query": "How is wear represented in the challenge problem?",
        "relevant_sections": [
            "_D. Damage Propagation Model for the Challenge Problem",
        ],
    },
]

# chuẩn hóa section
def normalize_section(section):
    """Normalize section titles before comparing them."""
    import re

    section = section.lower().strip()

    # Remove Markdown-style formatting characters.
    section = section.replace("_", "").replace("*", "")

    # Normalize whitespace and punctuation.
    section = re.sub(r"\s+", " ", section)
    section = re.sub(r"[^\w\s]", "", section)

    return section.strip()

def evaluate_retrieval(model, chunks, embeddings, k=3):
    hit_scores = []
    precision_scores = []
    reciprocal_ranks = []

    for case in TEST_CASES:
        results = retrieve(
            query=case["query"],
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=k,
        )

        relevant = {
            normalize_section(section)
            for section in case["relevant_sections"]
        }

        # Use the same section normalization as the labels.
        retrieved_sections = [
            item.get("metadata", {}).get("section", "")
            for item in results
        ]

        # A retrieved section counts as relevant if it matches
        # one of the manually labeled sections.

        normalized_retrieved = [
            normalize_section(section)
            for section in retrieved_sections
        ]

        matches = [
            section in relevant
            for section in normalized_retrieved
        ]

        if not any(matches):
            print("Expected:", [repr(s) for s in case["relevant_sections"]])
            print("Retrieved:", [repr(s) for s in retrieved_sections])

        hit_scores.append(float(any(matches)))
        precision_scores.append(sum(matches) / k)

        reciprocal_rank = 0.0
        for rank, is_relevant in enumerate(matches, start=1):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuery: {case['query']}")
        print("Retrieved sections:", retrieved_sections)
        print("\nRetrieved chunks:")
        for rank, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})

            section = metadata.get("section", "Unknown")
            rrf_score = item.get("score")
            dense_score = item.get("dense_score")
            bm25_score = item.get("bm25_score")
            reranker_score = item.get("reranker_score")
            text = item.get("text", item.get("content", ""))

            print(f"\nRank: {rank}")
            print(f"Section: {section}")
            print(f"RRF score: {rrf_score}")
            print(f"Dense cosine score: {dense_score}")
            print(f"BM25 score: {bm25_score}")

            if reranker_score is not None:
                print(f"CrossEncoder score: {reranker_score}")

            print(f"Text: {text[:500]}")
        print("Hit@K:", hit_scores[-1])
        print("Precision@K:", precision_scores[-1])
        print("Reciprocal rank:", reciprocal_rank)

    n = len(TEST_CASES)

    print("\n========== Overall metrics ==========")
    print(f"Hit@{k}: {sum(hit_scores) / n:.3f}")
    print(f"Precision@{k}: {sum(precision_scores) / n:.3f}")
    print(f"MRR@{k}: {sum(reciprocal_ranks) / n:.3f}")


def main():
    chunks, embeddings = load_index()
    model = SentenceTransformer(MODEL_NAME)

    evaluate_retrieval(
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        k=3,
    )


if __name__ == "__main__":
    main()
