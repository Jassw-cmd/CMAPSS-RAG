
import re
import numpy as np




def split_into_sentences(retrieved_chunks: list[dict]) -> list[dict]:
    """Split chunks into sentences while preserving local context."""

    candidates = []
    seen_sentences = set()

    for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {}) or {}
        section = metadata.get("section", "Unknown section")

        if not text:
            continue

        # Clean common HTML artifacts from converted documents.
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

        # Turn numbered items into separate lines.
        text = re.sub(r"\s*-\s*\d+\)\s*", "\n", text)

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]

        for i, sentence in enumerate(sentences):
            if len(sentence) < 25 or sentence.endswith(":"):
                continue

            key = re.sub(r"\s+", " ", sentence.lower())

            if key in seen_sentences:
                continue

            seen_sentences.add(key)

            # Preserve the previous and next sentences as context.
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            context = " ".join(sentences[start:end])

            candidates.append({
                "text": sentence,
                "section": section,
                "context": context[:1000],
                "chunk_rank": chunk_rank,
            })

    return candidates





def rank_evidence(
    question: str,
    candidates: list[dict],
    model,
    reranker_model=None,
    rrf_k: int = 60,
) -> list[dict]:
    """Rank evidence using contextual CrossEncoder scores."""

    if not candidates:
        return []

    # Step 1: Calculate dense similarity for diagnostics and fallback.
    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    sentence_embeddings = model.encode(
        [item["text"] for item in candidates],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dense_scores = sentence_embeddings @ question_embedding
    dense_order = np.argsort(dense_scores)[::-1]

    dense_ranks = np.empty(len(candidates), dtype=int)
    dense_ranks[dense_order] = np.arange(1, len(candidates) + 1)

    # Step 2: Let CrossEncoder see the section and local context.
    if reranker_model is not None:
        pairs = [
            [
                question,
                (
                    f"Section: {item['section']}\n"
                    f"Context: {item.get('context', item['text'])}\n"
                    f"Evidence sentence: {item['text']}"
                ),
            ]
            for item in candidates
        ]

        reranker_scores = np.asarray(
            reranker_model.predict(pairs)
        ).reshape(-1)

        reranker_order = np.argsort(reranker_scores)[::-1]

        reranker_ranks = np.empty(len(candidates), dtype=int)
        reranker_ranks[reranker_order] = np.arange(
            1, len(candidates) + 1
        )

        # Use CrossEncoder as the primary ranking signal.
        # Dense similarity is only a tie-breaker.
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda i: (
                float(reranker_scores[i]),
                float(dense_scores[i]),
            ),
            reverse=True,
        )

    else:
        reranker_scores = None
        reranker_ranks = None
        ranked_indices = list(dense_order)

    # Step 3: Return ranked evidence with diagnostic scores.
    ranked = []

    for index in ranked_indices:
        item = {
            **candidates[int(index)],
            "score": float(
                reranker_scores[index]
                if reranker_scores is not None
                else dense_scores[index]
            ),
            "dense_score": float(dense_scores[index]),
            "dense_rank": int(dense_ranks[index]),
        }

        if reranker_scores is not None:
            item["reranker_score"] = float(reranker_scores[index])
            item["reranker_rank"] = int(reranker_ranks[index])

        ranked.append(item)

    return ranked




def generate_extractive_answer(
    question: str,
    retrieved_chunks: list[dict],
    model,
    reranker_model=None,
    top_sentences: int = 3,
    min_similarity: float = 0.25,
    min_reranker_score=0.0,
) -> str:
    """
    Extract relevant evidence sentences and show their source sections.

    This function extracts evidence; it does not generate a complete
    natural-language answer or guarantee that the question is answerable.
    """

    if not question.strip():
        return "Please provide a question."

    if not retrieved_chunks:
        return "I could not find sufficient evidence in the available documents."

    if top_sentences < 1:
        return "top_sentences must be at least 1."

    candidates = split_into_sentences(retrieved_chunks)

    if not candidates:
        return "No usable evidence sentences were found in the retrieved chunks."

    ranked = rank_evidence(
        question=question,
        candidates=candidates,
        model=model,
        reranker_model=reranker_model,
        rrf_k = 60
    )

    best_score = ranked[0]["score"]

    # Use the cosine threshold only when using embedding similarity.
    if reranker_model is None and best_score < min_similarity:
        return (
            "I could not find sufficiently relevant evidence "
            "to answer this question reliably."
        )

    # A CrossEncoder threshold is optional and must be calibrated
    # using this project's evaluation examples before being enabled.
    if (
        reranker_model is not None
        and min_reranker_score is not None
        and best_score < min_reranker_score
    ):
        return (
            "I could not find sufficiently relevant evidence "
            "to answer this question reliably."
        )

    selected = ranked[:top_sentences]

    lines = ["Relevant evidence found in the documents:"]

    for i, item in enumerate(selected, start=1):
        lines.append(
            f"[S{i}] {item['text']}\n"
            f"     Source section: {item['section']}"
        )

    lines.append(
        "\nNote: These are extracted evidence sentences, "
        "not a fully generated answer. Relevance scores alone "
        "do not establish that the question is answerable."
    )

    return "\n".join(lines)
