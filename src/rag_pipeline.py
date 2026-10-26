
from answer_generator import (
    split_into_sentences,
    rank_evidence,
)
from answer_synthesizer import synthesize_answer
from retriever import retrieve


def answer_document_question(
    question: str,
    chunks: list[dict],
    embeddings,
    embedding_model,
    reranker_model,
    top_chunks: int = 10,
    candidate_k: int = 40,
    max_evidence: int = 5,
) -> str:
    """Retrieve evidence, rank it, and generate a grounded answer."""

    if not question.strip():
        return "Please provide a question."

    # 1. Retrieve relevant document chunks.
    retrieved_chunks = retrieve(
        query=question,
        model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=top_chunks,
        candidate_k=candidate_k,
        reranker_model=reranker_model,
    )

    if not retrieved_chunks:
        return (
            "I could not find sufficient evidence in the available "
            "documents to answer this question reliably."
        )

    # 2. Split the chunks into sentences while preserving context.
    candidates = split_into_sentences(retrieved_chunks)

    if not candidates:
        return (
            "I could not find usable evidence in the retrieved documents."
        )

    # 3. Rank the candidate sentences using contextual CrossEncoder.
    ranked_evidence = rank_evidence(
        question=question,
        candidates=candidates,
        model=embedding_model,
        reranker_model=reranker_model,
    )

    # 4. Synthesize the answer using the top-ranked evidence.
    return synthesize_answer(
        question=question,
        ranked_evidence=ranked_evidence,
        max_evidence=max_evidence,
        min_reranker_score=0.0,
    )
