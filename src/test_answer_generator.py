import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from retriever import retrieve
from answer_generator import (
    generate_extractive_answer,
    split_into_sentences,
    rank_evidence,
)
ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "data" / "processed" / "chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load(ROOT / "data" / "processed" / "embeddings.npy")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

questions = [
    "What is remaining useful life (RUL)?",
    "How did the authors simulate the competition data?",
    "What material is the engine turbine blade made of?",
]

for question in questions:
    print(f"\nQUESTION: {question}")
    print("-" * 70)

    retrieved = retrieve(
        query=question,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=10,
        reranker_model=reranker,
        candidate_k=40,
    )

    candidates = split_into_sentences(retrieved)

    ranked = rank_evidence(
        question=question,
        candidates=candidates,
        model=model,
        reranker_model=reranker,
    )

    print("\n=== TOP EVIDENCE SCORES ===")

    for rank, item in enumerate(ranked[:5], start=1):
        print(
            f"\nRank: {rank}"
            f"\nScore: {item['score']:.4f}"
            f"\nSection: {item['section']}"
            f"\nText: {item['text']}"
        )

    if "simulate the competition data" in question.lower():
        print("\n=== DEBUG: COMPETITION EVIDENCE RANKING ===")

        candidates = split_into_sentences(retrieved)

        ranked = rank_evidence(
            question=question,
            candidates=candidates,
            model=model,
            reranker_model=reranker,
        )

        target_phrase = "simulation of degradation in hpc module"

        target_results = [
            (rank, item)
            for rank, item in enumerate(ranked, start=1)
            if target_phrase in item["text"].lower()
        ]

        print(f"Number of candidate sentences: {len(candidates)}")

        print("\nTop 10 evidence sentences:")
        for rank, item in enumerate(ranked[:10], start=1):
            print(
                f"Evidence rank: {rank}"
                f"\nDense rank: {item['dense_rank']}"
                f"\nCrossEncoder rank: {item.get('reranker_rank', 'N/A')}"
                f"\nCrossEncoder score: {item['score']:.5f}"
                f"\nSection: {item['section']}"
                f"\nText: {item['text']}"
            )

        if target_results:
            rank, item = target_results[0]
            print(f"\nTARGET SENTENCE FOUND at rank {rank}")
            print(f"Score: {item['score']:.4f}")
            print(f"Text: {item['text']}")
            print(f"Dense rank: {item['dense_rank']}")
            print(f"CrossEncoder rank: {item.get('reranker_rank', 'N/A')}")
        else:
            print("\nTARGET SENTENCE NOT FOUND IN CANDIDATES")

    answer = generate_extractive_answer(
        question=question,
        retrieved_chunks=retrieved,
        model=model,
        reranker_model=reranker,
        top_sentences=3
    )

    print(answer)

print("\n=== CHUNKS IN VI. COMPETITION DATA ===")

for i, chunk in enumerate(chunks):
    metadata = chunk.get("metadata", {}) or {}
    section = metadata.get("section", "")

    if "COMPETITION DATA" in section.upper():
        print(f"\nChunk index: {i}")
        print(chunk.get("text", "")[:1500])