
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from retriever import retrieve
from answer_generator import split_into_sentences, rank_evidence
from answer_synthesizer import synthesize_answer


if not os.getenv("OLLAMA_MODEL"):
    raise SystemExit(
        'Set OLLAMA_MODEL, for example: '
        '$env:OLLAMA_MODEL = "qwen2.5:3b"'
    )


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "embeddings.npy"


with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

import numpy as np

embeddings = np.load(EMBEDDINGS_PATH)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


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
        model=embedding_model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=10,
        reranker_model=reranker_model,
        candidate_k=40,
    )

    candidates = split_into_sentences(retrieved)

    ranked_evidence = rank_evidence(
        question=question,
        candidates=candidates,
        model=embedding_model,
        reranker_model=reranker_model,
    )

    answer = synthesize_answer(
        question=question,
        ranked_evidence=ranked_evidence,
        max_evidence=5,
        min_reranker_score=0.0,
    )

    print(answer)
