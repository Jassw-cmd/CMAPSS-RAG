
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag_pipeline import answer_document_question


ROOT = Path(__file__).resolve().parent.parent

with open(
    ROOT / "data" / "processed" / "chunks.json",
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)

embeddings = np.load(
    ROOT / "data" / "processed" / "embeddings.npy"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

if not os.getenv("OLLAMA_MODEL"):
    raise SystemExit(
        'Run: $env:OLLAMA_MODEL = "qwen2.5:3b"'
    )

questions = [
    "What is remaining useful life (RUL)?",
    "How did the authors simulate the competition data?",
    "What material is the engine turbine blade made of?",
]

for question in questions:
    print(f"\nQUESTION: {question}")
    print("-" * 70)

    answer = answer_document_question(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )

    print(answer)
