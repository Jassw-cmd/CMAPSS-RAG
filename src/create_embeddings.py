from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from load_markdown import load_markdown
from chunk_markdown import chunk_markdown

BASE_DIR = Path(__file__).resolve().parents[1]
# print(BASE_DIR)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    documents = load_markdown()
    document = documents[0]

    chunks = chunk_markdown(document["text"], document["metadata"]["source"], chunk_size=1200, overlap=200)
    print("Total chunks:", len(chunks))

    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True, 
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    with open(
        OUTPUT_DIR / "chunks.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", embeddings.shape[1])
    print("Saved embeddings and chunks to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
