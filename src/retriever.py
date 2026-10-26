import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi


# ---------- Configuration ----------

MODEL_NAME = "all-MiniLM-L6-v2"

DENSE_RRF_WEIGHT = 1.05
BM25_RRF_WEIGHT = 1.00
RRF_K = 60


# ---------- Load saved document index ----------

def load_index(
    chunks_path=None,
    embeddings_path=None,
):
    """
    Load document chunks and their precomputed embeddings.

    Default files:
    data/processed/chunks.json
    data/processed/embeddings.npy
    """

    # Find the project root from src/retriever.py.
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    if chunks_path is None:
        chunks_path = processed_dir / "chunks.json"
    else:
        chunks_path = Path(chunks_path)

    if embeddings_path is None:
        embeddings_path = processed_dir / "embeddings.npy"
    else:
        embeddings_path = Path(embeddings_path)

    # Check that both files exist before loading them.
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}"
        )

    # Load the text chunks from JSON.
    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Load the numerical embedding matrix from NumPy.
    embeddings = np.load(embeddings_path)

    # Each chunk must correspond to one embedding.
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks does not match the number "
            "of embeddings: "
            f"{len(chunks)} chunks vs {len(embeddings)} embeddings."
        )

    print(f"Loaded {len(chunks)} document chunks.")
    print(f"Embedding matrix shape: {embeddings.shape}")

    return chunks, embeddings


# ---------- Text processing ----------

def tokenize(text):
    """Split text into lowercase word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def get_chunk_text(chunk):
    """Return the text content of a chunk."""
    if isinstance(chunk, str):
        return chunk

    return chunk.get("text", "")


def get_chunk_metadata(chunk):
    """Return metadata associated with a chunk."""
    if isinstance(chunk, str):
        return {}

    return chunk.get("metadata", {})


# ---------- Hybrid retrieval ----------

def retrieve(
    query,
    model,
    chunks,
    embeddings,
    top_k=3,
    reranker_model=None,
    candidate_k=15,
):
    """
    Hybrid retrieval pipeline:
    1. Dense retrieval using embeddings.
    2. BM25 keyword retrieval.
    3. Reciprocal Rank Fusion (RRF).
    4. Optional CrossEncoder reranking.
    """

    # Validate input.
    if not query or not query.strip():
        return []

    if chunks is None or len(chunks) == 0:
        return []

    if embeddings is None or len(chunks) != len(embeddings):
        return []

    top_k = max(1, min(int(top_k), len(chunks)))
    candidate_k = max(1, int(candidate_k))

    texts = [get_chunk_text(chunk) for chunk in chunks]

    # ---------- 1. Dense retrieval ----------

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dense_scores = embeddings @ query_embedding
    dense_ranking = np.argsort(dense_scores)[::-1]

    # ---------- 2. BM25 retrieval ----------

    tokenized_chunks = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # ---------- 3. Reciprocal Rank Fusion ----------

    # RRF combines rankings instead of comparing raw scores.
    rrf_scores = np.zeros(len(chunks), dtype=float)

    for rank, index in enumerate(dense_ranking, start=1):
        rrf_scores[index] += (
            DENSE_RRF_WEIGHT / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] += (
            BM25_RRF_WEIGHT / (RRF_K + rank)
        )

    rrf_ranking = np.argsort(rrf_scores)[::-1]

    # Select candidates for optional reranking.
    candidate_count = min(
        len(chunks),
        max(top_k, candidate_k),
    )

    candidate_ranking = rrf_ranking[:candidate_count]

    # ---------- 4. Optional CrossEncoder reranking ----------

    reranker_score_by_index = {}

    if reranker_model is not None:
        pairs = [
            (query, texts[int(index)])
            for index in candidate_ranking
        ]

        candidate_scores = np.asarray(
            reranker_model.predict(
                pairs,
                show_progress_bar=False,
            )
        ).reshape(-1)

        # Sort candidates by CrossEncoder score, highest first.
        rerank_order = np.argsort(candidate_scores)[::-1]

        final_ranking = candidate_ranking[
            rerank_order[:top_k]
        ]

        reranker_score_by_index = {
            int(index): float(score)
            for index, score in zip(
                candidate_ranking,
                candidate_scores,
            )
        }

    else:
        # Without CrossEncoder, keep the RRF ranking.
        final_ranking = candidate_ranking[:top_k]

    # ---------- 5. Build results ----------

    results = []

    for index in final_ranking:
        index = int(index)
        chunk = chunks[index]

        results.append({
            "text": texts[index],
            "metadata": get_chunk_metadata(chunk),
            "score": float(rrf_scores[index]),
            "dense_score": float(dense_scores[index]),
            "bm25_score": float(bm25_scores[index]),
            "reranker_score": reranker_score_by_index.get(index),
        })

    return results


# ---------- Multi-query retrieval ----------

def retrieve_multi_query(
    query: str,
    model,
    chunks,
    embeddings,
    subqueries: list[str],
    top_k: int = 5,
    candidate_k: int = 25,
    reranker_model=None,
):
    """
    Multi-query retrieval using rank fusion.

    1. Retrieve candidates for the original query and subqueries.
    2. Deduplicate candidates.
    3. Score candidates using CrossEncoder for each query.
    4. Fuse the query-specific rankings using RRF.
    5. Return the top_k candidates.
    """

    if not query or not query.strip():
        return []

    if reranker_model is None:
        raise ValueError(
            "retrieve_multi_query requires a reranker_model."
        )

    # Remove empty and duplicated queries.
    queries = list(dict.fromkeys(
        q.strip()
        for q in [query, *(subqueries or [])]
        if q and q.strip()
    ))

    # 1. Collect candidates from every query.
    candidates_by_text = {}

    for current_query in queries:
        hits = retrieve(
            query=current_query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=candidate_k,
            candidate_k=candidate_k,
            reranker_model=None,
        )

        for hit in hits:
            text = hit.get("text", "").strip()

            if text and text not in candidates_by_text:
                candidates_by_text[text] = hit.copy()

    candidates = list(candidates_by_text.values())

    if not candidates:
        return []

    # 2. Score every candidate against every query.
    score_rows = []

    for current_query in queries:
        pairs = [
            (current_query, candidate["text"])
            for candidate in candidates
        ]

        scores = np.asarray(
            reranker_model.predict(
                pairs,
                show_progress_bar=False,
            )
        ).reshape(-1)

        score_rows.append(scores)

    # Rows = queries; columns = candidate chunks.
    score_matrix = np.vstack(score_rows)

    # 3. Convert each query's scores into ranks.
    # Rank 1 means the highest CrossEncoder score.
    rank_matrix = np.empty(
        score_matrix.shape,
        dtype=int,
    )

    for query_index in range(len(queries)):
        order = np.argsort(
            score_matrix[query_index]
        )[::-1]

        rank_matrix[query_index, order] = np.arange(
            1,
            len(candidates) + 1,
        )

    # 4. Fuse the rankings using weighted RRF.
    # Give the original query twice the weight of each subquery.
    query_weights = np.ones(len(queries), dtype=float)
    query_weights[0] = 2.0

    multi_query_scores = np.zeros(
        len(candidates),
        dtype=float,
    )

    for query_index, weight in enumerate(query_weights):
        multi_query_scores += (
            weight
            / (RRF_K + rank_matrix[query_index])
        )

    # Identify the query under which each candidate ranks highest.
    best_query_indices = rank_matrix.argmin(axis=0)

    # 5. Attach diagnostic information to every result.
    for i, candidate in enumerate(candidates):
        best_query_index = int(best_query_indices[i])

        candidate["reranker_score"] = float(
            score_matrix[0, i]
        )

        candidate["multi_query_score"] = float(
            multi_query_scores[i]
        )

        candidate["matched_query"] = queries[
            best_query_index
        ]

        candidate["facet_scores"] = {
            current_query: float(score_matrix[j, i])
            for j, current_query in enumerate(queries)
        }

        candidate["facet_ranks"] = {
            current_query: int(rank_matrix[j, i])
            for j, current_query in enumerate(queries)
        }

    # 6. Rank by the fused score, not by the maximum raw score.
    candidates.sort(
        key=lambda item: item["multi_query_score"],
        reverse=True,
    )

    return candidates[:max(1, int(top_k))]


# ---------- Retrieval diagnostics ----------

def diagnose_retrieval(
    query,
    model,
    chunks,
    embeddings,
    top_k=10,
):
    """
    Compare Dense, BM25, and RRF rankings for one query.
    Displays the best candidates from each method.
    """

    if not query or not query.strip():
        print("Error: query is empty.")
        return

    if chunks is None or len(chunks) == 0:
        print("Error: chunks are empty.")
        return

    if embeddings is None or len(chunks) != len(embeddings):
        print("Error: chunks and embeddings are mismatched.")
        return

    texts = [get_chunk_text(chunk) for chunk in chunks]

    # ---------- 1. Dense retrieval ----------

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dense_scores = embeddings @ query_embedding
    dense_ranking = np.argsort(dense_scores)[::-1]

    # ---------- 2. BM25 retrieval ----------

    tokenized_chunks = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_chunks)

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # ---------- 3. RRF ----------

    rrf_scores = np.zeros(len(chunks), dtype=float)

    for rank, index in enumerate(dense_ranking, start=1):
        rrf_scores[index] += (
            DENSE_RRF_WEIGHT / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] += (
            BM25_RRF_WEIGHT / (RRF_K + rank)
        )

    rrf_ranking = np.argsort(rrf_scores)[::-1]

    # Convert rankings into index -> rank mappings.
    dense_rank = {
        int(index): rank
        for rank, index in enumerate(dense_ranking, start=1)
    }

    bm25_rank = {
        int(index): rank
        for rank, index in enumerate(bm25_ranking, start=1)
    }

    rrf_rank = {
        int(index): rank
        for rank, index in enumerate(rrf_ranking, start=1)
    }

    # Union of the top-k candidates from all three rankings.
    limit = min(max(1, int(top_k)), len(chunks))

    candidates = set(dense_ranking[:limit])
    candidates.update(bm25_ranking[:limit])
    candidates.update(rrf_ranking[:limit])

    # Display candidates in RRF order.
    candidates = sorted(
        candidates,
        key=lambda index: rrf_rank[int(index)],
    )

    print(f"\nQuery: {query}")

    print(
        f"{'RRF':>4} {'Dense':>5} {'BM25':>5} "
        f"{'Dense score':>11} {'BM25 score':>11} "
        f"{'RRF score':>10}  Section"
    )

    print("-" * 100)

    for index in candidates:
        index = int(index)
        metadata = get_chunk_metadata(chunks[index])

        section = (
            metadata.get("section")
            or metadata.get("heading")
            or "Unknown section"
        )

        preview = texts[index].replace("\n", " ")[:90]

        print(
            f"{rrf_rank[index]:>4} "
            f"{dense_rank[index]:>5} "
            f"{bm25_rank[index]:>5} "
            f"{dense_scores[index]:>11.4f} "
            f"{bm25_scores[index]:>11.4f} "
            f"{rrf_scores[index]:>10.6f}  "
            f"{section}"
        )

        print(f"      Chunk: {preview}")
