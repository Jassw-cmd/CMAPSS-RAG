
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

#Documents
documents = [
    "Engine 24 is operating at cycle 100.",
    "Engine 24 has a final observed cycle of 150.",
    "Engine 12 is operating at cycle 80.",
    "Sensor 7 measures a parameter of the engine.",
    "CMAPSS is a dataset for studying aircraft engine degradation.",
    "Remaining Useful Life (RUL) estimates how many cycles remain before an engine reaches failure.",
    "Engine sensor measurements can be used to monitor changes in engine condition.",
    "The current cycle of an engine is not necessarily its final observed cycle.",
    "Engine degradation refers to the deterioration of engine performance over time.",
    "A retrieval system ranks documents according to their relevance to a query."
]

documents_embeddings = model.encode(documents)

def retrieve(documents, documents_embeddings, model, query: str, top_k: int = 3):
    """
    Using documents embeddings and query embeddings to return the top_k most relevant to the query by cosine similarity
    """
    if not 1 <= top_k <= len(documents):
        raise ValueError("top_k must be between 1 and len(documents)")
    if len(documents) != len(documents_embeddings):
        raise ValueError("Documents and embeddings must have equal lengths")

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, documents_embeddings)[0]
    rank_indices = scores.argsort()[::-1][:top_k]

    result = [
        {
            "rank": rank,
            "document": documents[index],
            "score": float(scores[index])
        }
        for rank, index in enumerate(rank_indices, start=1)
    ]
    return result


def evaluate_retrieval(test_cases, documents, documents_embeddings, model, top_k: int = 3):
    #prefix check
    if not test_cases:
        raise ValueError("test_cases must not be empty")
    if not 1 <= top_k <= len(documents):
        raise ValueError("top_k must be between 1 and length of documents")
    if len(documents) != len(documents_embeddings):
        raise ValueError("Documents and embeddings must have equal lengths")
    if any(not test.get("expected") for test in test_cases):
        raise ValueError("Each test case must have expected documents")
    


    
    total_hits = 0
    total_precision = 0.0
    total_reciprocal_rank = 0.0

    details = []

    for test in test_cases:
        query = test["query"]
        expected = test["expected"]

        #normailize relevant_docs
        if isinstance(expected, str):
            relevant_docs = {expected}
        else:
            relevant_docs = set(expected)

        results = retrieve(documents, documents_embeddings, model, query, top_k = top_k)

        retrieved_documents = [item["document"] for item in results]

        relevant_counts = sum(doc in relevant_docs for doc in retrieved_documents)

        # Hit@K
        hit = int (relevant_counts > 0)

        # Precision@K
        precision = relevant_counts / len(retrieved_documents)

        # Reciprocal Rank
        first_relevant_rank = next(
            (
                item["rank"]
                for item in results
                if item["document"] in relevant_docs
            ),
            None
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        
        total_hits += hit
        total_precision += precision
        total_reciprocal_rank += reciprocal_rank

        details.append({
            "query": query,
            "expected": expected,
            "hit": hit,
            "precision_at_k": precision,
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": reciprocal_rank,
            "results": results
        })

    n = len(test_cases)
    hit_rate = total_hits / n
    precision_at_k = total_precision / n
    mrr_at_k = total_reciprocal_rank / n

    return {
        "total_queries": len(test_cases),
        
        "top_k": top_k,
        "hit_rate": hit_rate,
        "precision_at_k": precision_at_k,
        "mrr_at_k": mrr_at_k,
        "details": details
    }

def analyze_score_margin(test_cases, documents, documents_embeddings, model):
    for test in test_cases:
        results = retrieve(
            documents, documents_embeddings, model,
            query=test["query"], top_k=2
        )

        top_score = results[0]["score"]
        second_score = results[1]["score"]
        margin = top_score - second_score

        print(f"\nQuery: {test['query']}")
        print(f"Top-1 score: {top_score:.3f}")
        print(f"Top-2 score: {second_score:.3f}")
        print(f"Score margin: {margin:.3f}")
        print(f"Top-1 document: {results[0]['document']}")
        print(f"Top-2 document: {results[1]['document']}")


# main run

# test_cases: queries and expected answers
test_cases = [
    {
        "query": "At which cycle is Engine 24 operating?",
        "expected": [
            "Engine 24 is operating at cycle 100."
        ]
    },
    {
        "query": "When was Engine 24 last observed?",
        "expected": [
            "Engine 24 has a final observed cycle of 150."
        ]
    },
    {
        "query": "What is the purpose of CMAPSS?",
        "expected": [
            "CMAPSS is a dataset for studying aircraft engine degradation."
        ]
    },
    {
        "query": "What does RUL tell us about an engine?",
        "expected": [
            "Remaining Useful Life (RUL) estimates how many cycles remain before an engine reaches failure."
        ]
    },
    {
        "query": "How can we monitor changes in engine condition?",
        "expected": [
            "Sensor 7 measures a parameter of the engine.",
            "Engine sensor measurements can be used to monitor changes in engine condition."
        ]
    },
    {
        "query": "What is the difference between a current cycle and a final observed cycle?",
        "expected": [
            "Engine 24 is operating at cycle 100.",
            "Engine 24 has a final observed cycle of 150.",
            "The current cycle of an engine is not necessarily its final observed cycle."
        ]
    }
]




for k in [1, 3, 5]:
    metrics = evaluate_retrieval(
        test_cases,
        documents,
        documents_embeddings,
        model,
        top_k=k
    )

    print(f"\n--- K = {k} ---")
    print(f"Hit@{k}:       {metrics['hit_rate']:.2%}")
    print(f"Precision@{k}: {metrics['precision_at_k']:.2%}")
    print(f"MRR@{k}:       {metrics['mrr_at_k']:.3f}")

    for item in metrics["details"]:
        print(f"\nQuery: {item['query']}")
        print(f"Expected: {item['expected']}")
        print(f"First relevant rank: {item['first_relevant_rank']}")
        print(f"Precision@{k}: {item['precision_at_k']:.3f}")
        print(f"RR@{k}: {item['reciprocal_rank']:.3f}")

        for result in item["results"]:
            print(
                f"  Rank {result['rank']} | "
                f"Score {result['score']:.3f} | "
                f"{result['document']}"
            )


print("-"*70)

analyze_score_margin(
    test_cases,
    documents,
    documents_embeddings,
    model
)
