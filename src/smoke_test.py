
from sentence_transformers import SentenceTransformer, CrossEncoder

from load_data import load_cmapss
from retriever import load_index, MODEL_NAME
from query_router import process_query


def main():
    print("Loading CMAPSS data...")
    data = load_cmapss()

    print("Loading document index...")
    chunks, embeddings = load_index()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading CrossEncoder...")
    reranker_model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    tests = [
        {
            "name": "RUL definition",
            "query": "What is remaining useful life (RUL)?",
            "expected_route": "document",
        },
        {
            "name": "Wear explanation",
            "query": "What is wear in the challenge problem?",
            "expected_route": "document",
        },
        {
            "name": "Unanswerable material question",
            "query": "What material is the engine turbine blade made of?",
            "expected_route": "document",
        },
        {
            "name": "Structured RUL",
            "query": "What is the RUL of Engine 24 at cycle 100?",
            "expected_route": "structured",
            "expected_text": "47 cycles",
        },
        {
            "name": "Structured sensor",
            "query": (
                "What is the value of Sensor 7 "
                "of Engine 24 at cycle 100?"
            ),
            "expected_route": "structured",
            "expected_text": "553.51",
        },
        {
            "name": "Hybrid question",
            "query": (
                "How is damage propagation modeled, and what is the "
                "RUL of Engine 24 at cycle 100?"
            ),
            "expected_route": "hybrid",
            "expected_text": "47 cycles",
        },
    ]

    passed = 0
    failed = 0

    for test in tests:
        query = test["query"]

        print("\n" + "=" * 70)
        print(f"TEST: {test['name']}")
        print(f"Question: {query}")

        try:
            result = process_query(
                query=query,
                data=data,
                model=model,
                reranker_model=reranker_model,
                chunks=chunks,
                embeddings=embeddings,
            )

            answer = result.get("answer", "")
            route = result.get("route", "")

            print(f"Route: {route}")
            print(f"Answer: {answer}")

            assert route == test["expected_route"], (
                f"Expected route {test['expected_route']}, got {route}"
            )

            if "expected_text" in test:
                assert test["expected_text"] in answer, (
                    f"Expected answer to contain "
                    f"{test['expected_text']!r}"
                )

            if test["name"] == "Wear explanation":
                assert "could not find sufficient evidence" not in (
                    answer.lower()
                ), "Wear question was unexpectedly refused."

            if test["name"] == "Unanswerable material question":
                assert "could not find sufficient evidence" in (
                    answer.lower()
                ), "Expected abstention for unsupported material question."

            print("RESULT: PASS")
            passed += 1

        except Exception as error:
            print(f"RESULT: FAIL — {error}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"SMOKE TEST SUMMARY: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
