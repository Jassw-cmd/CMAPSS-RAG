
from query_router import (
    classify_query,
    extract_parameters,
    extract_document_query,
)


def test_query_routes():
    cases = [
        (
            "What is remaining useful life (RUL)?",
            "document",
        ),
        (
            "What is wear in the challenge problem?",
            "document",
        ),
        (
            "What material is the engine turbine blade made of?",
            "document",
        ),
        (
            "How is damage propagation modeled?",
            "document",
        ),
        (
            "What is the RUL of Engine 24 at cycle 100?",
            "structured",
        ),
        (
            "What is the value of Sensor 7 of Engine 24 at cycle 100?",
            "structured",
        ),
        (
            "How is damage propagation modeled, and what is the "
            "RUL of Engine 24 at cycle 100?",
            "hybrid",
        ),
    ]

    for question, expected_route in cases:
        actual_route = classify_query(question)

        assert actual_route == expected_route, (
            f"\nQuestion: {question}"
            f"\nExpected: {expected_route}"
            f"\nActual: {actual_route}"
        )

        print(f"PASS [{actual_route}]: {question}")


def test_parameter_extraction():
    question = (
        "What is the value of Sensor 7 of Engine 24 at cycle 100?"
    )

    expected = {
        "engine_id": 24,
        "cycle": 100,
        "sensor_id": 7,
    }

    actual = extract_parameters(question)

    assert actual == expected, (
        f"Expected {expected}, got {actual}"
    )

    print("PASS: parameter extraction")


def test_hybrid_query_extraction():
    question = (
        "How is damage propagation modeled, and what is the "
        "RUL of Engine 24 at cycle 100?"
    )

    expected = "How is damage propagation modeled"
    actual = extract_document_query(question)

    assert actual == expected, (
        f"\nExpected: {expected}"
        f"\nActual: {actual}"
    )

    print("PASS: hybrid query extraction")


if __name__ == "__main__":
    test_query_routes()
    test_parameter_extraction()
    test_hybrid_query_extraction()

    print("\nAll router logic tests passed.")
