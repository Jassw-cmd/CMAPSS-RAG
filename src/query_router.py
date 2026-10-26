import re
import json

from sentence_transformers import SentenceTransformer, CrossEncoder

from rag_pipeline import answer_document_question

from load_data import load_cmapss
from cmapss_retriever import (
    get_engine_data,
    get_sensor_value,
    get_rul,
)
from retriever import load_index, MODEL_NAME


# --------------------------------------------------
# 1. Query classification
# --------------------------------------------------

DOCUMENT_KEYWORDS = [
    "damage propagation",
    "prognostics",
    "modeling",
    "modelling",
    "methodology",
    "method",
    "approach",
    "theory",
    "how does",
    "how do",
    "why",
    "purpose of",
    "explain",
    "define",
    "definition of",

    # Additional technical concepts
    "wear",
    "degradation",
    "health index",
    "flow loss",
    "efficiency loss",
    "fault evolution",
    "failure",
    "initial wear",
    # Questions about physical components and materials
    "turbine",
    "turbine blade",
    "blade material",
    "material",
    "made of",
    "composition",
    "alloy",
]


def classify_query(query):
    query = query.lower().strip()

    if not query:
        return "unknown"

    has_engine = bool(
        re.search(r"\bengine\s*#?\s*\d+\b", query)
    )
    has_cycle = bool(
        re.search(r"\bcycle\s*#?\s*\d+\b", query)
    )
    has_sensor = bool(
        re.search(r"\bsensor\s*#?\s*\d+\b", query)
    )

    has_structured_target = (
        has_engine
        or has_cycle
        or has_sensor
    )

    needs_document = any(
        keyword in query
        for keyword in DOCUMENT_KEYWORDS
    )

    # RUL definition alone is a document question.
    asks_rul = bool(
        re.search(r"\brul\b|remaining useful life", query)
    )

    if (
        asks_rul
        and not has_structured_target
        and not needs_document
    ):
        return "document"

    # Explicitly combining technical concepts and data.
    if needs_document and has_structured_target:
        return "hybrid"

    if needs_document:
        return "document"

    # RUL with a specific engine requires structured data.
    if asks_rul and has_engine:
        return "structured"

    if has_structured_target:
        return "structured"

    # A general engine question may need clarification.
    return "unknown"


# --------------------------------------------------
# 2. Extract query parameters
# --------------------------------------------------

def extract_parameters(query):
    parameters = {}

    patterns = {
        "engine_id": r"\bengine\s*#?\s*(\d+)\b",
        "cycle": r"\bcycle\s*#?\s*(\d+)\b",
        "sensor_id": r"\bsensor\s*#?\s*(\d+)\b",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            parameters[key] = int(match.group(1))

    return parameters


# --------------------------------------------------
# 3. Structured retrieval
# --------------------------------------------------

def query_structured(query, data):
    parameters = extract_parameters(query)
    engine_id = parameters.get("engine_id")
    cycle = parameters.get("cycle")
    sensor_id = parameters.get("sensor_id")

    if engine_id is None:
        return {
            "status": "missing_information",
            "message": "Which engine are you asking about?",
        }

    try:
        engine = get_engine_data(data, engine_id)

        # Sensor query
        if sensor_id is not None:
            if cycle is None:
                return {
                    "status": "missing_information",
                    "message": (
                        f"At which cycle do you want Sensor "
                        f"{sensor_id} of Engine {engine_id}?"
                    ),
                }

            value = get_sensor_value(
                data, engine_id, cycle, sensor_id
            )

            return {
                "status": "success",
                "type": "sensor_value",
                "engine_id": engine_id,
                "cycle": cycle,
                "sensor_id": sensor_id,
                "value": value,
            }

        # RUL query
        if re.search(
            r"\brul\b|remaining useful life",
            query,
            re.IGNORECASE,
        ):
            result = get_rul(data, engine_id, cycle)

            if result["status"] == "success":
                result["type"] = "rul"
                result["final_cycle"] = int(engine["cycle"].max())

            return result

        # Final observed cycle query
        if re.search(
            r"final observed cycle|last cycle|maximum cycle",
            query,
            re.IGNORECASE,
        ):
            return {
                "status": "success",
                "type": "final_cycle",
                "engine_id": engine_id,
                "final_cycle": int(engine["cycle"].max()),
            }

        # General engine summary
        return {
            "status": "success",
            "type": "engine_summary",
            "engine_id": engine_id,
            "observed_cycles": len(engine),
            "first_cycle": int(engine["cycle"].min()),
            "final_cycle": int(engine["cycle"].max()),
        }

    except ValueError as error:
        return {
            "status": "error",
            "message": str(error),
        }


# --------------------------------------------------
# Format structured results
# --------------------------------------------------

def format_structured_result(result):
    if not isinstance(result, dict):
        return str(result)

    status = result.get("status")

    if status in {"missing_information", "error"}:
        return result.get(
            "message",
            "Unable to retrieve the requested structured data.",
        )

    result_type = result.get("type")
    engine_id = result.get("engine_id")

    if result_type == "sensor_value":
        return (
            f"Sensor {result['sensor_id']} of Engine {engine_id} "
            f"at cycle {result['cycle']} has value "
            f"{result['value']}."
        )

    if result_type == "rul":
        cycle = result.get("cycle")
        rul = result.get("rul")

        if cycle is None or rul is None:
            return (
                f"RUL information for Engine {engine_id} "
                "is incomplete."
            )

        return (
            f"Engine {engine_id} at cycle {cycle} has an "
            f"estimated RUL of {rul} cycles."
        )

    if result_type == "final_cycle":
        return (
            f"The final observed cycle of Engine {engine_id} "
            f"in the loaded data is {result['final_cycle']}."
        )

    if result_type == "engine_summary":
        return (
            f"Engine {engine_id} has "
            f"{result['observed_cycles']} observed cycles, "
            f"from cycle {result['first_cycle']} "
            f"to cycle {result['final_cycle']}."
        )

    # Fallback for an unexpected result structure.
    return json.dumps(result, ensure_ascii=False, indent=2)

# --------------------------------------------------
# 4. Document retrieval + answer synthesis
# --------------------------------------------------

def query_document(
    query,
    model,
    reranker_model,
    chunks,
    embeddings,
):
    answer = answer_document_question(
        question=query,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=model,
        reranker_model=reranker_model,
    )

    return {
        "status": "success",
        "answer": answer,
    }


# --------------------------------------------------
# 5. Unified query pipeline
# --------------------------------------------------


def extract_document_query(query: str) -> str:
    """Remove a combined RUL lookup clause before document retrieval."""

    document_query = re.sub(
        r",?\s*(?:and\s+)?"
        r"(?:what\s+is|what's|calculate|give\s+me|tell\s+me)\s+"
        r"(?:the\s+)?"
        r"(?:rul|remaining\s+useful\s+life)\s+"
        r"of\s+engine\s*#?\s*\d+\s+"
        r"at\s+cycle\s*#?\s*\d+\s*\??",
        "",
        query,
        flags=re.IGNORECASE,
    )

    document_query = re.sub(
        r"\s+",
        " ",
        document_query,
    ).strip(" ,?;")

    return document_query or query.strip()

def process_query(
    query,
    data,
    model,
    reranker_model,
    chunks,
    embeddings,
):
    route = classify_query(query)

    response = {
        "query": query,
        "route": route,
    }

    if route == "structured":
        structured_result = query_structured(query, data)

        response["structured_result"] = structured_result
        response["answer"] = format_structured_result(
            structured_result
        )

    elif route == "document":
        document_result = query_document(
            query,
            model,
            reranker_model,
            chunks,
            embeddings,
        )

        response["document_result"] = document_result
        response["answer"] = document_result["answer"]

    elif route == "hybrid":
        # Retrieve from both the structured dataset and documents.
        structured_result = query_structured(query, data)

        # Remove the structured clause before document retrieval.
        document_query = extract_document_query(query)

        document_result = query_document(
            document_query,
            model,
            reranker_model,
            chunks,
            embeddings,
        )

        response["structured_result"] = structured_result
        response["document_result"] = document_result
        response["document_query"] = document_query

        structured_answer = format_structured_result(
            structured_result
        )

        document_answer = document_result["answer"]

        response["answer"] = (
            "Structured CMAPSS result:\n"
            f"{structured_answer}\n\n"
            "Technical explanation from documents:\n"
            f"{document_answer}"
        )


    else:
        response["answer"] = (
            "I could not determine the required information "
            "source. Please clarify your question."
        )

    return response


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

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

    test_queries = [
        "What is wear in the challenge problem?",
        "What material is the engine turbine blade made of?",
        (
            "How is damage propagation modeled, and what is the "
            "RUL of Engine 24 at cycle 100?"
        ),
    ]

    for query in test_queries:
        result = process_query(
            query=query,
            data=data,
            model=model,
            reranker_model=reranker_model,
            chunks=chunks,
            embeddings=embeddings,
        )

        print("\n" + "=" * 70)
        print("Question:", query)
        print("Route:", result["route"])
        print("\nGenerated answer:")
        print(result["answer"])



if __name__ == "__main__":
    main()
