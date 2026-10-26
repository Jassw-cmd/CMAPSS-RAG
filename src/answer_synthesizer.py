
import os
import re

import requests


ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the available "
    "documents to answer this question reliably."
)


def format_evidence(evidence: list[dict]) -> str:
    """Format ranked evidence with source labels for the LLM."""

    blocks = []

    for i, item in enumerate(evidence, start=1):
        text = item.get("text", "").strip()
        section = item.get("section", "Unknown section")
        context = item.get("context", text).strip()

        blocks.append(
            f"[S{i}] Section: {section}\n"
            f"Evidence: {text}\n"
            f"Context: {context}"
        )

    return "\n\n".join(blocks)


def build_messages(question: str, evidence_text: str) -> list[dict]:
    """Build instructions and evidence for the language model."""

    system_prompt = """
You are a technical question-answering assistant for the CMAPSS project.

Follow these rules:

1. Use only the supplied evidence. Do not use outside knowledge.
2. Answer the question directly and concisely.
3. If at least one evidence item directly supports an answer, answer
   the supported part of the question. Do not abstain merely because
   the evidence does not cover every possible detail.
4. If the source contains a relevant definition, equation, or method,
   explain what it states using only the supplied evidence.
5. Source text may contain Markdown, HTML, or OCR formatting artifacts.
   Do not mistake formatting artifacts for missing evidence.
6. Preserve the distinction between the wear equation w(t) and the
   health-index equation h(t). Do not describe one as the other.
7. Cite each factual claim immediately after the claim it supports.
8. Use only the supplied source labels, such as [S1] or [S2].
   Write grouped sources as [S1] [S2], not [S1, S2].
9. Check that each cited source directly supports the associated claim.
10. Never invent facts, materials, numbers, equations, or source labels.
11. If the requested fact is not directly supported by any supplied
    evidence, return exactly:
INSUFFICIENT_EVIDENCE
12. Keep the answer technically accurate and concise.
""".strip()
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Write a grounded answer with source citations."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



def call_llm(
    messages: list[dict],
    model_name: str | None = None,
    base_url: str | None = None,
    timeout: int = 120,
) -> str:
    """Call a locally running Ollama model through its chat API."""

    model_name = (
        model_name
        or os.getenv("OLLAMA_MODEL")
        or "qwen2.5:3b"
    )
    base_url = (
        base_url
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )

    if not model_name:
        raise ValueError(
            "OLLAMA_MODEL is not configured. "
            "Set it to a model name shown by 'ollama list'."
        )

    url = f"{base_url.rstrip('/')}/api/chat"

    response = requests.post(
        url,
        json={
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "seed": 42,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"].strip()

def normalize_citations(answer: str) -> str:
    """Normalize grouped citations such as [S1, S2] into [S1] [S2]."""

    def expand_group(match):
        numbers = re.findall(r"\d+", match.group(1))
        return " ".join(f"[S{number}]" for number in numbers)

    # Handle groups such as [S1, S2], [S1,S2,S3], or [1, 2].
    answer = re.sub(
        r"\[\s*((?:S\s*)?\d+(?:\s*,\s*(?:S\s*)?\d+)+)\s*\]",
        expand_group,
        answer,
        flags=re.IGNORECASE,
    )

    # Normalize spacing in individual labels, e.g. [S 1] -> [S1].
    answer = re.sub(
        r"\[\s*S\s*(\d+)\s*\]",
        r"[S\1]",
        answer,
        flags=re.IGNORECASE,
    )

    return answer


def validate_citations(
    answer: str,
    evidence_count: int,
) -> bool:
    """Check that the answer cites existing source labels."""

    citation_numbers = [
        int(number)
        for number in re.findall(r"\[S(\d+)\]", answer)
    ]

    # A grounded answer must cite at least one supplied source.
    if not citation_numbers:
        return False

    # Reject references to evidence that was not supplied.
    return all(1 <= number <= evidence_count for number in citation_numbers)


def synthesize_answer(
    question: str,
    ranked_evidence: list[dict],
    max_evidence: int = 5,
    min_reranker_score: float | None = 0.0,
) -> str:
    """Generate a concise answer supported by ranked evidence."""

    if not question.strip():
        return "Please provide a question."

    if not ranked_evidence:
        return ABSTENTION_MESSAGE

    if max_evidence < 1:
        return "max_evidence must be at least 1."

    selected = ranked_evidence[:max_evidence]

    # Apply the project's current experimental answerability threshold.
    best_score = selected[0].get("reranker_score", selected[0].get("score"))

    if (
        min_reranker_score is not None
        and best_score is not None
        and float(best_score) < min_reranker_score
    ):
        return ABSTENTION_MESSAGE

    evidence_text = format_evidence(selected)
    messages = build_messages(question, evidence_text)

    try:
        answer = call_llm(messages)
        answer = normalize_citations(answer)
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        return f"Answer generation failed: {exc}"


    answer = normalize_citations(answer)

    if "INSUFFICIENT_EVIDENCE" in answer.upper():
        return ABSTENTION_MESSAGE

    # Retry once when the model produces invalid citations.
    if not validate_citations(answer, len(selected)):
        valid_labels = ", ".join(
            f"[S{i}]"
            for i in range(1, len(selected) + 1)
        )

        retry_messages = messages + [
            {
                "role": "assistant",
                "content": answer,
            },
           {
                "role": "user",
                "content": (
                    "Revise your previous answer to fix its citations. "
                    f"The valid citation labels are: {valid_labels}. "
                    "Keep factual claims that are directly supported by the "
                    "provided evidence. Place each citation immediately after "
                    "the claim it supports. Do not invent source labels. "
                    "Do not change a supported answer to INSUFFICIENT_EVIDENCE "
                    "merely because its citation formatting was invalid. "
                    "Return INSUFFICIENT_EVIDENCE only if the supplied evidence "
                    "does not directly support an answer."
                ),
            }
        ]

        try:
            answer = call_llm(retry_messages)
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            IndexError,
        ) as exc:
            return f"Answer generation failed during citation repair: {exc}"

        answer = normalize_citations(answer)

        if "INSUFFICIENT_EVIDENCE" in answer.upper():
            return ABSTENTION_MESSAGE

    # Fail closed if citations remain invalid after the retry.
    if not validate_citations(answer, len(selected)):
        return (
            "The model failed citation validation after one retry. "
            "The answer was not presented as verified.\n\n"
            f"Retrieved evidence:\n{evidence_text}"
        )

    return answer

