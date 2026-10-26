from src.llm_client import generate_with_ollama

answer = generate_with_ollama(
    "Explain remaining useful life (RUL) "
    "in jet engine prognostics in two sentences."
)

print(answer)
