import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

def generate_with_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json = payload,
            timeout=180,
        )
        response.raise_for_status()

        result = response.json()
        return result["response"].strip()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Check that Ollama is running."
        ) from exc
