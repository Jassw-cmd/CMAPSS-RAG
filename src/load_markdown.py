from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = BASE_DIR / "data" / "documents" / "Damage Propagation Modeling.md"

def load_markdown():
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {MARKDOWN_PATH}"
        )

    text = MARKDOWN_PATH.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("Markdown file is empty.")

    return [{
        "text": text,
        "metadata": {
            "source": MARKDOWN_PATH.name
        }
    }]


if __name__ == "__main__":
    documents = load_markdown()

    print("Documents loaded:", len(documents))
    print("Characters:", len(documents[0]["text"]))
    print("Metadata:", documents[0]["metadata"])
    print("\n--- First 2000 characters ---")
    print(documents[0]["text"][:2000])