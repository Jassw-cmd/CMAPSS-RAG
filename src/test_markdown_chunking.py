
from load_markdown import load_markdown
from chunk_markdown import chunk_markdown


def main():
    documents = load_markdown()
    document = documents[0]

    chunks = chunk_markdown(
        text=document["text"],
        source=document["metadata"]["source"],
        chunk_size=1200,
        overlap=200
    )

    print("Characters:", len(document["text"]))
    print("Total chunks:", len(chunks))

    assert chunks
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(
        chunk["metadata"]["source"]
        == "Damage Propagation Modeling.md"
        for chunk in chunks
    )
    oversized_chunks = [
        (i, len(chunk["text"]), chunk["metadata"])
        for i, chunk in enumerate(chunks)
        if len(chunk["text"]) > 1200
    ]

    if oversized_chunks:
        print("\nChunks exceeding 1200 characters:")

        for i, length, metadata in oversized_chunks:
            print(
                f"Chunk {i}: length={length}, "
                f"metadata={metadata}"
            )
    else:
        print("All chunks satisfy the size limit.")

    print("Passed!")

    for i, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {i} ---")
        print("Metadata:", chunk["metadata"])
        print("Length:", len(chunk["text"]))
        print(chunk["text"][:400])


if __name__ == "__main__":
    main()
