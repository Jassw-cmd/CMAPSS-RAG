from load_pdf import load_pdf_pages
from chunk_pdf import chunk_pdf_pages

pages = load_pdf_pages()
chunks = chunk_pdf_pages(pages)

print("Total pages:",len(pages))
print("Total chunks:", len(chunks))

"""
print("-"*50)
print(pages)
print("-"*50)
"""
for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", chunk["metadata"])
    print("Length:", len(chunk["text"]))
    print(chunk["text"][:300])