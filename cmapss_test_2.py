from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert a sentence into a vector
text = "Engine 24 is operating at cycle 100."
embedding = model.encode(text)

print("Vector shape:", embedding.shape)
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
