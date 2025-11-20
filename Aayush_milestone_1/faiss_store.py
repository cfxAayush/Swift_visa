import json
import faiss
import numpy as np
import os

# Load your final embeddings JSON
EMB_FILE = "chunk_embeddings.json"

with open(EMB_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract vectors + metadata
vectors = []
metadata = []

for item in data:
    vectors.append(item["embedding"])
    metadata.append({
        "pdf_name": item["pdf_name"],
        "chunk_id": item["chunk_id"],
        "text": item["text"]
    })

# Convert to numpy float32
vectors_np = np.array(vectors).astype("float32")

# Vector dimension
d = vectors_np.shape[1]

# Create FAISS index (L2 similarity)
index = faiss.IndexFlatL2(d)

# Add vectors to FAISS
index.add(vectors_np)

print("FAISS index created.")
print("Total vectors stored:", index.ntotal)

# Save FAISS index + metadata
faiss.write_index(index, "visa_index.faiss")

with open("visa_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("Index saved as visa_index.faiss")
print("Metadata saved as visa_metadata.json")
