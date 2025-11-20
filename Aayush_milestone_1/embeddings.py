from sentence_transformers import SentenceTransformer
import json
import os

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Correct folder where your JSON chunks are stored
INPUT_FOLDER = "outputs/json"

# Output file
OUTPUT_FILE = "chunk_embeddings.json"

os.makedirs(INPUT_FOLDER, exist_ok=True)

data = []

# Loop through all JSON files
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".json"):
        path = os.path.join(INPUT_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)  # <-- list of chunks

        # Loop inside the list
        for entry in chunks:
            text = entry["text"]
            embedding = model.encode(text).tolist()

            data.append({
                "pdf_name": entry["pdf_name"],
                "chunk_id": entry["chunk_id"],
                "text": text,
                "embedding": embedding
            })

# Save embeddings
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Embeddings created and saved in:", OUTPUT_FILE)
