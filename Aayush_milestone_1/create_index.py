import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

# FOLDERS
PDF_FOLDER = "pdfs"
OUTPUT_INDEX = "outputs/visa_index.faiss"
OUTPUT_META = "outputs/visa_metadata.json"

# MODEL + CHUNK SETTINGS
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
OVERLAP = 50


# -- READ PDF --
def read_pdf(path):
    text = ""
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += " " + content
    return text.strip()


# -- CHUNKING LOGIC --
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + size])
        chunks.append(chunk)
        start += size - overlap

    return chunks


# ------------ MAIN INDEX BUILDER ----------------
def build_index():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    metadata = []
    vectors = []

    chunk_id = 0

    print("Processing PDFs from:", PDF_FOLDER)

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, file)
        print(f"\nReading: {file}")

        text = read_pdf(pdf_path)
        chunks = chunk_text(text)

        print(f"Total chunks created: {len(chunks)}")

        for chunk in chunks:
            # Generate embedding
            vec = embedder.encode([chunk])[0].astype("float32")
            vectors.append(vec)

            # Save metadata
            metadata.append({
                "pdf_name": file,
                "chunk_id": chunk_id,
                "text": chunk
            })

            chunk_id += 1

    # Convert vectors to array
    vectors = np.array(vectors).astype("float32")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save index
    os.makedirs("outputs", exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX)

    # Save metadata
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nIndex + Metadata successfully created.")
    print("Total chunks:", len(metadata))
    print("Index saved to:", OUTPUT_INDEX)
    print("Metadata saved to:", OUTPUT_META)


if __name__ == "__main__":
    build_index()
