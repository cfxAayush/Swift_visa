from groq import Groq
from dotenv import load_dotenv
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Load embedder + FAISS + metadata
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../Aayush_milestone_1/outputs/visa_index.faiss")


with open("../Aayush_milestone_1/outputs/json/visa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


def embed_text(text):
    return embedder.encode([text])[0].astype("float32")


def retrieve_chunks(query, k=5):
    qvec = embed_text(query)
    _, ids = index.search(np.array([qvec]), k)

    chunks = []
    for cid in ids[0]:
        if cid == -1:
            continue
        chunks.append(metadata[cid])

    return chunks


def format_chunks(chunks):
    """Cleaner chunk formatting before sending to model."""
    formatted = []
    for c in chunks:
        formatted.append(f"[CHUNK {c['chunk_id']}]\n{c['text']}")
    return "\n\n".join(formatted)


def ask_groq_from_pdf(question, chunks):
    context = format_chunks(chunks)

    prompt = f"""
Answer ONLY using the PDF context.

Question:
{question}

Context:
{context}

Return:
1. Final answer
2. Explanation with chunk references
3. Confidence (0–1)
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return resp.choices[0].message.content


if __name__ == "__main__":
    question = "What are the requirements for USA visa?"
    chunks = retrieve_chunks(question, k=5)

    print("\n===== Retrieved Chunks =====")
    for c in chunks:
        print(f"\n--- CHUNK {c['chunk_id']} ---\n{c['text'][:250]}...")  # preview

    print("\n===== Model Answer =====\n")
    answer = ask_groq_from_pdf(question, chunks)
    print(answer)

