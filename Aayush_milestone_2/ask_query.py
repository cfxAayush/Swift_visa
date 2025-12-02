from groq import Groq
from dotenv import load_dotenv
import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment + model
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# FIX: Load FAISS + metadata from milestone_1/outputs
BASE_PATH = "../Aayush_milestone_1/outputs"

index = faiss.read_index(f"{BASE_PATH}/visa_index.faiss")

with open(f"{BASE_PATH}/visa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


# EMBEDDING FUNCTION
def embed_text(text):
    return embedder.encode([text])[0].astype("float32")


# RETRIEVAL FUNCTION
def retrieve_chunks(query, k=5):
    vec = embed_text(query)
    _, ids = index.search(np.array([vec]), k)
    return [metadata[cid] for cid in ids[0] if cid != -1]


# LLM CALL
def ask_groq(question, chunks):
    ctx = "\n\n".join(
        f"[CHUNK {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""
Answer the question ONLY using the context below.

Question:
{question}

Context:
{ctx}

Return EXACTLY:

Eligibility: Yes / No / Partial
Final Answer: (2–3 lines)
Explanation:
- bullet points only
- do NOT mention chunk IDs
Confidence: (0 to 1)
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return resp.choices[0].message.content


# MAIN
if __name__ == "__main__":
    question = input("Enter your visa question: ")

    chunks = retrieve_chunks(question)
    model_answer = ask_groq(question, chunks)

    print("\n Response\n")
    print(model_answer)

    # LOGGING
    log = {
        "question": question,
        "chunks_used": [c["chunk_id"] for c in chunks],
        "model_answer": model_answer
    }

    with open("decision_history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, indent=4) + "\n\n")
