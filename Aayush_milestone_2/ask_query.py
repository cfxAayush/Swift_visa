from groq import Groq
from dotenv import load_dotenv
import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# LOAD ENV + MODELS

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + metadata
index = faiss.read_index("../Aayush_milestone_1/outputs/visa_index.faiss")

with open("../Aayush_milestone_1/outputs/visa_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)



# EMBEDDING FUNCTION

def embed_text(text):
    return embedder.encode([text])[0].astype("float32")



# RETRIEVAL FUNCTION

def retrieve_chunks(query, k=5):
    vec = embed_text(query)
    _, ids = index.search(np.array([vec]), k)
    return [metadata[cid] for cid in ids[0] if cid != -1]



# EXTRACT CONFIDENCE

def extract_confidence(text):
    for line in text.split("\n"):
        if "Confidence:" in line:
            try:
                return float(line.split("Confidence:")[1].strip())
            except:
                return None
    return None



# LLM CALL

def ask_groq(question, chunks):
    # Build context with hidden chunk IDs
    ctx = "\n\n".join(
        f"[CHUNK {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""
You are a visa eligibility officer.
Answer ONLY using the PDF context provided.

Question:
{question}

Context:
{ctx}

Return EXACTLY this format:

Eligibility: Yes / No / Partial
Final Answer: (2–3 lines summary)
Explanation:
- Bullet points ONLY
- DO NOT show chunk IDs
- DO NOT show chunk numbers
Confidence: (0 to 1)
"""

    # Model call
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return resp.choices[0].message.content



# MAIN EXECUTION

if __name__ == "__main__":
    question = input("Enter your visa question: ")

    chunks = retrieve_chunks(question)
    model_answer = ask_groq(question, chunks)

    print("\nResponse:\n")
    print(model_answer)

    # Extract confidence
    confidence = extract_confidence(model_answer)

    
    # LOGGING
    
    log_entry = {
        "question": question,
        "chunks_used": [c["chunk_id"] for c in chunks],
        "model_answer": model_answer,
        "confidence": confidence
    }

    with open("decision_history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, indent=4) + "\n\n")

    # print("\nSaved to decision_history.json")
