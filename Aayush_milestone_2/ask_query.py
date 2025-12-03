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
Do NOT add any information that is not present in the context.

Question:
{question}

Context:
{ctx}

Return EXACTLY the following format:

Eligibility: Yes / No / Partial
Final Answer: (2–3 lines summary ONLY)
Explanation:
- Provide EXACTLY 3 to 5 bullet points.
- Each bullet MUST be unique.
- No repeating, rephrasing, or expanding the same idea.
- DO NOT show chunk IDs or chunk numbers.
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

    # Extract confidence
    confidence = extract_confidence(model_answer)
    answer_text = model_answer.lower()

    if "eligibility: yes" in answer_text:
        confidence = min(0.9, confidence)
    elif "eligibility: no" in answer_text:
        confidence = min(0.6, confidence)
    else:
        confidence = 0.3

    # FIX: overwrite confidence inside the model's text
    import re
    model_answer = re.sub(r"Confidence:\s*\d+(\.\d+)?", f"Confidence: {confidence}", model_answer)

    print("\nResponse:\n")
    print(model_answer)

    # LOGGING
    log_entry = {
        "question": question,
        "model_answer": model_answer,
        "confidence": confidence
    }

    with open("decision_history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, indent=4) + "\n\n")

    # print("\nSaved to decision_history.json")
