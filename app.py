import streamlit as st
import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re


# ------------------------ CONFIG ------------------------

st.set_page_config(
    page_title="Visa Eligibility Checker",
    page_icon="🛂",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    padding-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 30px;
}

.result-card {
    background-color: #111111;
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
    border: 1px solid #333333;
}

.chat-box {
    background-color: #1b1b1b;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 12px;
    border: 1px solid #333;
}

.input-label {
    font-size: 18px;
    font-weight: 600;
}

.conf-meter {
    width: 100%;
    height: 14px;
    border-radius: 6px;
    background-color: #333;
    margin-top: 10px;
}

.conf-fill {
    height: 14px;
    border-radius: 6px;
    background-color: #4caf50;
}

</style>
""", unsafe_allow_html=True)

# ------------------------ ENV + PATHS ------------------------

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

ROOT = "."
M1 = os.path.join(ROOT, "Aayush_milestone_1")
INDEX_PATH = os.path.join(M1, "outputs", "visa_index.faiss")
METADATA_PATH = os.path.join(M1, "outputs", "visa_metadata.json")

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

embedder = SentenceTransformer(EMBED_MODEL)

# ------------------------ FUNCTIONS ------------------------

def embed_text(text):
    return embedder.encode([text])[0].astype("float32")

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta

def retrieve(index, meta, query, k=5):
    qv = embed_text(query)
    D, I = index.search(np.array([qv]), k)
    results = [meta[cid] for cid in I[0] if 0 <= cid < len(meta)]
    return results

def ask_groq(question, chunks):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    ctx = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""
Answer ONLY using the context.

Question:
{question}

Context:
{ctx}

Return EXACTLY:

Eligibility: Yes / No / Partial
Final Answer: (2–3 lines)
Explanation:
- 3 to 5 bullet points
Confidence: (0 to 1)
"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content

def extract_confidence(ans_text):
    m = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", ans_text)
    return float(m.group(1)) if m else 0.0

def extract_country(chunks):
    """
    Tries to guess country from first retrieved chunk.
    Looks for words like Canada, USA, UK, Ireland.
    """
    if not chunks:
        return None

    text = chunks[0]["text"].lower()
    if "canada" in text: return "Canada"
    if "united states" in text or "usa" in text: return "United States"
    if "united kingdom" in text or "uk" in text: return "United Kingdom"
    if "ireland" in text: return "Ireland"
    return None

def country_flag(country):
    flags = {
        "Canada": "🇨🇦",
        "United States": "🇺🇸",
        "United Kingdom": "🇬🇧",
        "Ireland": "🇮🇪"
    }
    return flags.get(country, "🌍")



# ------------------------ UI ------------------------

st.markdown('<div class="main-title">Visa Eligibility Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask any visa-related question based on your PDF dataset.</div>', unsafe_allow_html=True)

st.markdown('<div class="input-label">Your Question</div>', unsafe_allow_html=True)
question = st.text_input(" ", placeholder="Type your visa question...")

ask_btn = st.button("Check Visa Eligibility")

if ask_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        index, meta = load_index()
        chunks = retrieve(index, meta, question)

        country = extract_country(chunks)
        country_icon = country_flag(country) if country else "🌍"

        if not GROQ_KEY:
            st.error("Missing GROQ_API_KEY in .env file.")
        else:
            answer = ask_groq(question, chunks)
            conf = extract_confidence(answer)

            # Chat style UI
            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            st.markdown(f"**You:** {question}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            st.markdown(f"**{country_icon} Visa Officer:**\n\n{answer}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Confidence meter
            st.markdown("### Confidence Level")
            st.markdown(f"**{conf * 100:.1f}%**")

            st.markdown(f"""
            <div class="conf-meter">
                <div class="conf-fill" style="width: {conf * 100}%"></div>
            </div>
            """, unsafe_allow_html=True)
