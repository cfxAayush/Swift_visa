import streamlit as st
import os
import json
import numpy as np
import faiss
import re
import html
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SwiftVisa – Visa Screening Agent",
    page_icon="🛂",
    layout="wide"
)

# ---------------- ENV ----------------
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_KEY = os.getenv("GROQ_API_KEY")

# ROOT = "."
M1 = os.path.join(PROJECT_ROOT, "Aayush_milestone_1")
INDEX_PATH = os.path.join(M1, "outputs", "visa_index.faiss")
METADATA_PATH = os.path.join(M1, "outputs", "visa_metadata.json")


embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "selected_case" not in st.session_state:
    st.session_state.selected_case = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ---------------- LIGHT PROFESSIONAL THEME ----------------
st.markdown("""
<style>
/* Main background with subtle gradient */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 50%, #f0f9ff 100%) !important;
    position: relative;
    min-height: 100vh;
}

/* Ensure main content is visible */
.main .block-container {
    position: relative;
    z-index: 10;
    padding-top: 2rem;
}

/* Subtle pattern overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.03;
    background-image: 
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(59, 130, 246, 0.1) 2px, rgba(59, 130, 246, 0.1) 4px),
        repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(59, 130, 246, 0.1) 2px, rgba(59, 130, 246, 0.1) 4px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

body {
    color: #1e293b;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.big-title {
    font-size: 32px;
    font-weight: 800;
    color: #1e40af;
    position: relative;
    z-index: 1;
    margin-bottom: 8px;
}

.subtitle {
    color: #64748b;
    margin-bottom: 24px;
    position: relative;
    z-index: 1;
    font-size: 15px;
}

.section {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    position: relative;
    z-index: 1;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.08);
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
    color: #2563eb;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 28px;
    border-radius: 16px;
    margin-top: 24px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    position: relative;
    z-index: 1;
    box-shadow: 0 8px 16px rgba(59, 130, 246, 0.12);
}

.nav-button {
    background: transparent;
    color: #475569;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    margin: 0 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.nav-button:hover {
    background: rgba(59, 130, 246, 0.1);
    color: #2563eb;
}

.nav-button.active {
    background: #2563eb;
    color: white;
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    color: white;
    padding: 12px 28px;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
}

.conf-bar {
    height: 12px;
    background: rgba(226, 232, 240, 0.8);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.conf-fill {
    height: 12px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
}

.history-item {
    padding: 12px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.7);
    margin-bottom: 8px;
    cursor: pointer;
    border: 1px solid rgba(59, 130, 246, 0.15);
    transition: all 0.3s ease;
    color: #334155;
}

.history-item:hover {
    background: rgba(255, 255, 255, 0.95);
    border-color: rgba(59, 130, 246, 0.4);
    transform: translateX(4px);
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.small {
    font-size: 13px;
    color: #64748b;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(248, 250, 252, 0.98) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(59, 130, 246, 0.2);
}

section[data-testid="stSidebar"] h2 {
    color: #1e40af;
}

/* Input fields - Dark style like the screenshot */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.stSelectbox > div > div > select {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(59, 130, 246, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Text area for message */
.stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
}

/* Radio buttons */
.stRadio > label {
    color: #334155 !important;
}

.stRadio > div {
    background: #1e293b !important;
    padding: 10px !important;
    border-radius: 8px !important;
}

.stRadio > div label {
    color: #e5e7eb !important;
}

/* Labels */
label {
    color: #475569 !important;
    font-weight: 500 !important;
}

/* Hide code blocks in contact info */
code {
    display: none !important;
}

pre {
    display: none !important;
}

/* About page styling */
.about-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 16px;
    margin: 20px 0;
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: 0 8px 16px rgba(59, 130, 246, 0.12);
}

.feature-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%);
    padding: 20px;
    border-radius: 12px;
    margin: 15px 0;
    border-left: 4px solid #3b82f6;
}

.contact-info {
    background: rgba(255, 255, 255, 0.9);
    padding: 25px;
    border-radius: 12px;
    margin: 15px 0;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.icon-text {
    display: flex;
    align-items: center;
    margin: 12px 0;
    color: #334155;
}

.icon-text span {
    margin-right: 12px;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------
def embed_text(text):
    return embedder.encode([text])[0].astype("float32")

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta

def retrieve(idx, meta, query, k=5):
    qv = embed_text(query)
    _, I = idx.search(np.array([qv]), k)
    return [meta[i] for i in I[0] if 0 <= i < len(meta)]

def ask_groq(q, chunks, applicant_name):
    if not GROQ_KEY:
        return """Eligibility: Partial
Final Answer: Missing or unclear information.
Explanation:
- Financial proof insufficient
- Age-related conditions may apply
- Travel purpose unclear
Confidence: 0.45"""

    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    ctx = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""
Applicant: {applicant_name}

Answer ONLY using the context provided.

Question:
{q}

Context:
{ctx}

Return EXACTLY:

Eligibility: Yes / No / Partial
Final Answer: (2–3 lines)
Explanation:
- 3 to 5 bullet points
Confidence: (0 to 1)
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content.strip()

def extract_conf(text):
    m = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text)
    return float(m.group(1)) if m else 0.0

# ---------------- NAVIGATION ----------------
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

with col1:
    st.markdown("<div class='big-title'>🛂 SwiftVisa</div>", unsafe_allow_html=True)

with col2:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.current_page = "Home"

with col3:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.current_page = "About"

with col4:
    if st.button("📧 Contact", key="nav_contact", use_container_width=True):
        st.session_state.current_page = "Contact"

st.markdown("---")

# ---------------- SIDEBAR (HISTORY) ----------------
with st.sidebar:
    st.markdown("📋 Application History")
    st.markdown("---")

    if not st.session_state.history:
        st.markdown("<div class='small'>No cases yet</div>", unsafe_allow_html=True)

    for i, case in enumerate(reversed(st.session_state.history)):
        if st.button(f"👤 {case['name']} • {case['time']}", key=f"hist_{i}"):
            st.session_state.selected_case = case
            st.session_state.current_page = "Home"

# ============================================
# HOME PAGE
# ============================================
if st.session_state.current_page == "Home":
    st.markdown("<div class='subtitle'>AI-Powered Visa Eligibility Assessment</div>", unsafe_allow_html=True)

    # ---------------- FORM ----------------
    with st.form("visa_form"):
        col1, col2 = st.columns(2)

        with col1:
            # st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
            applicant_name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=100, step=1)
            nationality = st.selectbox("Nationality", ["India", "USA", "Canada", "UK", "Other"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">✈️ Trip Info</div>', unsafe_allow_html=True)
            purpose = st.selectbox("Purpose of Travel", ["Tourism", "Study", "Work", "Business" , "other"])
            visa_rejection = st.radio("Previous Visa Rejection?", ["No", "Yes" , "First time "])
            st.markdown('</div>', unsafe_allow_html=True)

        question = st.text_input(
            "Visa Question",
            placeholder="e.g. I am 17 years old with ₹20,000 savings applying for US tourist visa"
        )

        submit = st.form_submit_button("🔍 Analyze Visa Eligibility")

    # ---------------- PROCESS ----------------
    if submit:
        idx, meta = load_index()

        enriched_query = f"""
        Name: {applicant_name}
        Age: {age}
        Nationality: {nationality}
        Purpose: {purpose}
        Previous Rejection: {visa_rejection}

        Question:
        {question}
        """

        chunks = retrieve(idx, meta, enriched_query)
        answer = ask_groq(enriched_query, chunks, applicant_name)
        conf = extract_conf(answer)

        case = {
            "name": applicant_name,
            "time": datetime.now().strftime("%H:%M"),
            "answer": answer,
            "confidence": conf,
            "chunks": chunks
        }

        st.session_state.history.append(case)
        st.session_state.selected_case = case

    # ---------------- DISPLAY SELECTED CASE ----------------
    if st.session_state.selected_case:
        case = st.session_state.selected_case

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"📊 Assessment Results for {case['name']}")

        # Parse and format the answer
        answer_lines = case['answer'].split('\n')
        formatted_answer = '<div style="background: #ffffff; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #e5e7eb;">'
        
        for line in answer_lines:
            line = line.strip()
            if line.startswith('Eligibility:'):
                formatted_answer += f'<p style="color: #1e40af; font-weight: 700; font-size: 17px; margin: 10px 0;">{html.escape(line)}</p>'
            elif line.startswith('Final Answer:'):
                formatted_answer += f'<p style="color: #0f172a; font-weight: 600; margin: 15px 0 8px 0; font-size: 15px;">{html.escape(line)}</p>'
            elif line.startswith('Explanation:'):
                formatted_answer += f'<p style="color: #0f172a; font-weight: 600; margin: 15px 0 8px 0; font-size: 15px;">{html.escape(line)}</p>'
            elif line.startswith('Confidence:'):
                # Skip this as we show it separately
                
                continue
            elif line.startswith('-'):
                formatted_answer += f'<p style="color: #334155; margin: 6px 0 6px 20px; line-height: 1.6;">{html.escape(line)}</p>'
            elif line:
                formatted_answer += f'<p style="color: #334155; margin: 6px 0; line-height: 1.6;">{html.escape(line)}</p>'
        
        formatted_answer += '</div>'
        st.markdown(formatted_answer, unsafe_allow_html=True)

        st.markdown("Confidence Score")
        st.markdown(f"""
        <div class="conf-bar">
            <div class="conf-fill" style="width:{case['confidence']*100}%"></div>
        </div>
        <p style='color: #000080; font-size: 14px; margin-top: 8px;'>{case['confidence']*100:.1f}% Confidence</p>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 View Retrieved Policy Documents"):
            st.markdown('<div style="background: #ffffff; padding: 15px; border-radius: 8px;">', unsafe_allow_html=True)
            for i, c in enumerate(case["chunks"], 1):
                st.markdown(f'<p style="color: #1e40af; font-weight: 600; margin: 10px 0;">Document {i}:</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: #334155; line-height: 1.6; margin: 5px 0 15px 0;">{html.escape(c["text"])}</p>', unsafe_allow_html=True)
                if i < len(case["chunks"]):
                    st.markdown('<hr style="border: none; border-top: 1px solid #e5e7eb; margin: 15px 0;">', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ABOUT PAGE
# ============================================
elif st.session_state.current_page == "About":
    st.markdown("<div class='big-title'>About SwiftVisa</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-Powered Visa Screening Intelligence</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
        <h2 style='color: #1e40af; margin-bottom: 20px;'>🤖 What is SwiftVisa?</h2>
        <p style='font-size: 16px; line-height: 1.8; color: #334155;'>
            SwiftVisa is an advanced AI-powered visa screening agent that provides instant, accurate visa eligibility 
            assessments based on official policy documents. Our intelligent system analyzes your application details 
            against comprehensive visa regulations to determine your eligibility and provide personalized guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
        <h2 style='color: #1e40af; margin-bottom: 20px;'>⚡ How It Works</h2>
        <div class='feature-box'>
            <h3 style='color: #2563eb; margin-bottom: 10px;'>📚 Policy Document Analysis</h3>
            <p style='color: #475569; line-height: 1.6;'>
                SwiftVisa uses advanced natural language processing to search through thousands of official visa 
                policy documents, regulations, and guidelines from various countries and visa categories.
            </p>
        </div>
        
        <div class='feature-box'>
            <h3 style='color: #2563eb; margin-bottom: 10px;'>🎯 Intelligent Matching</h3>
            <p style='color: #475569; line-height: 1.6;'>
                Our AI engine uses semantic search and vector embeddings to find the most relevant policy sections 
                that apply to your specific situation, ensuring accurate and contextual assessments.
            </p>
        </div>
        
        <div class='feature-box'>
            <h3 style='color: #2563eb; margin-bottom: 10px;'>✅ Eligibility Assessment</h3>
            <p style='color: #475569; line-height: 1.6;'>
                Based on retrieved policy documents, SwiftVisa provides a clear eligibility determination (Yes/No/Partial) 
                with detailed explanations, confidence scores, and specific requirements you need to meet.
            </p>
        </div>
        
        <div class='feature-box'>
            <h3 style='color: #2563eb; margin-bottom: 10px;'>📊 Confidence Scoring</h3>
            <p style='color: #475569; line-height: 1.6;'>
                Each assessment comes with a confidence score that reflects how well your application matches the 
                policy requirements, helping you understand the strength of your case.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
        <h2 style='color: #1e40af; margin-bottom: 20px;'>🌟 Key Features</h2>
        <ul style='font-size: 15px; line-height: 2; color: #334155;'>
            <li><strong>Real-time Analysis:</strong> Instant visa eligibility assessments</li>
            <li><strong>Document-Based:</strong> All answers derived from official policy documents</li>
            <li><strong>Multi-Applicant Support:</strong> Track multiple visa applications simultaneously</li>
            <li><strong>History Tracking:</strong> Review past assessments anytime from the sidebar</li>
            <li><strong>Transparent Results:</strong> View the exact policy documents used for your assessment</li>
            <li><strong>Confidence Metrics:</strong> Understand the reliability of each assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
        <h2 style='color: #1e40af; margin-bottom: 20px;'>⚠️ Important Disclaimer</h2>
        <p style='font-size: 15px; line-height: 1.8; color: #334155;'>
            SwiftVisa is an AI-powered tool designed to provide preliminary visa eligibility guidance based on 
            available policy documents. While we strive for accuracy, this tool should not replace professional 
            immigration advice or official visa application processes. Always consult with official embassy resources 
            or immigration lawyers for final decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# CONTACT PAGE
# ============================================
elif st.session_state.current_page == "Contact":
    st.markdown("<div class='big-title'>Get In Touch</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>We'd love to hear from you</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class='about-card'>
            <h2 style='color: #1e40af; margin-bottom: 20px;'>📬 Contact Information</h2>
            
            <div class='contact-info'>
                <div style='margin: 20px 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 24px; margin-right: 15px;'>📧</span>
                        <div>
                            <strong style='color: #1e40af; font-size: 16px;'>Email</strong>
                        </div>
                    </div>
                    <div style='margin-left: 39px; color: #64748b;'>support@swiftvisa.com</div>
                </div>
            </div>
            
            <div class='contact-info'>
                <div style='margin: 20px 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 24px; margin-right: 15px;'>📞</span>
                        <div>
                            <strong style='color: #1e40af; font-size: 16px;'>Phone</strong>
                        </div>
                    </div>
                    <div style='margin-left: 39px; color: #64748b;'>+1 (555) 123-4567</div>
                </div>
            </div>
            
            <div class='contact-info'>
                <div style='margin: 20px 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 24px; margin-right: 15px;'>🏢</span>
                        <div>
                            <strong style='color: #1e40af; font-size: 16px;'>Address</strong>
                        </div>
                    </div>
                    <div style='margin-left: 39px; color: #64748b;'>123 Immigration Avenue<br>Suite 456, Global City, GC 78901</div>
                </div>
            </div>
            
            <div class='contact-info'>
                <div style='margin: 20px 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 24px; margin-right: 15px;'>🕐</span>
                        <div>
                            <strong style='color: #1e40af; font-size: 16px;'>Business Hours</strong>
                        </div>
                    </div>
                    <div style='margin-left: 39px; color: #64748b;'>Monday - Friday: 9:00 AM - 6:00 PM<br>Saturday: 10:00 AM - 4:00 PM</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='about-card'>
            <h2 style='color: #1e40af; margin-bottom: 20px;'>💬 Send Us a Message</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            subject = st.selectbox("Subject", [
                "General Inquiry",
                "Technical Support",
                "Feature Request",
                "Partnership Opportunity",
                "Other"
            ])
            message = st.text_area("Message", height=150)
            
            if st.form_submit_button("📤 Send Message"):
                st.success("✅ Thank you for your message! We'll get back to you within 24 hours.")

    st.markdown("""
    <div class='about-card' style='margin-top: 30px;'>
        <h2 style='color: #1e40af; margin-bottom: 20px;'>🔗 Connect With Us</h2>
        <p style='font-size: 15px; color: #334155; line-height: 1.8;'>
            Follow us on social media for updates, visa tips, and immigration news:
        </p>
        <div style='display: flex; gap: 20px; margin-top: 20px;'>
            <span style='font-size: 24px;'>🐦 Twitter</span>
            <span style='font-size: 24px;'>💼 LinkedIn</span>
            <span style='font-size: 24px;'>📘 Facebook</span>
            <span style='font-size: 24px;'>📸 Instagram</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div class='small' style='text-align: center; padding: 20px; color: #64748b;'>SwiftVisa © 2024 | Powered by AI | Made with ❤️ for global travelers</div>", unsafe_allow_html=True)