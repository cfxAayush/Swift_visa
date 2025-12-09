# SwiftVisa RAG System  
A Retrieval-Augmented Generation (RAG) system that answers visa-related questions using real PDF documents.  
It supports USA, UK, Canada, Ireland, and Schengen visa guidelines.

This project uses:
- Sentence Transformers (MiniLM-L6-v2)
- FAISS vector indexing
- Groq LLM (Llama 3.1)
- Custom PDF chunking + metadata tracking

---

## 🌐 Features

### ✅ PDF → Chunks → Embeddings Pipeline
- PDFs are split into clean text chunks.
- Each chunk is embedded using MiniLM.
- FAISS index is generated for fast vector search.

### ✅ RAG Question Answering
- User asks any visa-related question.
- System retrieves the most relevant chunks.
- Groq LLM answers using ONLY the PDF context.
- Output includes:
  - Eligibility (Yes / No / Partial)
  - Final summarized answer
  - Clean explanation (no chunk IDs)
  - Confidence score

### ✅ Logging
Every query is saved in `decision_history.json`:
```json
{
    "question": "Schengen visa requirements?",
    "model_answer": "..."
  
}
