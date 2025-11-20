
This project converts visa PDFs into searchable vector embeddings.  
Simple 3-step process.

---

## 1. PDF → Chunks

Extract text from PDFs and split into chunks.

python preprocess_chunk.py

yaml
Copy code

Outputs:
- `outputs/chunks/`
- `outputs/json/`

---

## 2. Chunks → Embeddings

Generate embeddings using SentenceTransformer.

python embeddings.py

yaml
Copy code

Output:
- `chunk_embeddings.json`

---

## 3. Embeddings → FAISS

Store embeddings in a FAISS vector database.

python faiss_store.py

yaml
Copy code

Outputs:
- `visa_index.faiss`
- `visa_metadata.json`

---

## Requirements

pip install sentence-transformers
pip install PyPDF2
pip install faiss-cpu

yaml
Copy code

---

## Pipeline Flow