# Swift Visa – Retrieval Augmented Visa Question Answering System

Swift Visa is a Retrieval-Augmented Generation (RAG) based system designed to answer visa-related questions strictly from official PDF documents.  
The system retrieves relevant information from indexed visa documents and generates grounded answers without hallucinating or using external knowledge.

---

## Problem Statement

Visa information is typically distributed across long and complex PDF documents, making it difficult to extract clear and accurate answers.  
This project solves that problem by combining semantic search with a large language model to provide precise, document-backed responses.

---

## System Architecture

1. Visa PDFs are converted into text chunks  
2. Text chunks are converted into vector embeddings  
3. Embeddings are stored in a FAISS vector index  
4. User queries are embedded and matched against the index  
5. Top relevant chunks are retrieved  
6. The LLM generates an answer using only retrieved content  

---

## Technology Stack

- Python
- FAISS (Vector Search)
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Groq LLM API (`llama-3.1-8b-instant`)
- NumPy
- python-dotenv

---


---

## How It Works

- User enters a visa-related question
- The question is converted into an embedding
- FAISS retrieves the top relevant document chunks
- Retrieved chunks are sent to the LLM as context
- The LLM generates a structured answer using only that context

---

## Output Format

The model returns:
- Final Answer
- Explanation with document chunk references
- Confidence score (0 to 1)

---

## Example Usage

```python
question = "What are the requirements for a USA visa?"
chunks = retrieve_chunks(question, k=5)
answer = ask_groq_from_pdf(question, chunks)
print(answer)

Disclaimer

This system does not provide legal advice.
All responses are generated solely from provided visa documents.


Author

Aayush Shrivastava