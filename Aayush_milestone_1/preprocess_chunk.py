import os
import re
import json
from PyPDF2 import PdfReader

# Input PDF folder and output locations
PDF_FOLDER = "pdfs"
CHUNK_FOLDER = "outputs/chunks"
JSON_FOLDER = "outputs/json"

os.makedirs(CHUNK_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                full_text += "\n" + text
        except:
            pass
    return full_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_all_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print(f"Folder '{PDF_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for pdf in pdf_files:
        print(f"Processing: {pdf}")
        pdf_path = os.path.join(PDF_FOLDER, pdf)

        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned)

        base_name = os.path.splitext(pdf)[0]
        pdf_chunk_dir = os.path.join(CHUNK_FOLDER, base_name)
        os.makedirs(pdf_chunk_dir, exist_ok=True)

        json_output = []

        for i, chunk in enumerate(chunks, start=1):
            txt_file = os.path.join(pdf_chunk_dir, f"{base_name}_chunk_{i}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(chunk)

            json_output.append({
                "pdf_name": pdf,
                "chunk_id": i,
                "text": chunk
            })

        json_path = os.path.join(JSON_FOLDER, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_output, jf, indent=4)

        print(f"Done: {pdf} → Chunks: {len(chunks)}")

if __name__ == "__main__":
    process_all_pdfs()
