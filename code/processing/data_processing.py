import os
import json
from docx import Document
from processing.chunking import chunk_georgian_civil_code
from processing.text_processing import clean_noise

CHUNKS_FILE = "./data/chunks.json"
DOCX_FILE = "./data/document.docx"


def process_and_save_chunks():
    """Parse DOCX, გაასუფთავე, დაყავი, და შეინახე JSON ფორმატში."""
    print("🔄 Processing DOCX and creating chunks...")
    document = Document(DOCX_FILE)
    cleaned_paragraphs = clean_noise(document)
    document_text = "\n".join(cleaned_paragraphs)
    chunks = chunk_georgian_civil_code(document_text)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_FILE}")
    return chunks


def load_chunks():
    """თუ JSON უკვე არსებობს მანდედან წამოიღოს, თუ არა და დაამუშავოს თავიდან DOCX."""
    if not os.path.exists(CHUNKS_FILE):
        return process_and_save_chunks()

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


