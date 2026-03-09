"""
Run this ONCE on your local machine to generate the pre-built index files.
Uses BGE-Large (same as your notebook) so the deployed app matches exactly.

Requirements:
    pip install sentence-transformers faiss-cpu pdfplumber pypdf langchain-text-splitters rank-bm25 numpy

Usage:
    python build_assets.py

Output (commit all 3 to your GitHub repo under data/):
    data/chunks.json
    data/faiss.index
    data/bm25.pkl
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "seaman.pdf"

EMBED_MODEL   = "BAAI/bge-large-en-v1.5"   # ← matches your notebook exactly
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 80

# ── Section detector (same as notebook) ──────────────────────────────────────
_META_PATTERNS = [
    (r"(?i)\bsection\s+(\d+[A-Za-z]?)\b",               "Section {}"),
    (r"(?i)\bpart\s+(I{1,3}V?|VI{0,3}|\d+)\b",          "Part {}"),
    (r"(?i)\b(medical benefit|sickness|treatment|injur)", "Medical Benefits"),
    (r"(?i)\b(death benefit|burial|deceased)",            "Death & Burial Benefits"),
    (r"(?i)\b(disabilit)",                                "Disability Benefits"),
    (r"(?i)\b(repatri)",                                  "Repatriation"),
    (r"(?i)\b(overtime|working hours)",                   "Working Hours & Overtime"),
    (r"(?i)\b(basic salary|basic wage|monthly salary)",   "Wages & Salary"),
    (r"(?i)\b(allotment|remittance)",                     "Allotment & Remittance"),
    (r"(?i)\b(terminat|dismissal|disciplin)",             "Termination & Discipline"),
    (r"(?i)\b(insurance|coverage|personal accident)",     "Insurance"),
    (r"(?i)\b(placement fee|recruitment fee)",            "Placement & Recruitment"),
    (r"(?i)\b(due process|right to be heard)",            "Due Process"),
    (r"(?i)\b(beneficiar|qualified dependent)",           "Beneficiaries"),
    (r"(?i)\b(obligation|employer shall|ship owner)",     "Employer Obligations"),
]

def detect_section(text):
    probe = text[:200]
    for pattern, label in _META_PATTERNS:
        m = re.search(pattern, probe)
        if m:
            groups = m.groups()
            if "{}" in label and groups:
                return label.format(groups[0].strip().title())
            return label
    return "General"

def clean_text(raw):
    raw = raw.replace("-\n", "").replace("\n", " ")
    raw = re.sub(r"Page \d+ of \d+", "", raw)
    return re.sub(r"[ \t]+", " ", raw).strip()

def extract_pdf(path):
    full_text = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                w, h = page.width, page.height
                for bbox in [(0, 0, w/2, h), (w/2, 0, w, h)]:
                    col = page.crop(bbox)
                    raw = col.extract_text()
                    if raw:
                        cleaned = clean_text(raw)
                        if cleaned:
                            full_text.append(cleaned)
        if full_text:
            print(f"[OK] pdfplumber: {len(full_text)} column blocks")
            return "\n\n".join(full_text)
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")

    print("[INFO] Falling back to pypdf...")
    reader = PdfReader(str(path))
    fallback = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            fallback.append(cleaned)
    if not fallback:
        raise RuntimeError("No text extracted from PDF.")
    print(f"[OK] pypdf: {len(fallback)} pages")
    return "\n\n".join(fallback)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Put seaman.pdf in: {PDF_PATH}")

    print(f"\n{'='*60}")
    print(f"Building BGE-Large assets from: {PDF_PATH}")
    print(f"{'='*60}\n")

    # 1. Extract PDF
    print("Step 1/4 — Extracting PDF text...")
    pdf_text = extract_pdf(PDF_PATH)
    print(f"         {len(pdf_text):,} characters extracted\n")

    # 2. Chunk
    print("Step 2/4 — Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = [c.strip() for c in splitter.split_text(pdf_text) if len(c.strip()) > MIN_CHUNK_LEN]
    chunks = [
        {"chunk_id": f"chunk_{i}", "section": detect_section(c), "text": c}
        for i, c in enumerate(raw_chunks)
    ]
    avg = sum(len(c["text"]) for c in chunks) // len(chunks)
    sections = sorted(set(c["section"] for c in chunks))
    print(f"         {len(chunks)} chunks, avg {avg} chars")
    print(f"         Sections: {sections}\n")

    # 3. BGE-Large embeddings + FAISS
    print(f"Step 3/4 — Embedding with {EMBED_MODEL}...")
    print("          (downloading ~1.3 GB model on first run — be patient)\n")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\n         FAISS: {index.ntotal} vectors × {dim} dims\n")

    # 4. BM25
    print("Step 4/4 — Building BM25...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"         BM25: {len(tokenized)} documents\n")

    # Save
    print("Saving assets to data/...")
    chunks_path = DATA_DIR / "chunks.json"
    faiss_path  = DATA_DIR / "faiss.index"
    bm25_path   = DATA_DIR / "bm25.pkl"

    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(faiss_path))
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"\n{'='*60}")
    print("SUCCESS! Commit these 3 files to your GitHub repo:")
    print(f"  data/chunks.json  ({chunks_path.stat().st_size/1024:.0f} KB)")
    print(f"  data/faiss.index  ({faiss_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  data/bm25.pkl     ({bm25_path.stat().st_size/1024:.0f} KB)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
