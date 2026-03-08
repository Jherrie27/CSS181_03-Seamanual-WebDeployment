from __future__ import annotations

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


# =============================================================================
# PATHS / CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PDF_PATH = DATA_DIR / "seaman.pdf"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_PATH = DATA_DIR / "faiss.index"
BM25_PATH = DATA_DIR / "bm25.pkl"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 80


# =============================================================================
# METADATA DETECTION
# =============================================================================

_META_PATTERNS = [
    (r"(?i)\bsection\s+(\d+[A-Za-z]?)\b", "Section {}"),
    (r"(?i)\bpart\s+(I{1,3}V?|VI{0,3}|\d+)\b", "Part {}"),
    (r"(?i)\b(medical benefit|sickness|treatment|injur)", "Medical Benefits"),
    (r"(?i)\b(death benefit|burial|deceased)", "Death & Burial Benefits"),
    (r"(?i)\b(disabilit)", "Disability Benefits"),
    (r"(?i)\b(repatri)", "Repatriation"),
    (r"(?i)\b(overtime|working hours)", "Working Hours & Overtime"),
    (r"(?i)\b(basic salary|basic wage|monthly salary)", "Wages & Salary"),
    (r"(?i)\b(allotment|remittance)", "Allotment & Remittance"),
    (r"(?i)\b(terminat|dismissal|disciplin)", "Termination & Discipline"),
    (r"(?i)\b(insurance|coverage|personal accident)", "Insurance"),
    (r"(?i)\b(placement fee|recruitment fee)", "Placement & Recruitment"),
    (r"(?i)\b(due process|right to be heard)", "Due Process"),
    (r"(?i)\b(beneficiar|qualified dependent)", "Beneficiaries"),
    (r"(?i)\b(obligation|employer shall|ship owner)", "Employer Obligations"),
]


def detect_section(text: str) -> str:
    probe = text[:200]
    for pattern, label in _META_PATTERNS:
        match = re.search(pattern, probe)
        if match:
            groups = match.groups()
            if "{}" in label and groups:
                return label.format(groups[0].strip().title())
            return label
    return "General"


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def clean_text(raw: str) -> str:
    raw = raw.replace("-\n", "")
    raw = raw.replace("\n", " ")
    raw = re.sub(r"Page \d+ of \d+", "", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def load_and_clean_pdf(pdf_path: Path) -> str:
    full_text = []

    # Primary: pdfplumber, column-aware
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                width, height = page.width, page.height
                bboxes = [
                    (0, 0, width / 2, height),
                    (width / 2, 0, width, height),
                ]
                for bbox in bboxes:
                    col = page.crop(bbox)
                    raw = col.extract_text()
                    if raw:
                        cleaned = clean_text(raw)
                        if cleaned:
                            full_text.append(cleaned)
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")

    if full_text:
        print(f"[OK] pdfplumber extracted {len(full_text)} column blocks.")
        return "\n\n".join(full_text)

    # Fallback: pypdf
    print("[INFO] Falling back to pypdf.")
    fallback = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                fallback.append(cleaned)
    except Exception as e:
        raise RuntimeError(f"Both pdfplumber and pypdf failed: {e}") from e

    if not fallback:
        raise RuntimeError("No text could be extracted from the PDF.")

    print(f"[OK] pypdf extracted {len(fallback)} pages.")
    return "\n\n".join(fallback)


# =============================================================================
# CHUNKING
# =============================================================================

def build_chunks(pdf_text: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = [
        chunk.strip()
        for chunk in splitter.split_text(pdf_text)
        if len(chunk.strip()) > MIN_CHUNK_LEN
    ]

    rows = []
    for i, chunk in enumerate(raw_chunks):
        rows.append(
            {
                "chunk_id": f"chunk_{i}",
                "section": detect_section(chunk),
                "text": chunk,
            }
        )
    return rows


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_faiss(chunks: list[dict], model_name: str) -> faiss.Index:
    print(f"[INFO] Loading embedder: {model_name}")
    embedder = SentenceTransformer(model_name)

    texts = [row["text"] for row in chunks]
    print("[INFO] Encoding chunk embeddings...")
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[OK] FAISS index built with {index.ntotal} vectors ({dim} dimensions).")
    return index


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    print("[INFO] Building BM25 index...")
    tokenized_corpus = [row["text"].lower().split() for row in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[OK] BM25 built with {len(tokenized_corpus)} documents.")
    return bm25


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Could not find PDF at: {PDF_PATH}\n"
            "Put your source PDF there and run again."
        )

    print(f"[INFO] Reading PDF: {PDF_PATH}")
    pdf_text = load_and_clean_pdf(PDF_PATH)

    print("[INFO] Building chunks...")
    chunks = build_chunks(pdf_text)

    if not chunks:
        raise RuntimeError("Chunking produced zero chunks.")

    print(f"[OK] Total characters: {len(pdf_text):,}")
    print(f"[OK] Total chunks    : {len(chunks)}")
    avg_len = sum(len(c["text"]) for c in chunks) // len(chunks)
    print(f"[OK] Avg chunk length: {avg_len} chars")
    print(f"[OK] Sections found  : {sorted(set(c['section'] for c in chunks))}")

    faiss_index = build_faiss(chunks, EMBED_MODEL)
    bm25 = build_bm25(chunks)

    print("[INFO] Saving assets...")
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(faiss_index, str(FAISS_PATH))
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("\n[SUCCESS] Asset preparation complete.")
    print(f"Saved: {CHUNKS_PATH}")
    print(f"Saved: {FAISS_PATH}")
    print(f"Saved: {BM25_PATH}")


if __name__ == "__main__":
    main()