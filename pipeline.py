from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PDF_PATH    = DATA_DIR / "seaman.pdf"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_PATH  = DATA_DIR / "faiss.index"
BM25_PATH   = DATA_DIR / "bm25.pkl"

MODEL_LABEL        = "Llama-3.1-8B-Instant (Groq)"
MODEL_DISPLAY_NAME = "Llama-3.1-8B-Instant"
GROQ_MODEL         = "llama-3.1-8b-instant"

# ── Lightweight models that fit in Streamlit Cloud's 1 GB RAM ─────────────────
# BGE-Large (335M, 1024-dim) → all-MiniLM-L6-v2 (22M, 384-dim) ~15× smaller
# BGE-Reranker-v2-m3 (570M) → ms-marco-MiniLM-L-6-v2 (22M)    ~25× smaller
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_K_INIT  = 80
DEFAULT_K_FINAL = 5
MAX_RERANK_POOL = 120

# ── Groq client ───────────────────────────────────────────────────────────────

def _get_streamlit_secret(name):
    try:
        import streamlit as st
        if name in st.secrets:
            val = st.secrets[name]
            return str(val) if val else None
    except Exception:
        pass
    return None

def get_groq_api_key(explicit_key=None):
    api_key = explicit_key or os.environ.get("GROQ_API_KEY") or _get_streamlit_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in secrets/env.")
    return api_key

@lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=get_groq_api_key())

# ── Section detector ──────────────────────────────────────────────────────────

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

def _detect_section(text):
    probe = text[:200]
    for pattern, label in _META_PATTERNS:
        m = re.search(pattern, probe)
        if m:
            groups = m.groups()
            if "{}" in label and groups:
                return label.format(groups[0].strip().title())
            return label
    return "General"

def _clean_text(raw):
    raw = raw.replace("-\n", "").replace("\n", " ")
    raw = re.sub(r"Page \d+ of \d+", "", raw)
    return re.sub(r"[ \t]+", " ", raw).strip()

def _extract_pdf(pdf_path):
    import pdfplumber
    full_text = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                w, h = page.width, page.height
                for bbox in [(0, 0, w/2, h), (w/2, 0, w, h)]:
                    col = page.crop(bbox)
                    raw = col.extract_text()
                    if raw:
                        cleaned = _clean_text(raw)
                        if cleaned:
                            full_text.append(cleaned)
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")
    if full_text:
        return "\n\n".join(full_text)
    from pypdf import PdfReader
    fallback = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        if cleaned:
            fallback.append(cleaned)
    if not fallback:
        raise RuntimeError("No text extracted from seaman.pdf.")
    return "\n\n".join(fallback)

# ── Auto asset builder ────────────────────────────────────────────────────────

def _build_assets():
    try:
        import streamlit as st
        status = st.status("⚙️ First-run setup: indexing seaman.pdf…", expanded=True)
    except Exception:
        status = None

    def log(msg):
        print(msg)
        if status:
            try: status.write(msg)
            except Exception: pass

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"data/seaman.pdf not found. Commit it to your repo.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("📄 Extracting text from seaman.pdf…")
    pdf_text = _extract_pdf(PDF_PATH)

    log("✂️  Chunking text…")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = [c.strip() for c in splitter.split_text(pdf_text) if len(c.strip()) > 80]
    chunks = [
        {"chunk_id": f"chunk_{i}", "section": _detect_section(c), "text": c}
        for i, c in enumerate(raw_chunks)
    ]
    log(f"   → {len(chunks)} chunks")

    log(f"🔢 Building embeddings ({EMBED_MODEL})…")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [row["text"] for row in chunks]
    embeddings = embedder.encode(
        texts, batch_size=64, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log(f"   → FAISS: {index.ntotal} vectors × {dim} dims")

    log("📋 Building BM25 index…")
    tokenized = [row["text"].lower().split() for row in chunks]
    bm25 = BM25Okapi(tokenized)

    log("💾 Saving assets…")
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(FAISS_PATH))
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    log("✅ Done!")
    if status:
        try: status.update(label="✅ Index ready!", state="complete", expanded=False)
        except Exception: pass

def _ensure_assets():
    missing = [p for p in [CHUNKS_PATH, FAISS_PATH, BM25_PATH] if not p.exists()]
    if missing:
        print(f"[INFO] Missing: {[p.name for p in missing]} — building now…")
        _build_assets()

# ── Loaders (cached) ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_chunks():
    _ensure_assets()
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def load_chunk_texts():
    return [row["text"] for row in load_chunks()]

@lru_cache(maxsize=1)
def load_chunk_map():
    return {row["chunk_id"]: row for row in load_chunks()}

@lru_cache(maxsize=1)
def load_faiss_index():
    _ensure_assets()
    return faiss.read_index(str(FAISS_PATH))

@lru_cache(maxsize=1)
def load_bm25():
    _ensure_assets()
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def load_reranker():
    return CrossEncoder(RERANK_MODEL)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

_STOPWORDS = {
    "what","is","the","a","an","of","for","to","in","on","are","be","by","at",
    "as","if","or","and","how","who","does","do","can","was","were","will","with",
    "from","that","this","it","its","they","their","when","under","before","after",
    "must","shall","should","would","any","all","been","have","has","not","no",
    "which","about","into","also","upon","may","per","given","such","during",
    "give","get","set","let","put","take","make","use","than","then","each",
    "some","more","very","just","only","here","there","where","case","cases",
}

MARITIME_SYNONYMS = {
    r"\bcontract\b":                            "contract employment term duration period agreement",
    r"\bsalary\b|\bwage\b":                     "salary wage pay basic compensation monthly rate",
    r"\bmedical\b":                             "medical sickness injury treatment illness hospitalization medicine",
    r"\brepatri\w+":                            "repatriation repatriated return passage homeward cost expense transport",
    r"\bdeath\b|\bdi(ed|es)\b":                 "death deceased die died burial compensation beneficiary",
    r"\bdisciplin\w+":                          "disciplinary discipline offense misconduct penalty dismiss sanction violation",
    r"\bovertime\b":                            "overtime hours work additional pay rate beyond",
    r"\binsurance\b":                           "insurance coverage protection life accident personal",
    r"\bterminat\w+":                           "termination pre-termination dismissal grounds cause end",
    r"\bdisability\b|\bdisabled\b":             "disability disabled permanent total partial grade schedule injury",
    r"\ballotment\b":                           "allotment remittance family beneficiary percent percentage monthly mandatory",
    r"\bleave\b":                               "leave vacation rest days paid entitlement annual",
    r"\bplacement\b":                           "placement fee recruitment agency prohibited illegal banned",
    r"\bbenefit\w*":                            "benefit benefits compensation payment entitlement",
    r"\bseafarer\b":                            "seafarer crew member mariner employee worker",
    r"\bemployer\b":                            "employer owner company agency principal obligation shall",
    r"\bdocument\w*":                           "documents document copy given provided contract departure embarkation",
    r"\bdeparture\b":                           "departure embarkation sign-off prior departure leaving document copy",
    r"\bworking\s+hours\b|\bwork\s+hours\b|\bhours\b": "working hours eight regular daily standard per day",
    r"\bdue\s+process\b":                       "due process dismiss hearing explain opportunity informed",
    r"\b120\b":                                 "120 days hundred treatment disability assessment period beyond",
    r"\ballowanc\w+":                           "allowance subsistence vacation leave pay entitlement additional",
    r"\breimburs\w+":                           "reimbursement reimburse expense cost shoulder pay",
    r"\bliable\b|\bliability\b":                "liable liability responsible obligation penalty",
    r"\bbeneficiar\w+":                         "beneficiary beneficiaries qualified spouse children dependent family",
    r"\bsickness\b|\bill\w*\b":                 "sickness sick illness injury allowance treatment basic wage rate",
    r"\bfee\w*":                                "fee fees placement recruitment prohibited banned",
    r"\bobligat\w+":                            "obligation obligations employer shall must provide duty responsible",
    r"\bpercent\b|\b%\b":                       "percent percentage allotment mandatory minimum salary portion",
    r"\bfail\b|\bfails\b|\bfailed\b":           "fail fails failed failure employer repatriation liable responsible",
    r"\bshoulder\w*":                           "shoulder cost expense pay responsible employer repatriation",
    r"\bqualified\b":                           "qualified beneficiary spouse children dependent entitled",
    r"\bstandard\b":                            "standard regular normal working hours eight daily",
    r"\bcondit\w+":                             "conditions terms when may circumstances case grounds",
    r"\bground\w*":                             "grounds conditions cause reasons justification termination",
    r"\bobligation\w*":                         "obligation duty shall provide employer required",
}

_MQ_PROMPT = (
    "Generate exactly 3 different search queries to retrieve relevant clauses from the "
    "NAVEDTRA14067. Rephrase the question using formal legal language. "
    "Output only the 3 queries, one per line, no numbering or bullets.\n\nQuestion: {q}"
)
_MQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert maritime labor law assistant specializing in "
    "the NAVEDTRA14067.\n"
    "Each context block begins with [Source: <section>] indicating which part of the "
    "contract it comes from. Use this metadata to cite the correct section in your answer.\n"
    "RULES:\n"
    "1. Answer ONLY using information explicitly stated in the provided context.\n"
    "2. NEVER say: 'I cannot find', 'not mentioned', 'context does not contain', "
    "   'not in context', 'not provided'. These phrases are forbidden.\n"
    "3. Cite the source section (e.g. 'Under Medical Benefits...') when visible.\n"
    "4. If the context gives partial information, use it to construct the best possible answer."
)

STRICT_PROMPT = (
    "You are a strict NAVEDTRA14067 assistant. "
    "Use ONLY sentences directly verifiable in the provided context blocks. "
    "Each block starts with [Source: section]. Cite the section in your answer. "
    "No external knowledge. No speculation."
)

def _cos(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def _sigmoid(x):
    return float(1.0 / (1.0 + math.exp(-x)))

def _rrf_score(rank, k=60):
    return 1.0 / (k + rank)

def expand_query(query):
    expanded = query
    for pattern, expansion in MARITIME_SYNONYMS.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded += " " + expansion
    return expanded

def _multi_query(query):
    client = get_groq_client()
    try:
        resp = client.chat.completions.create(
            model=_MQ_MODEL,
            messages=[{"role": "user", "content": _MQ_PROMPT.format(q=query)}],
            max_tokens=120, temperature=0.4,
        )
        raw   = (resp.choices[0].message.content or "").strip()
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 10][:3]
        if lines:
            return [query] + lines
    except Exception:
        pass
    stems = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query) if w.lower() not in _STOPWORDS]
    kw = " ".join(stems[:6]) if stems else query
    return [
        query,
        f"What does the NAVEDTRA14067 state regarding {kw}?",
        f"Under the NAVEDTRA14067, what are the rules concerning {kw}?",
    ]

def retrieve_context(query, k_init=DEFAULT_K_INIT, k_final=DEFAULT_K_FINAL):
    chunks      = load_chunks()
    chunk_map   = load_chunk_map()
    chunk_texts = load_chunk_texts()
    bm25        = load_bm25()
    index       = load_faiss_index()
    embedder    = load_embedder()
    reranker    = load_reranker()

    queries  = _multi_query(query)
    expanded = expand_query(query)

    q_embs = embedder.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    faiss_ranks: dict = {}
    seen: set = set()
    search_k = min(k_init, len(chunk_texts))

    for qe in q_embs:
        _, ids = index.search(qe.reshape(1, -1), search_k)
        for rank, idx in enumerate(ids[0]):
            if 0 <= idx < len(chunks):
                cid = chunks[idx]["chunk_id"]
                if cid not in seen:
                    faiss_ranks[cid] = rank + 1
                    seen.add(cid)

    main_q_emb = q_embs[0]
    bm25_scores  = bm25.get_scores(expanded.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:search_k]
    bm25_ranks   = {chunks[i]["chunk_id"]: r+1 for r, i in enumerate(bm25_top_idx) if 0 <= i < len(chunks)}

    all_ids = set(faiss_ranks) | set(bm25_ranks)
    rrf = {cid: (_rrf_score(faiss_ranks.get(cid, k_init+1)) + _rrf_score(bm25_ranks.get(cid, k_init+1)))
           for cid in all_ids}
    pool_ids = sorted(rrf, key=rrf.get, reverse=True)[:k_init]

    query_stems = [w[:6].lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", query) if w.lower() not in _STOPWORDS]
    pool_set = set(pool_ids)
    boosted  = list(pool_ids)
    for row in chunks:
        if row["chunk_id"] not in pool_set and any(s in row["text"].lower() for s in query_stems):
            boosted.append(row["chunk_id"])
            pool_set.add(row["chunk_id"])

    pool_rows = [chunk_map[cid] for cid in boosted[:MAX_RERANK_POOL]]

    if len(pool_rows) > 1:
        ce_scores = reranker.predict([[query, row["text"]] for row in pool_rows])
        ranked    = sorted(zip(ce_scores, pool_rows), key=lambda x: x[0], reverse=True)
        top_sim   = _sigmoid(float(ranked[0][0]))
        ce_pool   = [row for _, row in ranked[:k_final*3]]
    else:
        ce_pool, top_sim = pool_rows[:k_final], 0.5

    if len(ce_pool) > k_final:
        c_embs  = embedder.encode([r["text"] for r in ce_pool], normalize_embeddings=True, convert_to_numpy=True)
        selected, remaining = [], list(range(len(ce_pool)))
        for _ in range(k_final):
            if not remaining: break
            if not selected:
                scores = [_cos(main_q_emb, c_embs[i]) for i in remaining]
            else:
                scores = [0.70*_cos(main_q_emb, c_embs[i]) - 0.30*max(_cos(c_embs[i], c_embs[j]) for j in selected)
                          for i in remaining]
            best = remaining[int(np.argmax(scores))]
            selected.append(best); remaining.remove(best)
        final_rows = [ce_pool[i] for i in selected]
    else:
        final_rows = ce_pool[:k_final]

    context_str = "\n\n---\n\n".join(f"[Source: {r.get('section','General')}]\n{r['text']}" for r in final_rows)
    return context_str, final_rows, top_sim

def _faith_inline(answer, chunk_rows):
    if not answer.strip() or not chunk_rows: return 0.0
    chunks   = [row["text"] for row in chunk_rows]
    embedder = load_embedder()
    sents    = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 12]
    c_embs   = embedder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
    if not sents:
        a_emb = embedder.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
        return float(max(_cos(a_emb, c) for c in c_embs))
    scores = []
    for sent in sents:
        s_emb = embedder.encode([sent], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores.append(max(_cos(s_emb, c) for c in c_embs))
    return float(np.mean(scores))

def _call_model(query, context, system_prompt):
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from official NAVEDTRA14067 documents:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"},
        ],
        max_tokens=300, temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

def generate_answer(query):
    if not query or not query.strip():
        return {"answer": "Please enter a valid question.", "context": "", "chunks": [],
                "latency_sec": 0.0, "confidence": 0.0, "query": query}
    t_start = time.time()
    try:
        context, chunk_rows, confidence = retrieve_context(query)
    except Exception as e:
        return {"answer": f"Retrieval error: {e}", "context": "", "chunks": [],
                "latency_sec": 0.0, "confidence": 0.0, "query": query}
    try:
        answer = _call_model(query, context, SYSTEM_PROMPT)
    except Exception as e:
        return {"answer": f"Generation error: {e}", "context": context, "chunks": chunk_rows,
                "latency_sec": 0.0, "confidence": confidence, "query": query}

    faith_orig = _faith_inline(answer, chunk_rows)
    if faith_orig < 0.35:
        try:
            alt = _call_model(query, context, STRICT_PROMPT)
            if _faith_inline(alt, chunk_rows) >= faith_orig + 0.05:
                answer = alt
        except Exception:
            pass

    latency = round(time.time() - t_start, 2)
    return {
        "query": query, "answer": answer, "context": context, "chunks": chunk_rows,
        "latency_sec": latency, "confidence": round(float(confidence), 4),
        "faithfulness_estimate": round(float(_faith_inline(answer, chunk_rows)), 4),
    }

# ── App support ───────────────────────────────────────────────────────────────

def warmup():
    _ensure_assets()
    _ = get_groq_client()
    _ = load_chunks()
    _ = load_bm25()
    _ = load_faiss_index()
    _ = load_embedder()
    _ = load_reranker()

def healthcheck():
    info = {
        "model_label": MODEL_LABEL, "groq_model": GROQ_MODEL,
        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
        "chunks_exists": CHUNKS_PATH.exists(),
        "faiss_exists":  FAISS_PATH.exists(),
        "bm25_exists":   BM25_PATH.exists(),
    }
    try:    info["num_chunks"]      = len(load_chunks())
    except Exception as e: info["chunks_error"] = str(e)
    try:    info["faiss_ntotal"]    = int(load_faiss_index().ntotal)
    except Exception as e: info["faiss_error"]  = str(e)
    try:    info["groq_key_loaded"] = bool(get_groq_api_key())
    except Exception as e: info["groq_error"]   = str(e)
    return info

if __name__ == "__main__":
    warmup()
    test = generate_answer("What is the maximum duration of a seafarer's employment contract?")
    print(test["answer"])
