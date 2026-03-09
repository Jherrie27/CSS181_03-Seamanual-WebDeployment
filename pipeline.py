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
    (r"(?i)\bchapter\s+(\d+)\b",                          "Chapter {}"),
    (r"(?i)\b(watch|watchstand)",                          "Watches"),
    (r"(?i)\b(marlinespike|splice|knot|hitch)",            "Marlinespike Seamanship"),
    (r"(?i)\b(anchor|mooring|line.handl|weather deck)",    "Deck Seamanship"),
    (r"(?i)\b(lifeboat|boat seamanship|davit)",            "Boat Seamanship"),
    (r"(?i)\b(ammunition|gunnery|projectile|caliber)",     "Ammunition & Gunnery"),
    (r"(?i)\b(navigation|colregs|lateral mark|buoy)",      "Navigation Rules"),
    (r"(?i)\b(sound.powered|telephone talker)",            "Watchstanders Equipment"),
    (r"(?i)\bsection\s+(\d+[A-Za-z]?)\b",                 "Section {}"),
    (r"(?i)\bpart\s+(I{1,3}V?|VI{0,3}|\d+)\b",            "Part {}"),
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
    r"\bwatch\b":                      "watch watchstander lookout duty post officer deck",
    r"\bknot\b|\bhitch\b":             "knot hitch clove bowline splice marlinespike line",
    r"\bsplice\b":                     "splice eye splice nylon manila strand tuck",
    r"\banchor\b":                     "anchor letting go brake chain fathom shot",
    r"\blifeboat\b":                   "lifeboat boat seamanship davit handling launching",
    r"\bprojectile\b|\bammunition\b":  "projectile ammunition caliber gun bore gunnery",
    r"\bnavigation\b":                 "navigation colregs rules lights buoy lateral",
    r"\bdeck\b":                       "deck seamanship mooring line handling weather",
    r"\bsound.powered\b":              "sound powered telephone talker communication",
    r"\bhelm\b|\bhelmsman\b":          "helm helmsman steering conning officer",
    r"\bfiring\b|\bgunner\w*":         "firing gunnery commence gun control authorized",
    r"\blookout\b":                    "lookout watch visual signal report sighting",
    r"\bmarlinespike\b":               "marlinespike seamanship knot line rope splice hitch",
    r"\bboat\b":                       "boat lifeboat davit seamanship handling crew",
    r"\bsignal\b":                     "signal visual flag semaphore communication report",
}

_MQ_PROMPT = (
    "Generate exactly 3 different search queries to retrieve relevant passages from the "
    "US Navy Seaman Training Manual (NAVEDTRA 14067). Use formal nautical/military language. "
    "Output only the 3 queries, one per line, no numbering or bullets.\n\nQuestion: {q}"
)
_MQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert US Navy seamanship instructor specializing in the "
    "NAVEDTRA 14067 Seaman training manual.\n"
    "Each context block begins with [Source: <section>]. Cite the section in your answer.\n"
    "RULES:\n"
    "1. Answer ONLY using information explicitly stated in the provided context.\n"
    "2. NEVER say: 'I cannot find', 'not mentioned', 'context does not contain', "
    "   'not in context', 'not provided'. These phrases are forbidden.\n"
    "3. Cite the source section (e.g. 'Under Watches...') when visible.\n"
    "4. If the context gives partial information, use it to construct the best possible answer."
)

STRICT_PROMPT = (
    "You are a strict NAVEDTRA 14067 instructor. "
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
        f"What does the Navy Seaman manual state regarding {kw}?",
        f"According to NAVEDTRA 14067, what are the procedures for {kw}?",
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
            {"role": "user", "content": f"Context from NAVEDTRA 14067:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"},
        ],
        max_tokens=300, temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ── Eval metrics ──────────────────────────────────────────────────────────────

def _compute_recall_at_k(chunk_rows, keywords):
    if not keywords: return 1.0
    combined = " ".join(r["text"] for r in chunk_rows).lower()
    return round(sum(1 for kw in keywords if kw.lower() in combined) / len(keywords), 4)

def _compute_precision_at_k(chunk_rows, keywords):
    if not chunk_rows or not keywords: return 0.0
    relevant = sum(1 for r in chunk_rows if any(kw.lower() in r["text"].lower() for kw in keywords))
    return round(relevant / len(chunk_rows), 4)

def _compute_context_relevancy(query, chunk_rows):
    if not chunk_rows: return 0.0
    embedder = load_embedder()
    q_emb  = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    c_embs = embedder.encode([r["text"] for r in chunk_rows], normalize_embeddings=True, convert_to_numpy=True)
    return round(float(np.mean([_cos(q_emb, c) for c in c_embs])), 4)

def _compute_answer_relevancy(query, answer):
    if not answer.strip(): return 0.0
    embedder = load_embedder()
    q_emb = embedder.encode([query],  normalize_embeddings=True, convert_to_numpy=True)[0]
    a_emb = embedder.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
    return round(_cos(q_emb, a_emb), 4)

def _compute_rouge_l(hypothesis, reference=""):
    if not hypothesis.strip() or not reference.strip(): return 0.0
    try:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        return round(scorer.score(reference, hypothesis)["rougeL"].fmeasure, 4)
    except Exception:
        return 0.0

def _compute_bert_score(answer, reference=""):
    if not answer.strip() or not reference.strip(): return 0.0
    try:
        from bert_score import score as bs_fn
        _, _, F1 = bs_fn([answer], [reference], model_type="distilbert-base-uncased",
                         lang="en", verbose=False)
        return round(float(F1[0]), 4)
    except Exception:
        return 0.0

# ── Generate answer ───────────────────────────────────────────────────────────

def generate_answer(query: str, keywords: list[str] | None = None,
                    reference: str = "") -> dict[str, Any]:
    if not query or not query.strip():
        return {"answer": "Please enter a valid question.", "context": "", "chunks": [],
                "latency_sec": 0.0, "confidence": 0.0, "metrics": {}, "query": query}

    t_start = time.time()
    try:
        context, chunk_rows, confidence = retrieve_context(query)
    except Exception as e:
        return {"answer": f"Retrieval error: {e}", "context": "", "chunks": [],
                "latency_sec": 0.0, "confidence": 0.0, "metrics": {}, "query": query}

    try:
        answer = _call_model(query, context, SYSTEM_PROMPT)
    except Exception as e:
        return {"answer": f"Generation error: {e}", "context": context, "chunks": chunk_rows,
                "latency_sec": 0.0, "confidence": confidence, "metrics": {}, "query": query}

    faith_orig = _faith_inline(answer, chunk_rows)
    if faith_orig < 0.35:
        try:
            alt = _call_model(query, context, STRICT_PROMPT)
            if _faith_inline(alt, chunk_rows) >= faith_orig + 0.05:
                answer = alt
        except Exception:
            pass

    latency = round(time.time() - t_start, 2)

    # Auto-extract keywords from query if not provided
    kws = keywords or [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)
                       if w.lower() not in _STOPWORDS][:6]

    metrics = {
        "recall_at_k":       _compute_recall_at_k(chunk_rows, kws),
        "precision_at_k":    _compute_precision_at_k(chunk_rows, kws),
        "context_relevancy": _compute_context_relevancy(query, chunk_rows),
        "faithfulness":      round(_faith_inline(answer, chunk_rows), 4),
        "answer_relevancy":  _compute_answer_relevancy(query, answer),
        "rouge_l":           _compute_rouge_l(answer, reference),
        "bert_score":        _compute_bert_score(answer, reference),
    }

    return {
        "query":       query,
        "answer":      answer,
        "context":     context,
        "chunks":      chunk_rows,
        "latency_sec": latency,
        "confidence":  round(float(confidence), 4),
        "faithfulness_estimate": metrics["faithfulness"],  # kept for backward compat
        "metrics":     metrics,
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
    test = generate_answer("What are the responsibilities of a seaman standing watch?")
    print(test["answer"])
    print(test["metrics"])
