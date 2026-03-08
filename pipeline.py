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

# =============================================================================
# PATHS / CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_PATH = DATA_DIR / "faiss.index"
BM25_PATH = DATA_DIR / "bm25.pkl"

MODEL_LABEL = "Llama-3.1-8B-Instant (Groq)"
MODEL_DISPLAY_NAME = "Llama-3.1-8B-Instant"
GROQ_MODEL = "llama-3.1-8b-instant"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_K_INIT = 80
DEFAULT_K_FINAL = 5
MAX_RERANK_POOL = 120

# =============================================================================
# SECRET / CLIENT HELPERS
# =============================================================================

def _get_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            val = st.secrets[name]
            return str(val) if val else None
    except Exception:
        pass
    return None


def get_groq_api_key(explicit_key: str | None = None) -> str:
    api_key = (
        explicit_key
        or os.environ.get("GROQ_API_KEY")
        or _get_streamlit_secret("GROQ_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Put it in .streamlit/secrets.toml locally "
            "or Streamlit Community Cloud → App settings → Secrets."
        )
    return api_key


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())


# =============================================================================
# LOADERS
# =============================================================================

def _ensure_assets() -> None:
    missing = [str(p) for p in [CHUNKS_PATH, FAISS_PATH, BM25_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required runtime assets:\n- " + "\n- ".join(missing) +
            "\n\nRun prepare_assets.py first."
        )


@lru_cache(maxsize=1)
def load_chunks() -> list[dict[str, Any]]:
    _ensure_assets()
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_chunk_texts() -> list[str]:
    return [row["text"] for row in load_chunks()]


@lru_cache(maxsize=1)
def load_chunk_map() -> dict[str, dict[str, Any]]:
    return {row["chunk_id"]: row for row in load_chunks()}


@lru_cache(maxsize=1)
def load_faiss_index():
    _ensure_assets()
    return faiss.read_index(str(FAISS_PATH))


@lru_cache(maxsize=1)
def load_bm25() -> BM25Okapi:
    _ensure_assets()
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


# =============================================================================
# NOTEBOOK LOGIC ADAPTED
# =============================================================================

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
    r"\bcontract\b": "contract employment term duration period agreement",
    r"\bsalary\b|\bwage\b": "salary wage pay basic compensation monthly rate",
    r"\bmedical\b": "medical sickness injury treatment illness hospitalization medicine",
    r"\brepatri\w+": "repatriation repatriated return passage homeward cost expense transport",
    r"\bdeath\b|\bdi(ed|es)\b": "death deceased die died burial compensation beneficiary",
    r"\bdisciplin\w+": "disciplinary discipline offense misconduct penalty dismiss sanction violation",
    r"\bovertime\b": "overtime hours work additional pay rate beyond",
    r"\binsurance\b": "insurance coverage protection life accident personal",
    r"\bterminat\w+": "termination pre-termination dismissal grounds cause end",
    r"\bdisability\b|\bdisabled\b": "disability disabled permanent total partial grade schedule injury",
    r"\ballotment\b": "allotment remittance family beneficiary percent percentage monthly mandatory",
    r"\bleave\b": "leave vacation rest days paid entitlement annual",
    r"\bplacement\b": "placement fee recruitment agency prohibited illegal banned",
    r"\bbenefit\w*": "benefit benefits compensation payment entitlement",
    r"\bseafarer\b": "seafarer crew member mariner employee worker",
    r"\bemployer\b": "employer owner company agency principal obligation shall",
    r"\bdocument\w*": "documents document copy given provided contract departure embarkation",
    r"\bdeparture\b": "departure embarkation sign-off prior departure leaving document copy",
    r"\bworking\s+hours\b|\bwork\s+hours\b|\bhours\b": "working hours eight regular daily standard per day",
    r"\bdue\s+process\b": "due process dismiss hearing explain opportunity informed",
    r"\b120\b": "120 days hundred treatment disability assessment period beyond",
    r"\ballowanc\w+": "allowance subsistence vacation leave pay entitlement additional",
    r"\breimburs\w+": "reimbursement reimburse expense cost shoulder pay",
    r"\bliable\b|\bliability\b": "liable liability responsible obligation penalty",
    r"\bbeneficiar\w+": "beneficiary beneficiaries qualified spouse children dependent family",
    r"\bsickness\b|\bill\w*\b": "sickness sick illness injury allowance treatment basic wage rate",
    r"\bfee\w*": "fee fees placement recruitment prohibited banned",
    r"\bobligat\w+": "obligation obligations employer shall must provide duty responsible",
    r"\bpercent\b|\b%\b": "percent percentage allotment mandatory minimum salary portion",
    r"\bfail\b|\bfails\b|\bfailed\b": "fail fails failed failure employer repatriation liable responsible",
    r"\bshoulder\w*": "shoulder cost expense pay responsible employer repatriation",
    r"\bqualified\b": "qualified beneficiary spouse children dependent entitled",
    r"\bstandard\b": "standard regular normal working hours eight daily",
    r"\bcondit\w+": "conditions terms when may circumstances case grounds",
    r"\bground\w*": "grounds conditions cause reasons justification termination",
    r"\bobligation\w*": "obligation duty shall provide employer required",
}

_MQ_PROMPT = (
    "Generate exactly 3 different search queries to retrieve relevant clauses from the "
    "POEA Standard Employment Contract. Rephrase the question using formal legal language. "
    "Output only the 3 queries, one per line, no numbering or bullets.\n\nQuestion: {q}"
)
_MQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert Philippine maritime labor law assistant specializing in "
    "the POEA Standard Employment Contract (SEC) for Filipino seafarers.\n"
    "Each context block begins with [Source: <section>] indicating which part of the "
    "contract it comes from. Use this metadata to cite the correct section in your answer.\n"
    "RULES:\n"
    "1. Answer ONLY using information explicitly stated in the provided context.\n"
    "2. NEVER say: 'I cannot find', 'not mentioned', 'context does not contain', "
    "'not in context', 'not provided'. These phrases are forbidden.\n"
    "3. Cite the source section (e.g. 'Under Medical Benefits...') when visible.\n"
    "4. If the context gives partial information, use it to construct the best possible answer."
)

STRICT_PROMPT = (
    "You are a strict POEA SEC legal assistant. "
    "Use ONLY sentences directly verifiable in the provided context blocks. "
    "Each block starts with [Source: section]. Cite the section in your answer. "
    "No external knowledge. No speculation."
)


# =============================================================================
# MATH HELPERS
# =============================================================================

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


# =============================================================================
# QUERY HELPERS
# =============================================================================

def expand_query(query: str) -> str:
    expanded = query
    for pattern, expansion in MARITIME_SYNONYMS.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded += " " + expansion
    return expanded


def _multi_query(query: str) -> list[str]:
    client = get_groq_client()
    try:
        resp = client.chat.completions.create(
            model=_MQ_MODEL,
            messages=[{"role": "user", "content": _MQ_PROMPT.format(q=query)}],
            max_tokens=120,
            temperature=0.4,
        )
        raw = (resp.choices[0].message.content or "").strip()
        lines = [line.strip() for line in raw.split("\n") if len(line.strip()) > 10][:3]
        if lines:
            return [query] + lines
    except Exception:
        pass

    stems = [
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)
        if w.lower() not in _STOPWORDS
    ]
    kw = " ".join(stems[:6]) if stems else query
    return [
        query,
        f"What does the POEA Standard Employment Contract state regarding {kw}?",
        f"Under the POEA SEC, what are the rules concerning {kw}?",
    ]


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_context(
    query: str,
    k_init: int = DEFAULT_K_INIT,
    k_final: int = DEFAULT_K_FINAL,
) -> tuple[str, list[dict[str, Any]], float]:
    chunks = load_chunks()
    chunk_map = load_chunk_map()
    chunk_texts = load_chunk_texts()
    bm25 = load_bm25()
    index = load_faiss_index()
    embedder = load_embedder()
    reranker = load_reranker()

    queries = _multi_query(query)
    expanded = expand_query(query)

    # Dense retrieval: multi-query union
    q_embs = embedder.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss_ranks_union: dict[str, int] = {}
    seen_ids: set[str] = set()

    search_k = min(k_init, len(chunk_texts))
    for qe in q_embs:
        scores, ids = index.search(qe.reshape(1, -1), search_k)
        _ = scores
        for rank, idx in enumerate(ids[0]):
            if 0 <= idx < len(chunks):
                chunk_id = chunks[idx]["chunk_id"]
                if chunk_id not in seen_ids:
                    faiss_ranks_union[chunk_id] = rank + 1
                    seen_ids.add(chunk_id)

    main_q_emb = q_embs[0]

    # Sparse retrieval: BM25
    bm25_scores = bm25.get_scores(expanded.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:search_k]
    bm25_ranks = {
        chunks[i]["chunk_id"]: rank + 1
        for rank, i in enumerate(bm25_top_idx)
        if 0 <= i < len(chunks)
    }

    # RRF fusion
    all_ids = set(faiss_ranks_union) | set(bm25_ranks)
    rrf = {
        chunk_id: (
            _rrf_score(faiss_ranks_union.get(chunk_id, k_init + 1))
            + _rrf_score(bm25_ranks.get(chunk_id, k_init + 1))
        )
        for chunk_id in all_ids
    }
    pool_ids = sorted(rrf, key=rrf.get, reverse=True)[:k_init]

    # Stem keyword boost
    query_stems = [
        w[:6].lower()
        for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)
        if w.lower() not in _STOPWORDS
    ]
    boosted_pool_ids = list(pool_ids)
    pool_set = set(pool_ids)

    for row in chunks:
        if row["chunk_id"] in pool_set:
            continue
        text_l = row["text"].lower()
        if any(stem in text_l for stem in query_stems):
            boosted_pool_ids.append(row["chunk_id"])
            pool_set.add(row["chunk_id"])

    # Cross-encoder rerank
    pool_ids_for_rerank = boosted_pool_ids[:MAX_RERANK_POOL]
    pool_rows = [chunk_map[cid] for cid in pool_ids_for_rerank]

    if len(pool_rows) > 1:
        pairs = [[query, row["text"]] for row in pool_rows]
        ce_scores = reranker.predict(pairs)
        ranked = sorted(zip(ce_scores, pool_rows), key=lambda x: x[0], reverse=True)
        top_sim = _sigmoid(float(ranked[0][0]))
        ce_pool = [row for _, row in ranked[: k_final * 3]]
    else:
        ce_pool = pool_rows[:k_final]
        top_sim = 0.5

    # MMR diversity
    if len(ce_pool) > k_final:
        candidate_texts = [row["text"] for row in ce_pool]
        c_embs = embedder.encode(
            candidate_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        selected: list[int] = []
        remaining = list(range(len(ce_pool)))

        for _ in range(k_final):
            if not remaining:
                break

            if not selected:
                scores = [_cos(main_q_emb, c_embs[i]) for i in remaining]
            else:
                scores = [
                    0.70 * _cos(main_q_emb, c_embs[i])
                    - 0.30 * max(_cos(c_embs[i], c_embs[j]) for j in selected)
                    for i in remaining
                ]

            best_idx = remaining[int(np.argmax(scores))]
            selected.append(best_idx)
            remaining.remove(best_idx)

        final_rows = [ce_pool[i] for i in selected]
    else:
        final_rows = ce_pool[:k_final]

    parts = []
    for row in final_rows:
        sec = row.get("section", "General")
        parts.append(f"[Source: {sec}]\n{row['text']}")

    context_str = "\n\n---\n\n".join(parts)
    return context_str, final_rows, top_sim


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def _faith_inline(answer: str, chunk_rows: list[dict[str, Any]]) -> float:
    if not answer.strip() or not chunk_rows:
        return 0.0

    chunks = [row["text"] for row in chunk_rows]
    embedder = load_embedder()

    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 12]
    c_embs = embedder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)

    if not sentences:
        a_emb = embedder.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0]
        return float(max(_cos(a_emb, c) for c in c_embs))

    scores = []
    for sent in sentences:
        s_emb = embedder.encode([sent], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores.append(max(_cos(s_emb, c) for c in c_embs))
    return float(np.mean(scores))


def _call_model(query: str, context: str, system_prompt: str) -> str:
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context from official POEA documents:\n---\n{context}\n---\n\n"
                    f"Question: {query}\n\nAnswer:"
                ),
            },
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_answer(query: str) -> dict[str, Any]:
    if not query or not query.strip():
        return {
            "answer": "Please enter a valid question.",
            "context": "",
            "chunks": [],
            "latency_sec": 0.0,
            "confidence": 0.0,
            "query": query,
        }

    t_start = time.time()

    try:
        context, chunk_rows, confidence = retrieve_context(query)
    except Exception as e:
        return {
            "answer": f"Retrieval error: {e}",
            "context": "",
            "chunks": [],
            "latency_sec": 0.0,
            "confidence": 0.0,
            "query": query,
        }

    try:
        answer = _call_model(query, context, SYSTEM_PROMPT)
    except Exception as e:
        return {
            "answer": f"Generation error: {e}",
            "context": context,
            "chunks": chunk_rows,
            "latency_sec": 0.0,
            "confidence": confidence,
            "query": query,
        }

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
        "query": query,
        "answer": answer,
        "context": context,
        "chunks": chunk_rows,
        "latency_sec": latency,
        "confidence": round(float(confidence), 4),
        "faithfulness_estimate": round(float(_faith_inline(answer, chunk_rows)), 4),
    }


# =============================================================================
# APP SUPPORT
# =============================================================================

def warmup() -> None:
    _ensure_assets()
    _ = get_groq_client()
    _ = load_chunks()
    _ = load_bm25()
    _ = load_faiss_index()
    _ = load_embedder()
    _ = load_reranker()


def healthcheck() -> dict[str, Any]:
    info: dict[str, Any] = {
        "model_label": MODEL_LABEL,
        "groq_model": GROQ_MODEL,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "chunks_exists": CHUNKS_PATH.exists(),
        "faiss_exists": FAISS_PATH.exists(),
        "bm25_exists": BM25_PATH.exists(),
    }

    try:
        info["num_chunks"] = len(load_chunks())
    except Exception as e:
        info["chunks_error"] = str(e)

    try:
        info["faiss_ntotal"] = int(load_faiss_index().ntotal)
    except Exception as e:
        info["faiss_error"] = str(e)

    try:
        info["groq_key_loaded"] = bool(get_groq_api_key())
    except Exception as e:
        info["groq_error"] = str(e)

    return info


if __name__ == "__main__":
    warmup()
    test = generate_answer("What is the maximum duration of a seafarer's employment contract?")
    print(test["answer"])