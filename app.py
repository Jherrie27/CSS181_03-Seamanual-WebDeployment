from __future__ import annotations

import traceback

import streamlit as st

from pipeline import MODEL_DISPLAY_NAME, generate_answer, healthcheck, warmup

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Seamanual",
    page_icon="",
    layout="wide",
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "example_query" not in st.session_state:
    st.session_state.example_query = ""

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# =============================================================================
# STARTUP
# =============================================================================

startup_error = None
try:
    warmup()
except Exception as e:
    startup_error = str(e)

# =============================================================================
# HELPERS
# =============================================================================

def reset_app() -> None:
    st.session_state.chat_history = []
    st.session_state.example_query = ""
    st.session_state.input_value = ""


def render_chunks(chunks: list[dict]) -> None:
    if not chunks:
        st.info("No retrieved chunks available.")
        return

    for i, row in enumerate(chunks, start=1):
        title = f"{i}. {row.get('section', 'General')}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Chunk ID:** `{row.get('chunk_id', 'N/A')}`")
            st.write(row.get("text", ""))


def render_metrics(turn: dict) -> None:
    m = turn.get("metrics", {})
    st.markdown("**Evaluation Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall@K",          f"{m.get('recall_at_k',       0.0):.4f}")
    col2.metric("Precision@K",       f"{m.get('precision_at_k',    0.0):.4f}")
    col3.metric("Context Relevancy", f"{m.get('context_relevancy', 0.0):.4f}")
    col4.metric("Faithfulness",      f"{m.get('faithfulness',      0.0):.4f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Answer Relevancy",  f"{m.get('answer_relevancy',  0.0):.4f}")
    col6.metric("ROUGE-L",           f"{m.get('rouge_l',           0.0):.4f}")
    col7.metric("BERTScore",         f"{m.get('bert_score',        0.0):.4f}")
    col8.metric("Latency (s)",       f"{turn.get('latency_sec',    0.0):.2f}")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("Seamanual")
    st.caption(f"Model: {MODEL_DISPLAY_NAME}")

    st.markdown("### About")
    st.write("RAG chatbot for the Seaman Training Manual (NAVEDTRA 14067).")

    if st.button("Clear conversation", use_container_width=True):
        reset_app()
        st.rerun()

    with st.expander("System status", expanded=False):
        try:
            st.json(healthcheck())
        except Exception as e:
            st.error(f"Healthcheck failed: {e}")

    st.markdown("---")
    st.markdown("### Example questions")

    examples = [
        ("Watchstander responsibilities",  "What are the responsibilities of a seaman standing watch on deck?"),
        ("Relieving a watch post",         "What should a watchstander do before leaving their post?"),
        ("Eye splice procedure",           "What is the correct procedure for making an eye splice with nylon line?"),
        ("Letting go the anchor",          "What is the proper procedure for letting go the anchor?"),
        ("Lifeboat handling",              "What is the proper procedure for handling a lifeboat?"),
        ("Naval projectile types",         "What are the different types of naval projectiles?"),
        ("Sound-powered telephone",        "What is a sound-powered telephone and how is it used?"),
        ("COMMENCE FIRING command",        "What does the COMMENCE FIRING command mean?"),
    ]
    for label, q in examples:
        if st.button(label, use_container_width=True):
            st.session_state.example_query = q
            st.session_state.input_value   = q

# =============================================================================
# MAIN
# =============================================================================

st.title("Seamanual")
st.caption("Seaman Training Manual — NAVEDTRA 14067")

if startup_error:
    st.error(
        "Startup failed. Check your secrets and asset files.\n\n"
        f"Details: {startup_error}"
    )
    st.stop()

st.markdown(
    """
Ask any question about the **Seaman Training Manual (NAVEDTRA 14067)**.

Pipeline: BGE-Large dense retrieval · BM25 sparse retrieval · RRF fusion · Cross-encoder reranker · Groq answer generation
"""
)

# =============================================================================
# INPUT FORM
# =============================================================================

default_query = st.session_state.input_value or st.session_state.example_query

with st.form("ask_form", clear_on_submit=False):
    user_query = st.text_area(
        "Question",
        value=default_query,
        height=100,
        placeholder="e.g. What are the responsibilities of a seaman standing watch?",
    )
    submitted = st.form_submit_button("Send Question", use_container_width=True)

# =============================================================================
# RUN QUERY
# =============================================================================

if submitted:
    clean_query = user_query.strip()

    if not clean_query:
        st.warning("Please enter a valid question.")
    elif len(clean_query) < 5:
        st.warning("Question too short.")
    else:
        st.session_state.input_value   = ""
        st.session_state.example_query = ""

        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = generate_answer(clean_query)

            st.session_state.chat_history.append({
                "user":        clean_query,
                "assistant":   result["answer"],
                "latency_sec": result.get("latency_sec", 0.0),
                "context":     result.get("context", ""),
                "chunks":      result.get("chunks", []),
                "metrics":     result.get("metrics", {}),
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc(), language="python")

# =============================================================================
# CHAT HISTORY
# =============================================================================

if st.session_state.chat_history:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])

        with st.chat_message("assistant"):
            st.write(turn["assistant"])
            render_metrics(turn)

            with st.expander("Retrieved context", expanded=False):
                context = turn.get("context", "")
                if context:
                    st.text(context)
                else:
                    st.info("No context returned.")

            with st.expander("Retrieved chunks", expanded=False):
                render_chunks(turn.get("chunks", []))
else:
    st.info("Ask a question to begin.")


