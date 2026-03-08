from __future__ import annotations

import traceback

import streamlit as st

from pipeline import MODEL_DISPLAY_NAME, generate_answer, healthcheck, warmup

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SeamanBot",
    page_icon="⚓",
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


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("SeamanBot")
    st.caption(f"Model: {MODEL_DISPLAY_NAME}")

    st.markdown("### About")
    st.write(
        "RAG chatbot for the POEA Standard Employment Contract for Filipino Seafarers."
    )

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

    if st.button("Maximum contract duration", use_container_width=True):
        st.session_state.example_query = "What is the maximum duration of a seafarer's employment contract?"
        st.session_state.input_value = st.session_state.example_query

    if st.button("Medical benefits on injury", use_container_width=True):
        st.session_state.example_query = "What medical benefits are available to seafarers who get injured on board?"
        st.session_state.input_value = st.session_state.example_query

    if st.button("Overtime pay", use_container_width=True):
        st.session_state.example_query = "Is a seafarer entitled to overtime pay?"
        st.session_state.input_value = st.session_state.example_query

    if st.button("Death benefits", use_container_width=True):
        st.session_state.example_query = "What death benefits are provided if a seafarer dies during the contract?"
        st.session_state.input_value = st.session_state.example_query

# =============================================================================
# MAIN
# =============================================================================

st.title("⚓ SeamanBot")
st.caption("POEA Standard Employment Contract for Filipino Seafarers")

if startup_error:
    st.error(
        "Startup failed. Check your secrets and asset files.\n\n"
        f"Details: {startup_error}"
    )
    st.stop()

st.markdown(
    """
Ask any question about the **POEA Standard Employment Contract for Filipino Seafarers**.

This Streamlit version is adapted from your notebook and uses:
- BGE-Large dense retrieval
- BM25 sparse retrieval
- RRF fusion
- BGE reranker
- Groq answer generation
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
        placeholder="e.g. What is the maximum contract duration?",
    )
    submitted = st.form_submit_button("Ask SeamanBot", use_container_width=True)

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
        st.session_state.input_value = ""
        st.session_state.example_query = ""

        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = generate_answer(clean_query)

            st.session_state.chat_history.append(
                {
                    "user": clean_query,
                    "assistant": result["answer"],
                    "latency_sec": result.get("latency_sec", 0.0),
                    "confidence": result.get("confidence", 0.0),
                    "faithfulness_estimate": result.get("faithfulness_estimate", 0.0),
                    "context": result.get("context", ""),
                    "chunks": result.get("chunks", []),
                }
            )

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

            col1, col2, col3 = st.columns(3)
            col1.metric("Latency (s)", f"{turn.get('latency_sec', 0.0):.2f}")
            col2.metric("Retrieval confidence", f"{turn.get('confidence', 0.0):.4f}")
            col3.metric("Faithfulness estimate", f"{turn.get('faithfulness_estimate', 0.0):.4f}")

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