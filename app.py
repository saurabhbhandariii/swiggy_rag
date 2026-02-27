"""
app.py
------
Streamlit UI for the  RAG QA System.
FREE VERSION — uses HuggingFace embeddings + Groq LLM (no OpenAI needed).
"""

import os
import tempfile

import streamlit as st

from rag_pipeline import (
    load_and_chunk_pdf,
    build_vectorstore,
    load_vectorstore,
    answer_question,
    FAISS_INDEX_DIR,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=" RAG QA System",
    page_icon="🍊",
    layout="wide",
)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
for key, default in [("vectorstore", None), ("processed", False), ("pdf_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Load existing FAISS index on startup (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def try_load_existing_index():
    return load_vectorstore(FAISS_INDEX_DIR)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Setup")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Any  PDF",
        type=["pdf"],
        help="Supports 200+ page PDFs",
    )

    process_btn = st.button("⚙️ Process Document", use_container_width=True)

    st.markdown("---")
    st.markdown("**Status**")
    status_ph = st.empty()

    # Auto-load existing index
    if not st.session_state.processed:
        existing = try_load_existing_index()
        if existing is not None:
            st.session_state.vectorstore = existing
            st.session_state.processed   = True
            status_ph.success("✅ Existing index loaded from disk.")
        else:
            status_ph.info("ℹ️ Upload a PDF and click Process Document.")

    if st.session_state.processed:
        status_ph.success("✅ Document processed and ready!")

    # ── Process button ──
    if process_btn:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a PDF first.")
        elif st.session_state.processed and st.session_state.pdf_name == uploaded_file.name:
            status_ph.success("✅ Already processed. Ready to answer!")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("📖 Chunking PDF..."):
                    chunks = load_and_chunk_pdf(tmp_path)
                status_ph.info(f"✂️ {len(chunks)} chunks created. Embedding (this runs locally)...")

                with st.spinner("🔢 Building FAISS index with local embeddings..."):
                    vs = build_vectorstore(chunks)

                st.session_state.vectorstore = vs
                st.session_state.processed   = True
                st.session_state.pdf_name    = uploaded_file.name
                status_ph.success(f"✅ Processed {len(chunks)} chunks from '{uploaded_file.name}'")

            except Exception as e:
                status_ph.error(f"❌ Error processing PDF: {e}")
                st.exception(e)
            finally:
                os.unlink(tmp_path)

    st.markdown("---")
    st.caption("LangChain · FAISS · HuggingFace · Groq LLaMA3 · Streamlit")


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────
st.title("🍊  RAG QA System")
st.markdown(
    "Ask any question about the Swiggy Annual Report. "
    "Answers are grounded **strictly** in the document — no hallucination."
)
st.markdown("---")

question   = st.text_input(
    "💬 Ask a question",
    placeholder="e.g. What was the embezzlement amount in Scootsy?",
)
answer_btn = st.button("🔍 Get Answer", type="primary")

if answer_btn:
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    elif not st.session_state.processed or st.session_state.vectorstore is None:
        st.error("❌ Process a document first using the sidebar.")
    else:
        with st.spinner("🤔 Searching and generating answer..."):
            try:
                answer, source_docs = answer_question(question, st.session_state.vectorstore)

                st.subheader("📝 Answer")
                st.markdown(answer)

                if source_docs:
                    st.markdown("---")
                    st.subheader("📚 Supporting Context")
                    for i, doc in enumerate(source_docs, 1):
                        page   = doc.metadata.get("page", "N/A")
                        source = doc.metadata.get("source", "")
                        with st.expander(f"Chunk {i} — Page {page} | {source}", expanded=(i == 1)):
                            st.caption(f"**Page:** {page}  |  **Source:** {source}")
                            st.write(doc.page_content)

                    pages = sorted({doc.metadata.get("page", "?") for doc in source_docs})
                    st.info(f"📄 Referenced pages: {', '.join(map(str, pages))}")

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.exception(e)
