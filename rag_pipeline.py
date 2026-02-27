"""
rag_pipeline.py
---------------
Core RAG pipeline: document loading, chunking, embedding,
vector store management, retrieval, re-ranking, and answer generation.

FREE VERSION:
  - Embeddings : HuggingFace sentence-transformers (runs locally, no API cost)
  - LLM        : Groq API — llama-3.3-70b-versatile (free tier, very fast)
  - Vector DB  : FAISS (local, always free)
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Free embeddings (local, no API key needed) ────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# ── Free LLM via Groq ─────────────────────────────────────────────────────
from langchain_groq import ChatGroq

# ── Cross-encoder re-ranking (already free / local) ──────────────────────
from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CHUNK_SIZE          = 1200
CHUNK_OVERLAP       = 200
FAISS_INDEX_DIR     = "faiss_index"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # free, local
LLM_MODEL           = "llama-3.3-70b-versatile"                 # free on Groq
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_SIMILARITY    = 10
TOP_K_RERANK        = 5
TOP_K_LLM           = 3
BATCH_SIZE          = 64

# ─────────────────────────────────────────────
# Anti-hallucination system prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial analyst assistant.
Answer ONLY from the provided context.
If the answer is not present in the context, say:
'The answer is not available in the provided document.'
Do not use outside knowledge.
Cite page numbers in your answer."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────
# Text cleaning utility
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove extra whitespace, fix hyphenation artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[a-z])-\s+', '', text)
    return text.strip()


# ─────────────────────────────────────────────
# Document Loading & Chunking
# ─────────────────────────────────────────────
def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF page-by-page and split into chunks.
    Metadata: 1-indexed page number + source filename.
    """
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Document] = []
    source_filename = Path(pdf_path).name

    for page_doc in loader.lazy_load():
        page_doc.page_content = clean_text(page_doc.page_content)
        if not page_doc.page_content:
            continue
        page_chunks = splitter.split_documents([page_doc])
        for chunk in page_chunks:
            chunk.metadata["source"] = source_filename
            chunk.metadata["page"]   = chunk.metadata.get("page", 0) + 1
        all_chunks.extend(page_chunks)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────
# Embedding + FAISS Vector Store
# ─────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns local HuggingFace embeddings.
    Model is downloaded once (~90 MB) and cached on disk — no API key needed.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document], index_dir: str = FAISS_INDEX_DIR) -> FAISS:
    """
    Embed chunks in batches and build/save FAISS index locally.
    """
    logger.info("Building FAISS vector store with HuggingFace embeddings...")
    embeddings = get_embeddings()
    os.makedirs(index_dir, exist_ok=True)

    vectorstore: Optional[FAISS] = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        logger.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.merge_from(FAISS.from_documents(batch, embeddings))

    vectorstore.save_local(index_dir)
    logger.info(f"FAISS index saved to: {index_dir}")
    return vectorstore


def load_vectorstore(index_dir: str = FAISS_INDEX_DIR) -> Optional[FAISS]:
    """Load persisted FAISS index from disk (returns None if not found)."""
    index_path = Path(index_dir)
    if index_path.exists() and (index_path / "index.faiss").exists():
        logger.info(f"Loading existing FAISS index from: {index_dir}")
        return FAISS.load_local(
            index_dir,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


# ─────────────────────────────────────────────
# Cross-Encoder Re-Ranking (free, local)
# ─────────────────────────────────────────────
class CrossEncoderReranker:
    _instance = None
    _model: Optional[CrossEncoder] = None

    @classmethod
    def get(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if CrossEncoderReranker._model is None:
            logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            CrossEncoderReranker._model = CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(self, query: str, docs: List[Document], top_k: int = TOP_K_RERANK) -> List[Document]:
        if not docs:
            return []
        scores = CrossEncoderReranker._model.predict([(query, d.page_content) for d in docs])
        return [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]]


# ─────────────────────────────────────────────
# Full Retrieval Pipeline
# ─────────────────────────────────────────────
def retrieve_and_rerank(query: str, vectorstore: FAISS) -> List[Document]:
    """Stage 1: FAISS top-10 → Stage 2: CrossEncoder top-5 → Stage 3: top-3 to LLM."""
    initial_docs = vectorstore.similarity_search(query, k=TOP_K_SIMILARITY)
    reranked     = CrossEncoderReranker.get().rerank(query, initial_docs, top_k=TOP_K_RERANK)
    return reranked[:TOP_K_LLM]


# ─────────────────────────────────────────────
# LLM Answer Generation (via Groq — free tier)
# ─────────────────────────────────────────────
def format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        page   = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "")
        parts.append(f"[Page {page} | {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(query: str, vectorstore: FAISS) -> Tuple[str, List[Document]]:
    """Full pipeline: retrieve → re-rank → generate. Returns (answer, source_docs)."""
    docs = retrieve_and_rerank(query, vectorstore)
    if not docs:
        return "The answer is not available in the provided document.", []

    context = format_context(docs)

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer, docs
