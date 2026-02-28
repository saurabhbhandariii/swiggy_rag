# RAG QA System

## Assignment Objective

Build a **production-ready Retrieval-Augmented Generation (RAG)** application that answers questions strictly from the Swiggy Annual Report PDF — with zero hallucination, fast retrieval, and a clean Streamlit UI.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (app.py)                   │
│  Sidebar: Upload PDF → Process Button → Status Messages     │
│  Main:    Question Input → Get Answer → Answer + Context    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               DOCUMENT PROCESSING (rag_pipeline.py)         │
│                                                             │
│  PyPDFLoader (lazy_load - page by page)                     │
│       ↓                                                     │
│  Text Cleaning (regex - whitespace, hyphenation)            │
│       ↓                                                     │
│  RecursiveCharacterTextSplitter                             │
│  chunk_size=1200, chunk_overlap=200                         │
│  + Metadata: page number, source filename                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               EMBEDDING + VECTOR STORE                      │
│                                                             │
│  OpenAIEmbeddings (text-embedding-3-large)                  │
│  Batched: 64 chunks/batch → memory safe                     │
│       ↓                                                     │
│  FAISS Index (built + saved locally to ./faiss_index/)      │
│  On rerun → load existing index (no re-embedding)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               ADVANCED RETRIEVAL PIPELINE                   │
│                                                             │
│  Stage 1: FAISS similarity_search → Top 10 chunks          │
│       ↓                                                     │
│  Stage 2: CrossEncoder re-ranking                           │
│           (ms-marco-MiniLM-L-6-v2) → Top 5                 │
│       ↓                                                     │
│  Stage 3: Final Top 3 chunks → passed to LLM               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               LLM ANSWER GENERATION                         │
│                                                             │
│  GPT-4o (temperature=0, streaming)                          │
│  Anti-hallucination system prompt                           │
│  Cites page numbers in answer                               │
│  Falls back safely if answer not in context                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag_app/
│
├── app.py              ← Streamlit UI
├── rag_pipeline.py     ← Core RAG logic
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
└── README.md           ← This file

faiss_index/            ← Auto-created after first processing
├── index.faiss
└── index.pkl
```

---

## Setup Steps

### 1. Clone / Download the project

```bash
git clone <https://github.com/saurabhbhandariii/swiggy_rag>
cd rag_app
```

### 2. Create a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
nano .env
```

### 5. Obtain the Swiggy Annual Report PDF

Download the Swiggy Annual Report from:
> 📎 **Source:** [Swiggy Investor Relations – Annual Reports](https://ir.swiggy.in/financial-information/annual-reports)  
> *(Download the latest available Annual Report PDF)*

---

## How to Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Usage Flow

1. **Upload PDF** via the sidebar file uploader
2. Click **"⚙️ Process Document"** — chunks, embeds, and saves the FAISS index
3. **Type a question** in the main area text box
4. Click **"🔍 Get Answer"**
5. View the **answer**, **supporting context chunks**, and **page references**

> ✅ On subsequent runs, the FAISS index is auto-loaded from disk — no re-embedding needed.

---

## Key Design Decisions

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 1200 chars | Balances context richness vs. retrieval precision |
| `chunk_overlap` | 200 chars | Prevents context loss at chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Respects paragraph/sentence structure |
| Separators | `\n\n`, `\n`, `. `, ` `, `""` | Hierarchical splitting for clean boundaries |

The PDF is loaded page-by-page using `PyPDFLoader.lazy_load()` to prevent memory exhaustion on 200+ page documents. Each chunk retains its source page number as metadata.

---

### Retrieval Strategy

A **3-stage pipeline** is used to maximize precision:

```
FAISS Similarity Search (Top 10)
         ↓
Cross-Encoder Re-Ranking (Top 5)
         ↓
Top 3 → LLM
```

**Stage 1 – Dense Retrieval (FAISS):**  
Fast approximate nearest-neighbor search using cosine similarity on `text-embedding-3-large` vectors. Returns the top-10 most semantically similar chunks.

**Stage 2 – Cross-Encoder Re-Ranking:**  
The `ms-marco-MiniLM-L-6-v2` cross-encoder scores each (query, chunk) pair jointly — unlike bi-encoders, it reads both together for much higher precision. The top-5 are kept.

**Stage 3 – LLM Context:**  
Only the final top-3 chunks are passed to GPT-4o, keeping the prompt focused and within token budget.

---

### Anti-Hallucination Method

Three complementary layers prevent hallucination:

1. **Strict System Prompt:**  
   The LLM is explicitly instructed to answer only from provided context and respond with a standard message (`"The answer is not available in the provided document."`) if the answer isn't present.

2. **Temperature = 0:**  
   Deterministic generation eliminates creative guessing.

3. **Page Citation Requirement:**  
   The prompt mandates citing page numbers, forcing the model to ground its answer in specific document locations.

4. **Context-Only Architecture:**  
   No web search, no external tools, no model fine-tuning — the LLM only sees the retrieved chunks.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
  | `OPENAI_GROQ_KEY` | Your groq API key (required) |

---

## Performance Notes

- **Batched embedding** (64 chunks/batch) prevents OOM errors on large PDFs
- **FAISS persistence** eliminates re-embedding on subsequent runs
- **`@st.cache_resource`** caches the index load across Streamlit reruns
- **`st.session_state`** tracks processing status and prevents duplicate work
- **Lazy PDF loading** (`lazy_load()`) streams pages without loading the full file into RAM

  demo--
  
