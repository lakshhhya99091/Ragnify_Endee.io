# Ragnify — AI Document Intelligence Platform

**Author:** Lakshay Arora
**Enrollment:** E22CSEU1555
**Institution:** Bennett University
**Submission:** Endee.io Internship-cum-Placement Evaluation — SDE / ML Track

---

## Origin and Motivation

This project was originally conceived and built during my engineering internship at **Indian Oil Corporation Limited (IOCL)**, one of India's largest public sector enterprises. At IOCL, engineers and operational staff routinely work with hundreds of PDFs — equipment manuals, safety compliance documents, inspection reports, and procurement records — spread across departments with no unified way to search or query them intelligently.

The problem was real and recurring: finding a specific clause in a 300-page compliance manual, or cross-referencing maintenance logs across multiple documents, took hours of manual effort. I built the first version of Ragnify to address this — a Retrieval Augmented Generation system that let staff ask plain English questions and get accurate, document-grounded answers instantly.

After the internship, I continued developing the project independently. This repository represents that evolution: a substantially upgraded version of the original IOCL prototype, now rebuilt around **Endee** as the vector database, with a complete dashboard interface, ML retrieval benchmarking, and production-grade engineering throughout.

---

## What This Project Does

Ragnify lets you upload any number of PDFs and ask natural language questions across all of them simultaneously. It finds the most relevant passages using semantic vector search powered by Endee, then passes those passages as context to a locally-running LLaMA-2 7B language model, which generates a precise, grounded answer — along with citations showing exactly which document and page the answer came from.

Everything runs on your machine. No cloud. No API keys. No data leaving the system.

---

## Why I Rebuilt It with Endee

The original IOCL version used FAISS as the vector store. FAISS works for prototyping but has significant limitations in production: it is entirely in-memory, loses all indexed data on restart, has no support for named collections, and cannot filter results by metadata such as document source or page number.

When I evaluated Endee as an alternative, it solved every one of these problems. Endee provides persistent named collections, structured metadata storage alongside vectors, and a clean API that integrates naturally into a LangChain-based RAG pipeline. Switching to Endee meant that document embeddings survive application restarts, queries can be filtered by source or page, and the architecture is genuinely production-ready rather than just a demo.

This is not a cosmetic change — it is a fundamental improvement to how the system stores and retrieves knowledge.

---

## Improvements Over the Original IOCL Prototype

### Vector Store: FAISS → Endee

Endee replaces FAISS as the sole retrieval backend. All document embeddings are indexed into a persistent Endee collection at ingestion time and queried semantically at runtime. The system falls back to FAISS automatically only if Endee is not installed, ensuring the application never breaks during environment setup.

### Five-Section Dashboard

The original was a single scrollable page with no structure. This version is a navigable five-section dashboard covering system health monitoring, PDF viewing, Q&A, ML model comparison, and an architecture reference — making it usable as a real tool rather than a demo script.

### Selective OCR

The original applied OCR to every page regardless of whether text was already present, which was slow and caused errors on digital PDFs. This version applies pytesseract only when PyMuPDF returns no extractable text, cutting processing time significantly on standard documents.

### Source-Level Answer Citations

Every answer now includes expandable cards showing the exact page number and source filename from which the context was retrieved. Users can verify any answer against the original document in seconds.

### ML Retrieval Comparison Panel

A dedicated benchmarking section compares how Linear Regression, Random Forest, and XGBoost rank document chunks for a given query, showing relevance scores and inference latency for each model. This panel makes retrieval behaviour transparent and interpretable.

### Chat History Export

Session history is recorded with timestamps and surfaced as a one-click download in the Q&A interface, rather than being silently appended to a hidden file.

---

## Technical Architecture

```
PDF Files
    │
    ▼
[PyMuPDF]  ──(blank page?)──►  [pytesseract OCR]
    │
    ▼
[RecursiveCharacterTextSplitter]   chunk_size=800, overlap=100
    │
    ▼
[sentence-transformers/all-MiniLM-L6-v2]   384-dim dense vectors
    │
    ▼
[Endee VectorStore]   persistent collection, metadata-aware indexing
    │
    ▼   (at query time)
[Endee semantic search]   top-K nearest chunks by cosine similarity
    │
    ▼
[LLaMA-2 7B · llama.cpp]   answer generation grounded in retrieved context
    │
    ▼
[Streamlit Dashboard]   answer + page-level source citations
```

---

## How Endee Is Integrated

```python
import endee

# Create a named, persistent collection
db = endee.VectorStore(collection_name="ragnify_docs")
db.clear()

# Index document chunk embeddings at ingestion
db.add(
    vectors=embeddings,     # List[List[float]] — 384-dim MiniLM vectors
    documents=texts,        # chunk text strings
    metadatas=metas         # {"page": int, "source": str}
)

# Semantic search at query time
query_vec = embed.embed_query(user_question)
results = db.search(vector=query_vec, top_k=3)
```

Retrieved chunks carry their metadata — page number and source filename — which the UI surfaces alongside every generated answer for full traceability.

---

## Comparison: Original Prototype vs. This Version

| Aspect | IOCL Prototype | This Version |
|---|---|---|
| Vector Database | FAISS (in-memory, no persistence) | Endee (persistent, metadata-aware) |
| UI Structure | Single scrollable page | Five-section navigable dashboard |
| OCR | Applied to all pages unconditionally | Applied selectively on blank pages only |
| Answer Citations | None | Expandable page and source cards |
| ML Panel | Plain unstructured text | Scored cards with latency per model |
| History | Silent file append | Timestamped download in UI |
| Design | Default Streamlit | Custom dark design system |
| Fallback Handling | None | Graceful FAISS fallback if Endee absent |
| Production Readiness | Prototype | Structured, modular, deployable |

---

## Key Strengths

**Fully local and private.** No document content, query, or answer is transmitted externally. Suitable for sensitive organisational data.

**No operating costs.** LLaMA-2 runs via llama.cpp and embeddings via HuggingFace locally. Zero per-token charges or API rate limits.

**Endee as the retrieval core.** The project is built around Endee end-to-end — from ingestion through indexing to query-time retrieval — demonstrating a complete production RAG integration.

**Multi-document cross-referencing.** Users query across all loaded PDFs simultaneously, with every answer traceable to a specific source and page.

**Scanned document support.** Selective OCR ensures the system works on image-based and scanned PDFs as well as digital ones.

**Retrieval transparency.** The ML Compare panel shows how different model families rank the same chunks, making retrieval behaviour interpretable rather than opaque.

**Modular architecture.** Each pipeline stage — ingestion, embedding, vector store, LLM, UI — is independently replaceable with minimal code changes.

---

## Setup and Execution

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR installed — https://github.com/UB-Mannheim/tesseract/wiki
- LLaMA-2 7B GGUF model file: `llama-2-7b-chat.Q4_K_M.gguf`

### Step 1: Star and Fork Endee

This step is mandatory per the evaluation criteria. Star and fork the repository before proceeding.

```bash
# After forking on GitHub:
git clone https://github.com/<YOUR_USERNAME>/endee
cd endee
pip install -e .
```

### Step 2: Clone This Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/ragnify
cd ragnify
```

### Step 3: Install Dependencies

```bash
pip install streamlit langchain langchain-community \
    sentence-transformers pymupdf pdf2image pytesseract \
    scikit-learn xgboost faiss-cpu pillow
```

### Step 4: Place the LLaMA-2 Model

```bash
mkdir models
# Copy llama-2-7b-chat.Q4_K_M.gguf into the models/ directory
```

### Step 5: Run

```bash
streamlit run ragnify_dashboard.py
```

Application runs at http://localhost:8501.

---

## Features

| Feature | Description |
|---|---|
| PDF Upload | Multiple PDFs up to 1 GB via sidebar |
| PDF Viewer | Side-by-side rendering with keyword highlighting |
| Q&A Engine | LLaMA-2 answers grounded in Endee-retrieved chunks |
| Source Citations | Page number and filename for every answer |
| ML Retrieval Compare | Linear Regression, Random Forest, XGBoost benchmarked per query |
| Chat History Export | Timestamped Q&A log downloadable from the interface |
| System Dashboard | Live status for PDFs, chunks, model, and Endee health |
| OCR Fallback | Automatic pytesseract OCR for scanned pages |

---

## Project Structure

```
ragnify/
├── ragnify_dashboard.py          Main Streamlit application
├── models/
│   └── llama-2-7b-chat.Q4_K_M.gguf
├── stored_pdfs/                  Uploaded PDFs (auto-created)
├── chat_history/
│   └── chat_history.txt          Session log
└── README.md
```

---

## Acknowledgements

- Endee (https://github.com/endee-io/endee) — Vector database at the core of all retrieval
- Indian Oil Corporation Limited (IOCL) — Origin of the problem statement and initial prototype
- Meta AI LLaMA-2 — Local language model
- LangChain — RAG orchestration and chunking
- HuggingFace sentence-transformers — Embedding model
- Streamlit — Application framework
