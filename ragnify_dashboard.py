import os, fitz, base64, pytesseract, requests, json, time
import streamlit as st
import numpy as np
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Endee vector database integration
# Install: pip install endee (after forking https://github.com/endee-io/endee)
try:
    import endee
    ENDEE_AVAILABLE = True
except ImportError:
    ENDEE_AVAILABLE = False
    st.warning("⚠️ Endee not installed. Run: pip install endee  |  Falling back to FAISS.")
    from langchain_community.vectorstores import FAISS

# ─── CONFIG ────────────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
UPLOAD_DIR   = "stored_pdfs"
HISTORY_DIR  = "chat_history"
MODEL_PATH   = os.path.abspath("./models/llama-2-7b-chat.Q4_K_M.gguf")
for d in [UPLOAD_DIR, HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(layout="wide", page_title="Ragnify · AI Document Intelligence", page_icon="⚡")

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --red:    #e10600;
    --dark:   #0a0a0a;
    --card:   #111111;
    --border: #222222;
    --text:   #e8e8e8;
    --muted:  #666666;
    --accent: #ff4433;
    --green:  #00ff88;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton>button:hover { background: #ff2010 !important; transform: translateY(-1px); }

/* Inputs */
.stTextInput>div>div>input, .stTextArea textarea {
    background: #161616 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput>div>div>input:focus { border-color: var(--red) !important; }

/* Cards */
.ragnify-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.ragnify-hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0000 50%, #0a0a0a 100%);
    border: 1px solid #2a0000;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.ragnify-hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(225,6,0,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #fff;
    margin: 0;
    line-height: 1;
}
.hero-title span { color: var(--red); }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.75rem;
    letter-spacing: 0.1em;
}
.stat-bar {
    display: flex; gap: 1.5rem; justify-content: center; margin-top: 2rem; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(225,6,0,0.1);
    border: 1px solid rgba(225,6,0,0.3);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--red);
}
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    border-left: 3px solid var(--red);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}
.answer-box {
    background: #0d1a0d;
    border: 1px solid #1a4d1a;
    border-radius: 10px;
    padding: 1.25rem;
    font-family: 'Inter', sans-serif;
    color: var(--green);
    line-height: 1.7;
}
.source-chip {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin: 0.25rem;
}
.ml-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1rem;
    border-top: 2px solid var(--red);
}
.about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.about-item {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.about-item h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    color: var(--red);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.5rem;
}
.about-item p {
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0;
    line-height: 1.6;
}
.tab-nav {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.endee-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: linear-gradient(90deg, #1a0a00, #2a1000);
    border: 1px solid #ff6600;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #ff6600;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#fff;
                border-left:3px solid #e10600;padding-left:0.75rem;margin-bottom:1.5rem;">
        ⚡ RAGNIFY
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", ["🏠 Dashboard", "📄 PDF Viewer", "💬 Q&A", "🧠 ML Compare", "👤 About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:0.8rem;color:#666;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Upload PDFs</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
                f.write(file.read())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

    pdf_files = os.listdir(UPLOAD_DIR)
    selected_pdfs = st.multiselect("Select PDFs to analyse", pdf_files, default=pdf_files[:2])
    pdf_paths = [os.path.join(UPLOAD_DIR, f) for f in selected_pdfs if os.path.exists(os.path.join(UPLOAD_DIR, f))]

    st.markdown("---")
    if ENDEE_AVAILABLE:
        st.markdown('<div class="endee-badge">🟠 Endee Vector DB · Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="endee-badge" style="border-color:#555;color:#555;">⚪ Endee · Not installed</div>', unsafe_allow_html=True)

# ─── MODELS ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return LlamaCpp(
        model_path=MODEL_PATH, n_ctx=2048, max_tokens=512,
        temperature=0.7, n_gpu_layers=20, verbose=False,
    )

embed = get_embedder()
try:
    llm = get_llm()
    LLM_OK = True
except Exception:
    LLM_OK = False

# ─── PDF PARSING ───────────────────────────────────────────────────────────────
def parse_pdf(path):
    chunks, texts = [], []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    img = convert_from_path(path, first_page=i+1, last_page=i+1)[0]
                    text = pytesseract.image_to_string(img)
                if text.strip():
                    texts.append((i + 1, text))
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for pg, content in texts:
            for chunk in splitter.split_text(content):
                chunks.append(Document(page_content=chunk, metadata={"page": pg, "source": os.path.basename(path)}))
    except Exception as e:
        st.error(f"❌ Error parsing {path}: {e}")
    return chunks

# ─── ENDEE VECTOR STORE ────────────────────────────────────────────────────────
def build_vector_store(docs):
    """Build vector store using Endee if available, else FAISS."""
    if ENDEE_AVAILABLE:
        # Endee integration
        db = endee.VectorStore(collection_name="ragnify_docs")
        db.clear()
        texts  = [d.page_content for d in docs]
        metas  = [d.metadata for d in docs]
        vecs   = embed.embed_documents(texts)
        db.add(vectors=vecs, documents=texts, metadatas=metas)
        return db, vecs
    else:
        db   = FAISS.from_documents(docs, embedding=embed)
        vecs = embed.embed_documents([d.page_content for d in docs])
        return db, vecs

def endee_query(db, query_vec, top_k=3):
    """Semantic search via Endee."""
    return db.search(vector=query_vec, top_k=top_k)

# ─── LOAD DOCS ─────────────────────────────────────────────────────────────────
all_docs, faiss_db, embeddings_matrix = [], None, None
if pdf_paths:
    for path in pdf_paths:
        all_docs += parse_pdf(path)
    if all_docs:
        faiss_db, embeddings_matrix = build_vector_store(all_docs)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Dashboard":
    st.markdown("""
    <div class="ragnify-hero">
        <p class="hero-title">RAGNI<span>FY</span></p>
        <p class="hero-sub">// AI DOCUMENT INTELLIGENCE · POWERED BY ENDEE VECTOR DB</p>
        <div class="stat-bar">
            <span class="stat-pill">⚡ LLaMA-2 7B</span>
            <span class="stat-pill">🔍 Semantic Search</span>
            <span class="stat-pill">📊 ML Compare</span>
            <span class="stat-pill">🗄️ Endee Vector Store</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📄 PDFs Loaded", len(pdf_paths)),
        ("🧩 Chunks", len(all_docs)),
        ("🕐 History", len(open(os.path.join(HISTORY_DIR,"chat_history.txt")).read().split("Q:")) - 1
                       if os.path.exists(os.path.join(HISTORY_DIR,"chat_history.txt")) else 0),
        ("🧠 Model", "LLaMA-2 7B" if LLM_OK else "Not loaded"),
    ]
    for col, (label, val) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class="ragnify-card" style="text-align:center">
                <div style="font-size:1.8rem;font-family:'Syne',sans-serif;font-weight:800;color:#e10600">{val}</div>
                <div style="font-size:0.8rem;color:#666;margin-top:0.3rem">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">System Status</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="ragnify-card">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:2;color:#888">
                <span style="color:#{'00ff88' if LLM_OK else 'e10600'}">●</span> LLaMA-2 Model &nbsp;&nbsp; {'READY' if LLM_OK else 'NOT LOADED'}<br>
                <span style="color:#{'00ff88' if ENDEE_AVAILABLE else 'ffaa00'}">●</span> Endee Vector DB &nbsp; {'ACTIVE' if ENDEE_AVAILABLE else 'FALLBACK (FAISS)'}<br>
                <span style="color:#{'00ff88' if all_docs else 'e10600'}">●</span> Document Store &nbsp;&nbsp; {len(all_docs)} chunks indexed<br>
                <span style="color:#00ff88">●</span> Embeddings &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; all-MiniLM-L6-v2<br>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="ragnify-card">
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#888;line-height:1.9">
                🔷 Upload PDFs via sidebar<br>
                🔷 Select PDFs to analyse<br>
                🔷 Ask questions in Q&A tab<br>
                🔷 Compare ML models for retrieval<br>
                🔷 Download full chat history
            </div>
        </div>""", unsafe_allow_html=True)

    if all_docs:
        st.markdown('<div class="section-head">Indexed Documents</div>', unsafe_allow_html=True)
        sources = {}
        for doc in all_docs:
            src = doc.metadata.get("source","unknown")
            sources[src] = sources.get(src, 0) + 1
        for src, cnt in sources.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:#111;border:1px solid #1e1e1e;border-radius:8px;
                        padding:0.75rem 1rem;margin-bottom:0.5rem">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem">📄 {src}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#e10600">{cnt} chunks</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PDF VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📄 PDF Viewer":
    st.markdown('<div class="section-head">PDF Viewer</div>', unsafe_allow_html=True)
    search_term = st.text_input("🔍 Highlight keyword across all PDFs", placeholder="Enter keyword...")

    if not pdf_paths:
        st.info("Upload and select PDFs from the sidebar.")
    else:
        cols = st.columns(min(len(pdf_paths), 2))
        for idx, path in enumerate(pdf_paths):
            with cols[idx % 2]:
                try:
                    with fitz.open(path) as doc:
                        temp = f"temp_{idx}.pdf"
                        highlighted_page = 1
                        for i, page in enumerate(doc):
                            if search_term:
                                boxes = page.search_for(search_term)
                                if boxes:
                                    for box in boxes:
                                        page.add_highlight_annot(box)
                                    highlighted_page = i + 1
                                    break
                        doc.save(temp)
                    with open(temp, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{encoded}#page={highlighted_page}" '
                        f'width="100%" height="520" type="application/pdf" style="border:1px solid #222;border-radius:8px;"></iframe>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<div class="source-chip">📄 {os.path.basename(path)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Cannot render {path}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q&A
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "💬 Q&A":
    st.markdown('<div class="section-head">Ask Your Documents</div>', unsafe_allow_html=True)

    if ENDEE_AVAILABLE:
        st.markdown('<div class="endee-badge">🟠 Retrieval via Endee Vector DB</div>', unsafe_allow_html=True)

    query = st.text_input("Your question", placeholder="What does the document say about...")

    if query:
        if not all_docs:
            st.warning("No PDFs indexed. Please upload and select PDFs from the sidebar.")
        elif not LLM_OK:
            st.warning("LLaMA model not loaded. Check MODEL_PATH in config.")
        else:
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    if ENDEE_AVAILABLE:
                        q_vec = embed.embed_query(query)
                        results = endee_query(faiss_db, q_vec, top_k=3)
                        answer = llm(f"Answer based on context:\n\n{results}\n\nQuestion: {query}")
                        source_docs = []
                    else:
                        retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
                        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
                        result = qa.invoke({"query": query})
                        answer = result["result"]
                        source_docs = result["source_documents"]

                    st.markdown('<div class="section-head">Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    if source_docs:
                        st.markdown('<div class="section-head" style="font-size:1rem">Sources</div>', unsafe_allow_html=True)
                        for doc in source_docs:
                            with st.expander(f"📄 Page {doc.metadata.get('page','?')} · {doc.metadata.get('source','')}"):
                                st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:#888;line-height:1.6">{doc.page_content[:400]}...</div>', unsafe_allow_html=True)

                    # Save history
                    with open(os.path.join(HISTORY_DIR, "chat_history.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now()}\nQ: {query}\nA: {answer}\n\n")
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")

    # Download history
    hist_path = os.path.join(HISTORY_DIR, "chat_history.txt")
    if os.path.exists(hist_path):
        st.markdown("---")
        with open(hist_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button("📥 Download Chat History", content, file_name="ragnify_chat_history.txt")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🧠 ML Compare":
    st.markdown('<div class="section-head">ML Model Retrieval Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;color:#666;margin-bottom:1.5rem;font-family:'Inter',sans-serif">
        Compare how Linear Regression, Random Forest, and XGBoost rank document chunks 
        for your query using cosine-similarity scoring on embeddings.
    </div>""", unsafe_allow_html=True)

    query = st.text_input("Query to compare", placeholder="Enter your question...")

    if query and all_docs and embeddings_matrix is not None:
        with st.spinner("Running models..."):
            q_vec = embed.embed_query(query)
            X = np.array(embeddings_matrix)
            y = np.dot(X, np.array(q_vec))

            models = {
                "📈 Linear Regression": LinearRegression(),
                "🌲 Random Forest":     RandomForestRegressor(n_estimators=50, random_state=42),
                "⚡ XGBoost":           xgb.XGBRegressor(n_estimators=50, verbosity=0),
            }

            cols = st.columns(3)
            for (name, model), col in zip(models.items(), cols):
                with col:
                    try:
                        t0 = time.time()
                        model.fit(X, y)
                        pred = model.predict(X)
                        top_idx = int(np.argmax(pred))
                        elapsed = time.time() - t0
                        best   = all_docs[top_idx]
                        score  = float(pred[top_idx])
                        st.markdown(f"""
                        <div class="ml-card">
                            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.5rem">{name}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#e10600">
                                Score: {score:.4f} &nbsp;|&nbsp; {elapsed*1000:.0f}ms
                            </div>
                            <hr style="border-color:#222;margin:0.75rem 0">
                            <div style="font-size:0.8rem;color:#aaa;line-height:1.6">
                                {best.page_content[:280]}...
                            </div>
                            <div class="source-chip" style="margin-top:0.75rem">
                                Page {best.metadata.get('page','?')} · {best.metadata.get('source','')}
                            </div>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"{name} failed: {e}")
    elif query and not all_docs:
        st.info("Upload and select PDFs from the sidebar first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "👤 About":
    st.markdown("""
    <div class="ragnify-hero" style="text-align:left">
        <p style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#e10600;letter-spacing:0.15em;margin:0">// ABOUT THIS PROJECT</p>
        <p style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:#fff;margin:0.5rem 0">Ragnify</p>
        <p style="color:#888;font-size:0.95rem;margin:0;max-width:600px">
            An AI-powered document intelligence platform leveraging RAG (Retrieval Augmented Generation), 
            local LLMs, and Endee vector database for semantic search across your PDFs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-grid">
        <div class="about-item">
            <h4>🗄️ Vector Database</h4>
            <p>Endee — open-source vector DB for semantic similarity search. All document embeddings are indexed and queried via Endee for fast, accurate retrieval.</p>
        </div>
        <div class="about-item">
            <h4>🧠 LLM</h4>
            <p>LLaMA-2 7B (GGUF Q4_K_M) running locally via llama.cpp. No API keys, full privacy. GPU-accelerated with n_gpu_layers=20.</p>
        </div>
        <div class="about-item">
            <h4>📐 Embeddings</h4>
            <p>sentence-transformers/all-MiniLM-L6-v2 via HuggingFace. 384-dim dense vectors for semantic similarity scoring.</p>
        </div>
        <div class="about-item">
            <h4>📄 PDF Processing</h4>
            <p>PyMuPDF (fitz) for text extraction + pytesseract OCR fallback for scanned/image-based PDFs. Chunked via LangChain splitter.</p>
        </div>
        <div class="about-item">
            <h4>📊 ML Comparison</h4>
            <p>LinearRegression, RandomForest, and XGBoost trained on embedding dot-products to rank chunks. Compares retrieval quality across models.</p>
        </div>
        <div class="about-item">
            <h4>🖥️ Frontend</h4>
            <p>Streamlit with custom CSS theming. F1-inspired dark UI with red accents, Syne + JetBrains Mono typography.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Endee Integration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ragnify-card">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#888;line-height:2">
            <span style="color:#ff6600">$ git clone</span> https://github.com/endee-io/endee<br>
            <span style="color:#ff6600">$ pip install</span> endee<br><br>
            <span style="color:#666"># Endee replaces FAISS as the vector store:</span><br>
            <span style="color:#aaa">db = endee.VectorStore(collection_name=<span style="color:#00ff88">"ragnify_docs"</span>)</span><br>
            <span style="color:#aaa">db.add(vectors=embeddings, documents=texts, metadatas=metas)</span><br>
            <span style="color:#aaa">results = db.search(vector=query_vec, top_k=<span style="color:#00ff88">3</span>)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">How RAG Works</div>', unsafe_allow_html=True)
    steps = [
        ("1. Ingest", "PDFs are parsed, OCR'd if needed, and split into 800-char chunks."),
        ("2. Embed", "Each chunk is converted to a 384-dim vector using MiniLM."),
        ("3. Index", "Vectors are stored in Endee vector database for fast retrieval."),
        ("4. Query", "User query is embedded and top-K similar chunks retrieved from Endee."),
        ("5. Generate", "LLaMA-2 generates an answer grounded in the retrieved context."),
    ]
    for step, desc in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.75rem">
            <div style="min-width:80px;font-family:'Syne',sans-serif;font-weight:700;color:#e10600;font-size:0.85rem">{step}</div>
            <div style="color:#888;font-size:0.88rem;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Repository</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ragnify-card">
        <div style="font-size:0.9rem;color:#888;line-height:1.9">
            ⭐ <strong style="color:#fff">Star</strong> the Endee repo: 
            <a href="https://github.com/endee-io/endee" target="_blank" style="color:#e10600">github.com/endee-io/endee</a><br>
            🍴 <strong style="color:#fff">Fork</strong> it to your account before starting the project.<br>
            📁 <strong style="color:#fff">Build</strong> on the forked version and link your GitHub repo on Superset.
        </div>
    </div>
    """, unsafe_allow_html=True)
