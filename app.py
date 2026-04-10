"""
app.py – DocChat: AI-powered document chatbot
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud  (connect GitHub repo)
"""

import base64
from typing import List, Dict, Optional

import streamlit as st

from utils.document_processor import extract_from_pdf, extract_from_docx
from utils.vector_store import VectorStore, get_embed_model
from utils.llm_handler import get_answer, get_document_summary


# ══════════════════════════════════ PAGE CONFIG ══════════════════════════════

st.set_page_config(
    page_title="DocChat – AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pre-load the embedding model once at startup — cached across all sessions
@st.cache_resource(show_spinner="⚙️ Loading AI model (one-time, ~10 seconds)…")
def _load_embed_model():
    return get_embed_model()

# ══════════════════════════════════ CUSTOM CSS ═══════════════════════════════

st.markdown(
    """
    <style>
    /* Tighten top padding */
    .block-container { padding-top: 1.8rem; }

    /* Welcome card gradient */
    .welcome-card {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .welcome-card h2 { font-size: 2rem; margin-bottom: 0.4rem; }
    .welcome-card p  { font-size: 1.1rem; opacity: 0.9; }

    /* Feature cards */
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        height: 100%;
    }
    .feature-card h3 { margin-bottom: 0.4rem; }

    /* Suggested question buttons */
    div[data-testid="stHorizontalBlock"] button {
        width: 100%;
        white-space: normal;
        height: auto;
        padding: 0.45rem 0.6rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Warm up the embedding model immediately — cached after first load
_load_embed_model()

# ══════════════════════════════════ SESSION STATE ════════════════════════════

def _init_state() -> None:
    # Auto-load API key from Streamlit Secrets (set in Streamlit Cloud dashboard)
    secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    defaults: Dict = {
        "vector_store": VectorStore(),
        "messages": [],       # [{"role", "content", "images", "sources"}]
        "documents": {},      # {filename: {"chunks": n, "images": n, "pages": n}}
        "doc_images": {},     # {filename: {page_num: [img_dict, ...]}}
        "doc_full_text": {},  # {filename: combined text string}
        "api_key": secret_key,
        "model": "llama-3.3-70b-versatile",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    # Always refresh from secrets in case session was reset
    if secret_key and not st.session_state.get("api_key"):
        st.session_state["api_key"] = secret_key


_init_state()


# ══════════════════════════════════ HELPERS ══════════════════════════════════

def _show_images(images: List[Dict], expanded: bool = True) -> None:
    """Render up to 3 document images in an expander."""
    if not images:
        return
    with st.expander("🖼️ Related diagrams / images from the document", expanded=expanded):
        cols = st.columns(min(len(images), 3))
        for i, img in enumerate(images[:3]):
            with cols[i]:
                try:
                    img_bytes = base64.b64decode(img["data"])
                    st.image(
                        img_bytes,
                        caption=(
                            f"{img.get('doc_name', '')}  ·  "
                            f"p.{img.get('page', '')}"
                        ),
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("⚠️ Could not render image")


def _show_sources(chunks: List[Dict]) -> None:
    """Render a collapsible list of citation sources."""
    if not chunks:
        return
    seen: set = set()
    unique: List[Dict] = []
    for c in chunks:
        key = f"{c.get('doc_name', '')}|{c.get('source', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    with st.expander("📚 Sources used"):
        for c in unique:
            score = c.get("relevance_score")
            score_str = f"  *(relevance: {score:.2f})*" if score is not None else ""
            st.markdown(
                f"- **{c.get('doc_name', 'Unknown')}** — {c.get('source', '')}  {score_str}"
            )


def _collect_images_for_chunks(chunks: List[Dict]) -> List[Dict]:
    """Deduplicated images from the same pages as retrieved chunks."""
    images: List[Dict] = []
    seen: set = set()
    for chunk in chunks:
        doc = chunk.get("doc_name", "")
        page = chunk.get("page", 0)
        for img in st.session_state.doc_images.get(doc, {}).get(page, []):
            key = img["data"][:32]
            if key not in seen:
                seen.add(key)
                images.append(img)
    return images


def _all_doc_images(doc_name: str, limit: int = 3) -> List[Dict]:
    """Collect all images from a document (used for visual-query fallback)."""
    result: List[Dict] = []
    for page_imgs in st.session_state.doc_images.get(doc_name, {}).values():
        result.extend(page_imgs)
        if len(result) >= limit:
            break
    return result[:limit]


# ══════════════════════════════════ CORE ACTIONS ═════════════════════════════

def _handle_question(question: str) -> None:
    """Full RAG pipeline: retrieve → answer → display."""
    # Show user bubble
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "images": [], "sources": []}
    )

    with st.chat_message("assistant"):
        # Step 1 – retrieval
        with st.spinner("🔍 Searching your documents…"):
            chunks = st.session_state.vector_store.search(
                question, st.session_state.api_key, top_k=5
            )

        # Step 2 – LLM answer
        with st.spinner("💬 Generating answer…"):
            answer = get_answer(
                question,
                chunks,
                st.session_state.api_key,
                st.session_state.model,
            )

        st.markdown(answer)

        # Step 3 – images
        images = _collect_images_for_chunks(chunks)

        # Fallback: if user asks about visuals and no images found via chunks,
        # pull images from the top-ranked document
        _visual_keywords = {
            "diagram", "chart", "figure", "image", "picture",
            "graph", "visual", "illustration", "table", "screenshot",
        }
        if not images and any(kw in question.lower() for kw in _visual_keywords):
            if chunks:
                top_doc = chunks[0].get("doc_name", "")
                images = _all_doc_images(top_doc)

        _show_images(images, expanded=True)
        _show_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "images": images[:3],
            "sources": chunks,
        }
    )


def _handle_summarize(doc_name: str) -> None:
    """Generate and display a document summary."""
    full_text = st.session_state.doc_full_text.get(doc_name, "")
    if not full_text:
        st.warning("No text found for this document.")
        return

    question = f"📋 Summarize: {doc_name}"
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "images": [], "sources": []}
    )

    with st.chat_message("assistant"):
        with st.spinner("📋 Summarizing document…"):
            summary = get_document_summary(
                full_text, doc_name, st.session_state.api_key, st.session_state.model
            )
        st.markdown(summary)
        # Show a selection of images from the document in summaries too
        doc_imgs = _all_doc_images(doc_name, limit=3)
        _show_images(doc_imgs, expanded=False)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": summary,
            "images": doc_imgs,
            "sources": [],
        }
    )


# ══════════════════════════════════ SIDEBAR ══════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 DocChat")
    st.caption("AI-powered document assistant")
    st.divider()

    # ── API / model settings ──────────────────────────────────────────────────
    _key_from_secrets = bool(st.secrets.get("GROQ_API_KEY", "")) if hasattr(st, "secrets") else False

    with st.expander(
        "🔑 API Settings",
        expanded=not bool(st.session_state.api_key),
    ):
        if _key_from_secrets:
            # Key is loaded server-side — never expose it in the UI
            st.success("✅ API key configured (server-side)")
        else:
            new_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="gsk_…",
                help="Get a free key at https://console.groq.com",
            )
            if new_key != st.session_state.api_key:
                st.session_state.api_key = new_key

            if st.session_state.api_key:
                st.success("✅ API key configured")
            else:
                st.info(
                    "👉 [Get free Groq key →](https://console.groq.com)"
                )

        st.session_state.model = st.selectbox(
            "Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
            ],
            index=0,
            help="70b = best quality  |  8b instant = fastest",
        )

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or Word files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if not st.session_state.api_key:
            st.warning("⚠️ Enter your Groq API key above first.")
        else:
            for f in uploaded_files:
                if f.name not in st.session_state.documents:
                    with st.status(
                        f"⚙️ Processing **{f.name}**…", expanded=True
                    ) as status:
                        try:
                            st.write("📖 Reading file…")
                            raw = f.read()

                            st.write("✂️ Extracting text and images…")
                            if f.name.lower().endswith(".pdf"):
                                result = extract_from_pdf(raw, f.name)
                            else:
                                result = extract_from_docx(raw, f.name)

                            n_chunks = len(result["chunks"])
                            n_imgs = sum(
                                len(v) for v in result["images"].values()
                            )
                            max_page = max(
                                (c["page"] for c in result["chunks"]), default=1
                            )

                            st.write(f"🔢 Embedding {n_chunks} text chunks…")
                            st.session_state.vector_store.add_documents(
                                result["chunks"], st.session_state.api_key
                            )

                            # Persist metadata
                            st.session_state.documents[f.name] = {
                                "chunks": n_chunks,
                                "images": n_imgs,
                                "pages": max_page,
                            }
                            st.session_state.doc_images[f.name] = result["images"]
                            st.session_state.doc_full_text[f.name] = "\n".join(
                                c["text"] for c in result["chunks"]
                            )

                            status.update(
                                label=(
                                    f"✅ {f.name}  "
                                    f"({n_chunks} sections · {n_imgs} images)"
                                ),
                                state="complete",
                            )
                        except Exception as err:
                            status.update(
                                label=f"❌ {f.name}: {err}", state="error"
                            )

    # ── Loaded docs list ──────────────────────────────────────────────────────
    if st.session_state.documents:
        st.divider()
        st.markdown("### 📚 Loaded Documents")
        for doc_name, meta in list(st.session_state.documents.items()):
            short = doc_name if len(doc_name) <= 26 else doc_name[:23] + "…"
            with st.expander(f"📄 {short}"):
                st.caption(
                    f"{meta['pages']} pages · "
                    f"{meta['chunks']} sections · "
                    f"{meta['images']} images"
                )
                if st.button(
                    "📋 Summarize this document", key=f"sum__{doc_name}"
                ):
                    st.session_state["_summarize_doc"] = doc_name

    st.divider()

    # ── Quick actions ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    if col1.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("🔄 Reset all", use_container_width=True):
        st.session_state.messages = []
        st.session_state.documents = {}
        st.session_state.doc_images = {}
        st.session_state.doc_full_text = {}
        st.session_state.vector_store.clear()
        st.rerun()

    # ── Export chat ───────────────────────────────────────────────────────────
    if st.session_state.messages:
        lines = []
        for m in st.session_state.messages:
            role = "You" if m["role"] == "user" else "DocChat"
            lines.append(f"[{role}]\n{m['content']}\n")
        st.download_button(
            "💾 Export chat (.txt)",
            data="\n".join(lines),
            file_name="docchat_chat_export.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ══════════════════════════════════ MAIN AREA ════════════════════════════════

st.markdown("## 💬 DocChat")
st.caption("Upload documents in the sidebar · Then ask anything below")

# ── Welcome screen (no docs yet) ─────────────────────────────────────────────
if not st.session_state.documents and not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>👋 Welcome to DocChat</h2>
            <p>Your documents, made conversational with AI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="feature-card">
            <h3>📄 PDF & Word</h3>
            <p>Upload any PDF or DOCX file — reports, manuals, contracts, research papers</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="feature-card">
            <h3>🔍 Smart Q&amp;A</h3>
            <p>Semantic search finds the right passage; Gemini crafts a grounded answer with citations</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="feature-card">
            <h3>🖼️ Visual Aware</h3>
            <p>Diagrams and images from the relevant pages are shown inline next to the answer</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.info(
        "👈  **Get started:** enter your free Groq API key and upload a document in the sidebar."
    )
    st.stop()

# ── Replay chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            _show_images(msg["images"], expanded=False)
        if msg.get("sources"):
            _show_sources(msg["sources"])

# ── Handle summary request from sidebar button ────────────────────────────────
_summarize_doc: Optional[str] = st.session_state.pop("_summarize_doc", None)
if _summarize_doc:
    _handle_summarize(_summarize_doc)

# ── Suggested starter questions (only before first message) ───────────────────
_pending_q: Optional[str] = st.session_state.pop("_pending_q", None)

if not st.session_state.messages and not _pending_q and st.session_state.documents:
    st.markdown("---")
    st.markdown("**💡 Suggested questions — click to ask:**")
    _suggestions = [
        "What is this document about?",
        "Summarize the key points",
        "Are there any diagrams or figures?",
        "What are the main conclusions?",
        "List all topics covered",
        "What statistics or data are mentioned?",
    ]
    for _row_start in range(0, len(_suggestions), 3):
        _row = _suggestions[_row_start : _row_start + 3]
        _cols = st.columns(len(_row))
        for _col, _sug in zip(_cols, _row):
            if _col.button(_sug, use_container_width=True, key=f"sug__{_sug[:20]}"):
                st.session_state["_pending_q"] = _sug
                st.rerun()

# Process suggestion that was set in a previous render pass
if _pending_q:
    _handle_question(_pending_q)

# ── Live chat input ───────────────────────────────────────────────────────────
_placeholder = (
    "Ask a question about your documents…"
    if st.session_state.documents
    else "Upload a document in the sidebar first…"
)
_ready = bool(st.session_state.api_key and st.session_state.documents)

if user_input := st.chat_input(_placeholder, disabled=not _ready):
    _handle_question(user_input)
