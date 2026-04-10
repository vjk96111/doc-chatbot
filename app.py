"""
app.py - DocChat: AI-powered document chatbot
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud  (connect GitHub repo)
"""

import base64
import time
from typing import List, Dict, Optional

import streamlit as st

from utils.document_processor import extract_from_pdf, extract_from_docx
from utils.vector_store import VectorStore, get_embed_model
from utils.llm_handler import (
    get_answer,
    get_document_summary,
    get_highlight,
    keyword_search,
)


# PAGE CONFIG
st.set_page_config(
    page_title="DocChat - AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Embedding model cached across all sessions
@st.cache_resource(show_spinner="⚙️ Loading AI model (one-time)…")
def _load_embed_model():
    return get_embed_model()


# CSS (dark/light)
def _get_css(dark: bool) -> str:
    bg      = "#1e1e2e" if dark else "#ffffff"
    sb_bg   = "#16162a" if dark else "#f1f3f5"
    card_bg = "#2a2a3d" if dark else "#f8f9fa"
    card_bd = "#3d3d5c" if dark else "#dee2e6"
    text    = "#e0e0f0" if dark else "#212529"
    cap     = "#a0a0c0" if dark else "#6c757d"
    inp_bg  = "#24243a" if dark else "#ffffff"
    code_fg = "#a8d8ff" if dark else "#d63384"
    hl_bg   = "#3a3a1a" if dark else "#fffde7"
    return f"""
    <style>
    /* ── Main app background ── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {bg} !important;
        color: {text} !important;
    }}
    .main .block-container {{
        background-color: {bg} !important;
        padding-top: 1.8rem;
    }}
    /* ── Sidebar ── */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {sb_bg} !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] div {{
        color: {text} !important;
    }}
    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_bd} !important;
        border-radius: 10px !important;
        margin: 2px 0 !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th {{
        color: {text} !important;
    }}
    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {{
        background-color: {inp_bg} !important;
        color: {text} !important;
    }}
    [data-testid="stBottom"] > div {{
        background-color: {bg} !important;
    }}
    /* ── Text / Password inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {inp_bg} !important;
        color: {text} !important;
        border-color: {card_bd} !important;
    }}
    /* ── Expanders ── */
    [data-testid="stExpander"] details {{
        background-color: {card_bg} !important;
        border: 1px solid {card_bd} !important;
        border-radius: 8px;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > div {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}
    /* ── Selectbox / Radio ── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stRadio"] label {{
        color: {text} !important;
    }}
    /* ── General markdown text ── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: {text};
    }}
    /* ── Headings ── */
    h1, h2, h3, h4 {{ color: {text}; }}
    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"] p {{
        color: {cap} !important;
    }}
    /* ── Code ── */
    code {{
        background-color: {card_bg} !important;
        color: {code_fg} !important;
    }}
    /* ── Dividers ── */
    hr {{ border-color: {card_bd}; }}
    /* ── Custom cards ── */
    .welcome-card {{
        background: linear-gradient(135deg,#1a73e8 0%,#0d47a1 100%);
        color:white;padding:2.5rem 2rem;border-radius:14px;
        text-align:center;margin-bottom:1.5rem;
    }}
    .welcome-card h2{{font-size:2rem;margin-bottom:0.4rem;}}
    .welcome-card p {{font-size:1.1rem;opacity:0.9;}}
    .feature-card {{
        background:{card_bg};border:1px solid {card_bd};
        border-radius:10px;padding:1.2rem 1rem;
        text-align:center;height:100%;color:{text};
    }}
    .feature-card h3{{margin-bottom:0.4rem;}}
    .highlight-box {{
        background:{hl_bg};
        border-left:4px solid #f9a825;padding:0.6rem 1rem;
        border-radius:0 8px 8px 0;font-style:italic;
        margin:0.5rem 0;color:{text};
    }}
    .bookmark-card {{
        background:{card_bg};border:1px solid {card_bd};
        border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;color:{text};
    }}
    div[data-testid="stHorizontalBlock"] button {{
        width:100%;white-space:normal;height:auto;
        padding:0.45rem 0.6rem;font-size:0.85rem;
    }}
    </style>
    """


# LOGIN GATE
def _check_login() -> bool:
    users: Dict[str, str] = {}
    try:
        if "USERS" in st.secrets:
            users = dict(st.secrets["USERS"])
        elif "APP_PASSWORD" in st.secrets:
            users = {"user": st.secrets["APP_PASSWORD"]}
    except Exception:
        pass
    if not users:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.markdown(
        """<div style="max-width:380px;margin:6rem auto;padding:2.5rem 2rem;
           border:1px solid #dee2e6;border-radius:14px;
           box-shadow:0 4px 24px rgba(0,0,0,.08);text-align:center;">
           <div style="font-size:3rem;margin-bottom:.5rem;">📄</div>
           <h2 style="margin-bottom:.2rem;">DocChat</h2>
           <p style="color:#6c757d;margin-bottom:1.5rem;">Sign in to continue</p>
           </div>""",
        unsafe_allow_html=True,
    )
    col = st.columns([1, 1.4, 1])[1]
    with col:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if username in users and users[username] == password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.stop()
    return False


_check_login()


# SESSION STATE
def _init_state() -> None:
    secret_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    defaults: Dict = {
        "vector_store": VectorStore(),
        "messages":      [],
        "documents":     {},
        "doc_images":    {},
        "doc_full_text": {},
        "api_key":       secret_key,
        "model":         "openai/gpt-oss-120b",
        "language":      "English",
        "dark_mode":     False,
        "highlight_on":  False,
        "answer_style":  "Normal",   # "Short", "Normal", or "Long"
        "bookmarks":     [],
        "feedback":      {},
        "search_mode":   "Semantic",
        "_follow_ups":   [],   # follow-up questions from last answer
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if secret_key and not st.session_state.get("api_key"):
        st.session_state["api_key"] = secret_key


_init_state()
st.markdown(_get_css(st.session_state.dark_mode), unsafe_allow_html=True)
_load_embed_model()


# HELPERS
def _show_images(images: List[Dict], expanded: bool = True) -> None:
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
                        caption=f"{img.get('doc_name','')}  ·  p.{img.get('page','')}",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("⚠️ Could not render image")


def _show_sources(chunks: List[Dict]) -> None:
    if not chunks:
        return
    seen: set = set()
    unique: List[Dict] = []
    for c in chunks:
        key = f"{c.get('doc_name','')}|{c.get('source','')}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    with st.expander("📚 Sources used"):
        for c in unique:
            score = c.get("relevance_score")
            score_str = f"  *(relevance: {score:.2f})*" if score is not None else ""
            st.markdown(f"- **{c.get('doc_name','Unknown')}** — {c.get('source','')}  {score_str}")


def _collect_images_for_chunks(chunks: List[Dict]) -> List[Dict]:
    images: List[Dict] = []
    seen: set = set()
    for chunk in chunks:
        doc  = chunk.get("doc_name", "")
        page = chunk.get("page", 0)
        for img in st.session_state.doc_images.get(doc, {}).get(page, []):
            key = img["data"][:32]
            if key not in seen:
                seen.add(key)
                images.append(img)
    return images


def _all_doc_images(doc_name: str, limit: int = 3) -> List[Dict]:
    result: List[Dict] = []
    for page_imgs in st.session_state.doc_images.get(doc_name, {}).values():
        result.extend(page_imgs)
        if len(result) >= limit:
            break
    return result[:limit]


def _doc_stats(doc_name: str) -> Dict:
    text = st.session_state.doc_full_text.get(doc_name, "")
    words = text.lower().split()
    stop = {"the","a","an","and","or","but","in","on","at","to","for","of","is","it","as",
            "be","was","are","with","that","this","from","by","not","have","has","had","he",
            "she","they","we","you","i","his","her","their","its","been","were"}
    freq: Dict[str, int] = {}
    for w in words:
        w = w.strip(".,;:!?()[]\"'")
        if len(w) > 3 and w not in stop:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
    return {"word_count": len(words), "top_keywords": top}


# ── Follow-up / vague query detection ───────────────────────────────────────
_VAGUE_PATTERNS = [
    "give me", "give a", "long answer", "longer", "more detail", "detailed",
    "elaborate", "explain more", "go deeper", "expand", "tell me more",
    "more info", "in detail", "more about", "more on", "describe more",
    "describe in", "explain in",
]

def _resolve_search_query(question: str) -> str:
    """If question is a vague follow-up instruction, return the last substantive question."""
    q_lower = question.lower().strip()
    if len(q_lower) < 70 and any(p in q_lower for p in _VAGUE_PATTERNS):
        prior = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        if prior:
            return prior[-1]   # last real question (before current one is appended)
    return question


# CORE ACTIONS
def _handle_question(question: str, model_override: str = None, answer_style: str = None) -> None:
    # Capture chat history BEFORE appending current question
    recent_history = [
        {"role": m["role"], "content": m["content"][:600]}
        for m in st.session_state.messages[-4:]   # up to last 2 Q+A exchanges
    ]

    # If the user typed a vague follow-up, use the last real question for retrieval
    search_query = _resolve_search_query(question)
    # Use model_override or the user's selected model
    model_to_use = model_override or st.session_state.model
    # Use explicitly passed style or the current session state style
    _style = answer_style or st.session_state.get("answer_style", "Normal")

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "images": [], "sources": []}
    )

    with st.chat_message("assistant"):
        # Retrieval
        with st.spinner("🔍 Searching your documents…"):
            t0 = time.time()
            mode = st.session_state.search_mode
            if mode == "Keyword":
                chunks = keyword_search(search_query, st.session_state.vector_store.chunks, top_k=3)
            elif mode == "Hybrid":
                sem = st.session_state.vector_store.search(search_query, st.session_state.api_key, top_k=3)
                kw  = keyword_search(search_query, st.session_state.vector_store.chunks, top_k=3)
                seen_texts = set()
                chunks = []
                for c in sem + kw:
                    if c["text"] not in seen_texts:
                        seen_texts.add(c["text"])
                        chunks.append(c)
                chunks = chunks[:4]
            else:
                chunks = st.session_state.vector_store.search(search_query, st.session_state.api_key, top_k=3)
            t_search = time.time() - t0

        # Answer — follow-up questions come back in the same call, no extra delay
        with st.spinner("💬 Generating answer…"):
            t1 = time.time()
            answer, follow_ups, actual_model = get_answer(
                question, chunks,
                st.session_state.api_key,
                model_to_use,
                st.session_state.language,
                include_followups=True,
                chat_history=recent_history if recent_history else None,
                answer_style=_style,
            )
            t_llm = time.time() - t1

        # Persist follow-ups in session state so they survive reruns
        st.session_state["_follow_ups"] = follow_ups

        st.markdown(answer)

        # Highlight
        if st.session_state.highlight_on and chunks:
            with st.spinner("🔎 Finding key sentence…"):
                highlight = get_highlight(question, chunks, st.session_state.api_key, model_to_use)
            if highlight:
                st.markdown(
                    f'<div class="highlight-box">📌 <b>Key excerpt:</b> {highlight}</div>',
                    unsafe_allow_html=True,
                )

        # Timing — show ACTUAL model used, warn on fallback
        _fallback_note = (
            f"  *(fell back from `{model_to_use}`)*"
            if actual_model != model_to_use and actual_model != "none"
            else ""
        )
        st.caption(
            f"⏱ Search: {t_search:.1f}s · Answer: {t_llm:.1f}s · "
            f"Total: {t_search+t_llm:.1f}s · Model: `{actual_model}`{_fallback_note} · "
            f"Lang: {st.session_state.language}"
        )
        if actual_model != model_to_use and actual_model != "none":
            st.warning(
                f"⚠️ `{model_to_use}` was busy or unavailable; "
                f"answered with `{actual_model}` instead."
            )

        # Images
        images = _collect_images_for_chunks(chunks)
        _visual_keywords = {"diagram","chart","figure","image","picture","graph","visual","illustration","table","screenshot"}
        if not images and any(kw in question.lower() for kw in _visual_keywords):
            if chunks:
                images = _all_doc_images(chunks[0].get("doc_name",""))
        _show_images(images, expanded=True)
        _show_sources(chunks)

        # Feedback + Bookmark
        msg_idx = len(st.session_state.messages)
        existing_fb = st.session_state.feedback.get(msg_idx)
        fb1, fb2, bm_col = st.columns([1, 1, 3])
        if existing_fb:
            st.caption(f"You rated this: {'👍' if existing_fb == 'up' else '👎'}")
        else:
            if fb1.button("👍", key=f"up_{msg_idx}", help="Good answer"):
                st.session_state.feedback[msg_idx] = "up"
                st.rerun()
            if fb2.button("👎", key=f"dn_{msg_idx}", help="Poor answer"):
                st.session_state.feedback[msg_idx] = "down"
                st.rerun()
        if bm_col.button("🔖 Bookmark", key=f"bm_{msg_idx}"):
            st.session_state.bookmarks.append({
                "q":  question,
                "a":  answer[:300] + ("…" if len(answer) > 300 else ""),
                "ts": time.strftime("%H:%M"),
            })
            st.toast("Bookmarked! ✅")

        # Follow-up questions are stored in session state and rendered
        # in the MAIN FLOW below — not here — so they survive reruns correctly.

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "images": images[:3], "sources": chunks,
        "actual_model":    actual_model,
        "model_requested": model_to_use,
        "t_search":        round(t_search, 1),
        "t_llm":           round(t_llm, 1),
    })


def _handle_summarize(doc_name: str) -> None:
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
        doc_imgs = _all_doc_images(doc_name, limit=3)
        _show_images(doc_imgs, expanded=False)
    st.session_state.messages.append({
        "role": "assistant", "content": summary,
        "images": doc_imgs, "sources": [],
    })


# SIDEBAR
with st.sidebar:
    st.markdown("## 📄 DocChat")
    st.caption("AI-powered document assistant")

    _has_login = False
    try:
        _has_login = bool(st.secrets.get("USERS") or st.secrets.get("APP_PASSWORD"))
    except Exception:
        pass
    if _has_login:
        if st.button("🚪 Sign out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    st.divider()

    with st.expander("⚙️ Settings", expanded=False):
        new_dark = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
        if new_dark != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark
            st.rerun()

        lang = st.selectbox(
            "🌐 Answer language",
            options=["English", "Kannada", "Hindi"],
            index=["English", "Kannada", "Hindi"].index(st.session_state.language),
        )
        if lang != st.session_state.language:
            st.session_state.language = lang

        st.session_state.highlight_on = st.toggle(
            "📌 Show key excerpt", value=st.session_state.highlight_on
        )

        _sm_opts = ["Semantic", "Keyword", "Hybrid"]
        st.session_state.search_mode = st.radio(
            "🔍 Search mode",
            options=_sm_opts,
            index=_sm_opts.index(st.session_state.get("search_mode", "Semantic")),
            horizontal=True,
            help="Semantic=AI similarity · Keyword=exact words · Hybrid=both",
        )

    st.divider()

    _key_from_secrets = bool(st.secrets.get("GROQ_API_KEY", "")) if hasattr(st, "secrets") else False
    with st.expander("🔑 API Settings", expanded=not bool(st.session_state.api_key)):
        if _key_from_secrets:
            st.success("✅ API key configured (server-side)")
        else:
            new_key = st.text_input(
                "Groq API Key", type="password",
                value=st.session_state.api_key, placeholder="gsk_…",
                help="Get a free key at https://console.groq.com",
            )
            if new_key != st.session_state.api_key:
                st.session_state.api_key = new_key
            if st.session_state.api_key:
                st.success("✅ API key configured")
            else:
                st.info("👉 [Get free Groq key →](https://console.groq.com)")

        # Available Groq models — model ID → friendly label
        _MODEL_IDS = [
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "llama-3.1-8b-instant",
        ]
        _MODEL_LABELS = {
            "llama-3.1-8b-instant":                        "⚡ Llama 3.1 8B  (fastest)",
            "meta-llama/llama-4-scout-17b-16e-instruct":   "🦙 Llama 4 Scout 17B  (new · fast)",
            "llama-3.3-70b-versatile":                     "🧠 Llama 3.3 70B  (reliable quality)",
            "openai/gpt-oss-20b":                          "🚀 GPT-OSS 20B  (1000 TPS · smart)",
            "openai/gpt-oss-120b":                         "💯 GPT-OSS 120B  (best quality)",
            "qwen/qwen3-32b":                              "🌏 Qwen3 32B  (great multilingual)",
        }
        _cur_model = st.session_state.get("model", _MODEL_IDS[0])
        if _cur_model not in _MODEL_IDS:
            _cur_model = _MODEL_IDS[0]
        st.session_state.model = st.selectbox(
            "Model",
            options=_MODEL_IDS,
            index=_MODEL_IDS.index(_cur_model),
            format_func=lambda x: _MODEL_LABELS.get(x, x),
            help="🦙 Llama 4 = newest  ·  💯 GPT-OSS 120B = best quality  ·  ⚡ 8B = fastest",
        )

    st.divider()

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or Word files", type=["pdf", "docx", "doc"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded_files:
        if not st.session_state.api_key:
            st.warning("⚠️ Enter your Groq API key above first.")
        else:
            for f in uploaded_files:
                if f.name not in st.session_state.documents:
                    with st.status(f"⚙️ Processing **{f.name}**…", expanded=True) as status:
                        try:
                            st.write("📖 Reading file…")
                            raw = f.read()
                            st.write("✂️ Extracting text and images…")
                            result = (
                                extract_from_pdf(raw, f.name)
                                if f.name.lower().endswith(".pdf")
                                else extract_from_docx(raw, f.name)
                            )
                            n_chunks = len(result["chunks"])
                            n_imgs   = sum(len(v) for v in result["images"].values())
                            max_page = max((c["page"] for c in result["chunks"]), default=1)
                            st.write(f"🔢 Embedding {n_chunks} text chunks…")
                            st.session_state.vector_store.add_documents(result["chunks"], st.session_state.api_key)
                            st.session_state.documents[f.name]     = {"chunks": n_chunks, "images": n_imgs, "pages": max_page}
                            st.session_state.doc_images[f.name]    = result["images"]
                            st.session_state.doc_full_text[f.name] = "\n".join(c["text"] for c in result["chunks"])
                            status.update(label=f"✅ {f.name}  ({n_chunks} sections · {n_imgs} images)", state="complete")
                        except Exception as err:
                            status.update(label=f"❌ {f.name}: {err}", state="error")

    if st.session_state.documents:
        st.divider()
        st.markdown("### 📚 Loaded Documents")
        for doc_name, meta in list(st.session_state.documents.items()):
            short = doc_name if len(doc_name) <= 24 else doc_name[:21] + "…"
            with st.expander(f"📄 {short}"):
                st.caption(f"{meta['pages']} pages · {meta['chunks']} sections · {meta['images']} images")
                stats = _doc_stats(doc_name)
                st.caption(f"~{stats['word_count']:,} words")
                if stats["top_keywords"]:
                    kw_str = "  ·  ".join(f"`{w}`" for w, _ in stats["top_keywords"][:5])
                    st.caption(f"Top keywords: {kw_str}")
                c1, c2 = st.columns(2)
                if c1.button("📋 Summarize", key=f"sum__{doc_name}", use_container_width=True):
                    st.session_state["_summarize_doc"] = doc_name
                if c2.button("🗑 Remove", key=f"rm__{doc_name}", use_container_width=True):
                    del st.session_state.documents[doc_name]
                    del st.session_state.doc_images[doc_name]
                    del st.session_state.doc_full_text[doc_name]
                    old_chunks = st.session_state.vector_store.chunks
                    old_embeds = st.session_state.vector_store._embeddings
                    vs = VectorStore()
                    for i, c in enumerate(old_chunks):
                        if c.get("doc_name") != doc_name:
                            vs.chunks.append(c)
                            vs._embeddings.append(old_embeds[i])
                    if vs.chunks:
                        vs._rebuild_index()
                    st.session_state.vector_store = vs
                    st.rerun()

    st.divider()

    if st.session_state.bookmarks:
        with st.expander(f"🔖 Bookmarks ({len(st.session_state.bookmarks)})", expanded=False):
            for bm in reversed(st.session_state.bookmarks):
                st.markdown(
                    f'<div class="bookmark-card"><b>Q:</b> {bm["q"]}<br>'
                    f'<small>{bm["ts"]}</small><br><i>{bm["a"]}</i></div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear bookmarks", use_container_width=True):
                st.session_state.bookmarks = []
                st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()
    if col2.button("🔄 Reset all", use_container_width=True):
        for k in ["messages","documents","doc_images","doc_full_text","feedback","bookmarks"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else {}
        st.session_state.vector_store.clear()
        st.rerun()

    if st.session_state.messages:
        lines = []
        for m in st.session_state.messages:
            role = "You" if m["role"] == "user" else "DocChat"
            lines.append(f"[{role}]\n{m['content']}\n")
        st.download_button(
            "💾 Export chat (.txt)", data="\n".join(lines),
            file_name="docchat_chat_export.txt", mime="text/plain",
            use_container_width=True,
        )


# MAIN AREA
st.markdown("## 💬 DocChat")
st.caption("Upload documents in the sidebar · Then ask anything below")

if not st.session_state.documents and not st.session_state.messages:
    st.markdown(
        '<div class="welcome-card"><h2>👋 Welcome to DocChat</h2>'
        '<p>Your documents, made conversational with AI.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📄", "PDF & Word",   "Upload any PDF or DOCX — reports, manuals, contracts, research papers"),
        (c2, "🔍", "Smart Q&A",    "Semantic search finds the right passage; AI crafts a grounded answer with citations"),
        (c3, "🖼️", "Visual Aware", "Diagrams and images from the relevant pages are shown inline next to the answer"),
    ]:
        col.markdown(
            f'<div class="feature-card"><h3>{icon} {title}</h3><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
    st.info("👈  **Get started:** upload a document in the sidebar.")
    st.stop()

for _i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            _show_images(msg["images"], expanded=False)
        if msg.get("sources"):
            _show_sources(msg["sources"])
        # ── Persistent timing + feedback / bookmark buttons for assistant msgs ──
        if msg["role"] == "assistant":
            # Show timing/model info (stored in message dict so it survives reruns)
            if msg.get("actual_model"):
                _fb_note = (
                    f"  *(fell back from `{msg['model_requested']}`)*"
                    if msg.get("model_requested") and msg["actual_model"] != msg["model_requested"]
                    else ""
                )
                st.caption(
                    f"⏱ Search: {msg['t_search']:.1f}s · Answer: {msg['t_llm']:.1f}s · "
                    f"Total: {msg['t_search']+msg['t_llm']:.1f}s · "
                    f"Model: `{msg['actual_model']}`{_fb_note}"
                )
            _msg_q = (
                st.session_state.messages[_i - 1]["content"]
                if _i > 0 and st.session_state.messages[_i - 1]["role"] == "user"
                else ""
            )
            _existing_fb = st.session_state.feedback.get(_i)
            _fb1, _fb2, _bm_col = st.columns([1, 1, 3])
            if _existing_fb:
                _fb1.caption(f"{'👍' if _existing_fb == 'up' else '👎'} Rated")
            else:
                if _fb1.button("👍", key=f"up_{_i}", help="Good answer"):
                    st.session_state.feedback[_i] = "up"
                    st.rerun()
                if _fb2.button("👎", key=f"dn_{_i}", help="Poor answer"):
                    st.session_state.feedback[_i] = "down"
                    st.rerun()
            if _bm_col.button("🔖 Bookmark", key=f"bm_{_i}"):
                st.session_state.bookmarks.append({
                    "q":  _msg_q,
                    "a":  msg["content"][:300] + ("…" if len(msg["content"]) > 300 else ""),
                    "ts": time.strftime("%H:%M"),
                })
                st.toast("Bookmarked! ✅")

_summarize_doc: Optional[str] = st.session_state.pop("_summarize_doc", None)
if _summarize_doc:
    _handle_summarize(_summarize_doc)

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
        _row  = _suggestions[_row_start : _row_start + 3]
        _cols = st.columns(len(_row))
        for _col, _sug in zip(_cols, _row):
            if _col.button(_sug, use_container_width=True, key=f"sug__{_sug[:20]}"):
                st.session_state["_pending_q"] = _sug
                st.rerun()

if _pending_q:
    # Clear follow-ups before answering new question
    st.session_state["_follow_ups"] = []
    _handle_question(_pending_q)

# ── Follow-up questions (rendered in main flow so buttons survive reruns) ─────
_follow_ups = st.session_state.get("_follow_ups", [])
if _follow_ups and st.session_state.messages:
    st.markdown("**💡 Follow-up questions:**")
    fu_cols = st.columns(len(_follow_ups))
    for _fi, (_fc, _fq) in enumerate(zip(fu_cols, _follow_ups)):
        if _fc.button(_fq, key=f"fu_main_{_fi}_{_fq[:15]}", use_container_width=True):
            st.session_state["_pending_q"] = _fq
            st.session_state["_follow_ups"] = []
            st.rerun()

# ── Answer style buttons (Short / Long) ─────────────────────────────────────
if st.session_state.documents:
    _cur_style = st.session_state.get("answer_style", "Normal")
    _asc1, _asc2, _asc3 = st.columns([1, 1, 4])
    if _asc1.button(
        "📝 Short Answer", key="btn_style_short",
        use_container_width=True,
        type="primary" if _cur_style == "Short" else "secondary",
    ):
        st.session_state.answer_style = "Normal" if _cur_style == "Short" else "Short"
        st.rerun()
    if _asc2.button(
        "📚 Long Answer", key="btn_style_long",
        use_container_width=True,
        type="primary" if _cur_style == "Long" else "secondary",
    ):
        st.session_state.answer_style = "Normal" if _cur_style == "Long" else "Long"
        st.rerun()
    if _cur_style != "Normal":
        _asc3.caption(f"✅ **{_cur_style} Answer** mode active · click the button again to reset")

_placeholder = (
    "Ask a question about your documents…"
    if st.session_state.documents
    else "Upload a document in the sidebar first…"
)
_ready = bool(st.session_state.api_key and st.session_state.documents)

if user_input := st.chat_input(_placeholder, disabled=not _ready):
    st.session_state["_follow_ups"] = []
    _handle_question(user_input)
