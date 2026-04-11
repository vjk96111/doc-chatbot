"""
app.py - DocChat: AI-powered document chatbot
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud  (connect GitHub repo)
"""

import base64
import os
import time
from typing import List, Dict, Optional

# Load .env for local development (no-op if file is absent or on Streamlit Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from utils.document_processor import extract_from_pdf, extract_from_docx
from utils.vector_store import VectorStore, get_embed_model
from utils.llm_handler import (
    get_answer,
    get_document_summary,
    get_highlight,
    keyword_search,
    probe_key_limits,
)


# PAGE CONFIG
st.set_page_config(
    page_title="DocChat - AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",   # collapses on mobile, expanded on desktop
)

# Embedding model cached across all sessions
# Loaded BEFORE the login gate so it warms up while the user types credentials.
# @st.cache_resource means it only runs once per server process — never again.
@st.cache_resource(show_spinner=False)
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
    btn_bg  = "#2e2e4a" if dark else "#ffffff"
    btn_bd  = "#5a5a8a" if dark else "#ced4da"
    code_fg = "#a8d8ff" if dark else "#d63384"
    hl_bg   = "#3a3a1a" if dark else "#fffde7"
    return f"""
    <style>
    /* ── Main app background ── */
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {{
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
    [data-testid="stSidebar"] span:not([data-baseweb]),
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small {{
        color: {text} !important;
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
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
    .stTextArea > div > div > textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{
        background-color: {inp_bg} !important;
        color: {text} !important;
        border-color: {card_bd} !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: {cap} !important;
    }}
    /* ── BaseUI Selectbox ── */
    [data-baseweb="select"] > div:first-child {{
        background-color: {inp_bg} !important;
        border-color: {card_bd} !important;
        color: {text} !important;
    }}
    [data-baseweb="select"] [data-id="select-value"],
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: {text} !important;
    }}
    [data-baseweb="select"] svg {{
        fill: {text} !important;
    }}
    /* ── BaseUI Dropdown popup (opened selectbox) ── */
    [data-baseweb="popover"],
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {{
        background-color: {card_bg} !important;
        border-color: {card_bd} !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"],
    [data-baseweb="option"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="option"]:hover {{
        background-color: {card_bd} !important;
    }}
    /* ── Buttons ── */
    .stButton > button,
    [data-testid^="baseButton"] {{
        background-color: {btn_bg} !important;
        color: {text} !important;
        border-color: {btn_bd} !important;
    }}
    .stButton > button:hover,
    [data-testid^="baseButton"]:hover {{
        background-color: {card_bd} !important;
        border-color: {text} !important;
    }}
    /* Keep primary buttons their accent color */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"] {{
        background-color: #e05252 !important;
        color: #ffffff !important;
        border-color: #e05252 !important;
    }}
    .stButton > button p,
    [data-testid^="baseButton"] p {{
        color: {text} !important;
    }}
    /* ── Expanders ── */
    [data-testid="stExpander"] details {{
        background-color: {card_bg} !important;
        border: 1px solid {card_bd} !important;
        border-radius: 8px;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] details > div {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}
    [data-testid="stExpander"] summary svg {{
        fill: {text} !important;
    }}
    /* ── Toggle ── */
    [data-testid="stToggle"] label {{
        color: {text} !important;
    }}
    /* ── Radio ── */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p {{
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
    /* ── Info / Warning / Error boxes ── */
    [data-testid="stAlert"] {{
        background-color: {card_bg} !important;
        border-color: {card_bd} !important;
    }}
    [data-testid="stAlert"] p {{
        color: {text} !important;
    }}
    /* ── Dividers ── */
    hr {{ border-color: {card_bd}; }}
    /* ── Status widget ── */
    [data-testid="stStatusWidget"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}
    /* ── Top toolbar / header bar ── */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {{
        background-color: {bg} !important;
        border-bottom: 1px solid {card_bd} !important;
    }}
    header[data-testid="stHeader"] *,
    [data-testid="stHeader"] a,
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg {{
        color: {text} !important;
        fill: {text} !important;
    }}
    /* ── Deploy/Fork/toolbar buttons ── */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stDeployButton"] {{
        background-color: {bg} !important;
    }}
    [data-testid="stDeployButton"] button {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border-color: {card_bd} !important;
    }}
    /* Hide the deploy bar in production if desired; keep visible but themed */
    .reportview-container .main footer {{ visibility: hidden; }}
    /* ── File uploader ── */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > div {{
        background-color: {inp_bg} !important;
        border: 1.5px dashed {card_bd} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {{
        color: {cap} !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border-color: {card_bd} !important;
    }}
    /* Uploaded file chip inside uploader */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border-color: {card_bd} !important;
    }}
    /* ── stStatus / st.status widget (processing) ── */
    [data-testid="stStatusContainer"],
    [data-testid="stStatusContainer"] > div {{
        background-color: {card_bg} !important;
        border-color: {card_bd} !important;
        color: {text} !important;
    }}
    /* ── Toast notifications ── */
    [data-testid="stToast"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border: 1px solid {card_bd} !important;
    }}
    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {{
        background-color: {btn_bg} !important;
        color: {text} !important;
        border-color: {btn_bd} !important;
    }}
    /* ── Metric widget ── */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border-color: {card_bd} !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        color: {text} !important;
    }}
    /* ── Scrollbar ── */
    ::-webkit-scrollbar-track {{ background: {bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {card_bd}; border-radius: 4px; }}
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

    /* ══════════════════════════════════════════════════════
       MOBILE  (≤ 768 px)
       Streamlit auto-collapses the sidebar — initial_sidebar_state="auto"
       These rules handle everything inside the main content area.
       ══════════════════════════════════════════════════════ */
    @media only screen and (max-width: 768px) {{

        /* Main content: use full viewport, leave room for fixed chat input */
        .main .block-container {{
            padding: 0.5rem 0.7rem 5.5rem 0.7rem !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }}

        /* Sidebar overlay: full-height, 85% width when open */
        [data-testid="stSidebar"] {{
            min-width: 85vw !important;
            max-width: 92vw !important;
        }}

        /* Typography: scale down */
        h1 {{ font-size: 1.35rem !important; }}
        h2 {{ font-size: 1.15rem !important; }}
        h3 {{ font-size: 1.0rem  !important; }}
        p, li {{ line-height: 1.55 !important; }}

        /* Welcome / feature cards */
        .welcome-card {{
            padding: 1.2rem 0.9rem !important;
            border-radius: 10px !important;
        }}
        .welcome-card h2 {{ font-size: 1.35rem !important; }}
        .welcome-card p  {{ font-size: 0.9rem  !important; }}
        .feature-card {{ height: auto !important; margin-bottom: 0.4rem; }}

        /* Horizontal blocks: wrap to 2-per-row on tablet/large phone */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
        }}
        [data-testid="column"] {{
            min-width: calc(50% - 0.35rem) !important;
            flex-grow: 1 !important;
            box-sizing: border-box !important;
        }}

        /* Chat messages: compact */
        [data-testid="stChatMessage"] {{
            padding: 0.5rem 0.65rem !important;
            margin-bottom: 0.35rem !important;
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {{
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
        }}

        /* Chat input: bigger touch target, no horizontal overflow */
        [data-testid="stChatInput"] textarea {{
            font-size: 1rem !important;
            min-height: 46px !important;
        }}
        [data-testid="stBottom"] {{
            padding: 0.3rem 0.5rem !important;
        }}
        [data-testid="stBottom"] > div {{
            max-width: 100% !important;
        }}

        /* All buttons: large touch targets */
        .stButton > button,
        [data-testid^="baseButton"] {{
            min-height: 44px !important;
            font-size: 0.88rem !important;
        }}
        div[data-testid="stHorizontalBlock"] button {{
            font-size: 0.82rem !important;
            padding: 0.45rem 0.25rem !important;
        }}

        /* Expanders */
        [data-testid="stExpander"] summary {{
            font-size: 0.88rem !important;
            padding: 0.45rem 0.65rem !important;
        }}
        [data-testid="stExpander"] details > div {{
            padding: 0.35rem 0.5rem !important;
        }}

        /* Captions */
        .stCaption,
        [data-testid="stCaptionContainer"] p {{
            font-size: 0.7rem !important;
        }}

        /* Bookmark cards */
        .bookmark-card {{
            padding: 0.55rem !important;
            font-size: 0.87rem !important;
        }}

        /* Code: wrap not scroll */
        code, pre {{
            white-space: pre-wrap !important;
            word-break: break-word !important;
            font-size: 0.8rem !important;
        }}

        /* Tables: scroll horizontally instead of overflowing viewport */
        [data-testid="stChatMessage"] table,
        [data-testid="stMarkdownContainer"] table,
        .stMarkdown table,
        .element-container table {{
            display: block !important;
            overflow-x: auto !important;
            max-width: 100% !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.8rem !important;
        }}
        [data-testid="stChatMessage"] th,
        [data-testid="stChatMessage"] td,
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {{
            white-space: nowrap !important;
            padding: 0.3rem 0.5rem !important;
        }}

        /* Images fill width */
        [data-testid="stImage"] img {{
            max-width: 100% !important;
            height: auto !important;
        }}

        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {{ margin: 0.4rem 0 !important; }}
    }}

    /* Very small phones (≤ 480 px): single-column everything */
    @media only screen and (max-width: 480px) {{
        .main .block-container {{
            padding: 0.35rem 0.45rem 5.5rem 0.45rem !important;
        }}
        /* Stack ALL columns to full-width */
        [data-testid="column"] {{
            min-width: 100% !important;
            max-width: 100% !important;
            flex: none !important;
        }}
        h2 {{ font-size: 1.05rem !important; }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {{
            font-size: 0.85rem !important;
        }}
    }}
    </style>
    """


# LOGIN GATE
# Kick off model loading BEFORE the login check so it warms up in background
# while the user is on the login page. By the time they sign in, it's ready.
_load_embed_model()

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
    # Prefer st.secrets (Streamlit Cloud), then os.environ (local .env via load_dotenv)
    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not secret_key:
        secret_key = os.environ.get("GROQ_API_KEY", "")
    # Read extra API keys from secrets/env (GROQ_API_KEY_2, GROQ_API_KEY_3)
    extra_keys_from_secrets: List[str] = []
    for suffix in ["_2", "_3", "_4"]:
        k = ""
        try:
            k = st.secrets.get(f"GROQ_API_KEY{suffix}", "")
        except Exception:
            pass
        if not k:
            k = os.environ.get(f"GROQ_API_KEY{suffix}", "")
        if k:
            extra_keys_from_secrets.append(k)

    defaults: Dict = {
        "vector_store":  VectorStore(),
        "messages":      [],
        # conv_history is separate from messages (display list).
        # It persists across Clear Chat; resets only on new doc upload or language change.
        "conv_history":  [],
        "documents":     {},
        "doc_images":    {},
        "doc_full_text": {},
        "api_key":       secret_key,
        # Extra API keys for automatic rotation when primary key hits daily limit
        "extra_api_keys": extra_keys_from_secrets,
        "model":         "openai/gpt-oss-120b",
        "language":      "English",
        "dark_mode":     True,
        "highlight_on":  False,
        "answer_style":  "Normal",   # "Short", "Normal", or "Long"
        "_lang_version": 0,           # increments on language change to bust history
        "bookmarks":     [],
        "feedback":      {},
        "search_mode":   "Semantic",
        "_follow_ups":   [],   # follow-up questions from last answer
        # Per-session token usage: {key_index: {model: {"prompt": int, "completion": int, "total": int}}}
        "api_usage":     {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if secret_key and not st.session_state.get("api_key"):
        st.session_state["api_key"] = secret_key


_init_state()
# Inject mobile viewport meta tag (Streamlit omits width=device-width by default)
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">',
    unsafe_allow_html=True,
)
st.markdown(_get_css(st.session_state.dark_mode), unsafe_allow_html=True)
# Note: _load_embed_model() was already called before _check_login() — no-op here (cached)


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
    with st.expander("📚 Sources used", expanded=False):
        for i, c in enumerate(unique, start=1):
            score = c.get("relevance_score")
            score_str = f"  *(relevance: {score:.2f})*" if score is not None else ""
            doc   = c.get("doc_name", "Unknown")
            src   = c.get("source", "")
            page  = c.get("page")
            page_str = f"  ·  p.{page}" if page else ""
            excerpt = c.get("text", "")[:160].replace("\n", " ")
            if len(c.get("text", "")) > 160:
                excerpt += "…"
            st.markdown(
                f"**{i}.** **{doc}** — {src}{page_str}{score_str}  \n"
                f"> {excerpt}"
            )


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


# Patterns that indicate user wants the document STRUCTURE (TOC / section overview)
# NOTE: "all topics", "all sections" etc. are intentionally NOT here — they are
# handled by _is_list_all_query which scans actual document text rather than
# returning only chunk metadata (page numbers), which is useless for PDFs.
_TOC_PATTERNS = [
    "all contents", "all parts", "table of contents", "toc",
    "what does this document", "what is in this document", "what is in the document",
    "what does the document contain", "what does the document cover",
    "entire document", "full document", "whole document",
    "full overview", "complete overview", "overview of the document",
    "overview of this document", "everything in", "everything covered",
    "contents of this document", "contents of the document",
]

# Patterns that indicate user wants ALL instances of a SPECIFIC item type
_LIST_ALL_PATTERNS = [
    "list all", "list every", "show all", "show every",
    "give me all", "give all", "all the ",
    "complete list", "full list",
    "list them", "list those", "list these",
    "can you list", "could you list", "please list",
    "enumerate all", "enumerate every",
    "names of all", "names of the",
    # structural queries that need actual content, not just page metadata
    "all topics", "all sections", "all chapters", "all headings", "all heading",
    "what topics", "what subjects", "what areas", "what chapters",
    "all the sections", "all the topics", "list sections",
]

def _is_toc_query(question: str) -> bool:
    """Return True if the question asks for the document structure / table of contents."""
    q = question.lower().strip()
    return any(p in q for p in _TOC_PATTERNS)

def _is_list_all_query(question: str) -> bool:
    """Return True if the question asks for all instances of a specific item (e.g. all measures)."""
    q = question.lower().strip()
    return any(p in q for p in _LIST_ALL_PATTERNS)

# Stopwords to strip when extracting the target noun from "list all X" queries
_LIST_ALL_STOPWORDS = {
    "list", "all", "every", "show", "give", "me", "the",
    "complete", "full", "a", "an", "of", "in", "from",
    "document", "this", "that", "these", "those",
}

def _extract_list_target(question: str) -> str:
    """
    Extract the target noun from a 'list all X' query.
    e.g. 'list all 166 measures' -> 'measures'
         'show all DAX tables'   -> 'dax tables'
         'full list of columns'  -> 'columns'
    Returns empty string if nothing useful found.
    """
    import re
    q = question.lower().strip()
    # Remove leading pattern words and numbers, keep the noun phrase
    words = re.sub(r'\d+', '', q).split()
    meaningful = [w for w in words if w not in _LIST_ALL_STOPWORDS and len(w) > 2]
    return " ".join(meaningful[-3:]) if meaningful else ""   # last 1-3 content words

# Generic meta-nouns that mean "everything in the document" — scanning for
# these keywords in chunk text would be useless, so we use a compact topic scan.
_META_NOUNS = {
    "topics", "topic", "questions", "question", "sections", "section",
    "chapters", "chapter", "headings", "heading", "parts", "part",
    "subjects", "subject", "areas", "area", "items", "item",
    "contents", "content",
}

# How many chars to take from the start of each chunk for the topic scan.
# 300 chars is enough to capture 1-2 numbered headings per page.
_TOPIC_SCAN_CHARS = 300

def _build_topic_scan_chunks(all_chunks: list) -> list:
    """
    For generic 'list all topics/sections/questions' queries:
    extract only the first _TOPIC_SCAN_CHARS characters of each unique
    page/section and combine into ONE compact synthetic chunk.
    The LLM sees all headings without the full answer bodies → 5-10x faster
    than sending all 80k chars while still giving a correct answer.
    """
    seen_pages: set = set()
    lines: list = []
    for c in all_chunks:
        page_key = (c.get("doc_name", ""), c.get("source", ""))
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        snippet = c["text"][:_TOPIC_SCAN_CHARS].replace("\n", " ").strip()
        if snippet:
            lines.append(f"[{c.get('doc_name', '?')} | {c.get('source', '')}]  {snippet}")
    combined = (
        "DOCUMENT HEADINGS/TOPICS SCAN "
        "(first snippet of each page — enough to identify all section titles):\n\n"
        + "\n\n".join(lines)
    )
    return [{"text": combined, "source": "Topics Scan",
             "doc_name": "All Documents", "page": 0}]


def _scan_all_chunks_for(noun: str, all_chunks: list) -> list:
    """
    Return chunks that contain any word from *noun*.
    Ordered: chunks whose section/source name contains the noun first,
    then other containing chunks — so the most relevant pages come first.
    Prompt cap in llm_handler (80k chars) prevents token blowout.

    Special case: if the noun is a generic meta-word (topics, sections, chapters…)
    build a compact topic-scan instead of sending all 80k chars — much faster.
    """
    if not noun:
        return all_chunks
    words = [w for w in noun.split() if len(w) > 2]
    if not words:
        return all_chunks

    # Generic meta-noun → compact heading scan (fast, correct)
    if all(w in _META_NOUNS for w in words):
        return _build_topic_scan_chunks(all_chunks)

    primary, secondary = [], []
    for c in all_chunks:
        text_lower = c["text"].lower()
        source_lower = c.get("source", "").lower()
        in_source = any(w in source_lower for w in words)
        in_text   = any(w in text_lower   for w in words)
        if in_source:
            primary.append(c)
        elif in_text:
            secondary.append(c)

    # Fallback: if keyword search returns < 3 chunks, the noun may not appear
    # verbatim in the text — return all chunks so nothing is missed.
    result = primary + secondary
    return result if len(result) >= 3 else all_chunks

def _is_comprehensive_query(question: str) -> bool:
    """Return True if the question is a TOC or targeted full-list query."""
    return _is_toc_query(question) or _is_list_all_query(question)

def _build_toc_chunks(all_chunks: list) -> list:
    """
    Build a compact Table-of-Contents context from chunk metadata.
    Returns a SINGLE synthetic chunk containing only section titles —
    no full text — so the prompt stays tiny and the LLM responds fast.
    """
    seen: set = set()
    lines: list = []
    for c in all_chunks:
        key = (c.get("doc_name", ""), c.get("source", ""))
        if key not in seen:
            seen.add(key)
            lines.append(f"• [{c.get('doc_name','?')}]  {c.get('source','')}")
    toc_text = "DOCUMENT STRUCTURE (all sections found):\n" + "\n".join(lines)
    return [{"text": toc_text, "source": "Table of Contents",
              "doc_name": "All Documents", "page": 0}]


def _deduplicate_chunks(chunks: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """Remove near-duplicate chunks (>= threshold text similarity). Keeps first occurrence."""
    from difflib import SequenceMatcher
    unique: List[Dict] = []
    for chunk in chunks:
        is_dup = False
        for kept in unique:
            ratio = SequenceMatcher(None, chunk["text"], kept["text"]).ratio()
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
    return unique


# CORE ACTIONS
def _handle_question(question: str, model_override: str = None, answer_style: str = None) -> None:
    # Pass the full conversation history stored in conv_history.
    # This is separate from `messages` (display list) so it survives Clear Chat.
    chat_history = st.session_state.get("conv_history", []).copy()

    # If the user typed a vague follow-up, use the last real question for retrieval
    search_query = _resolve_search_query(question)
    # Use model_override or the user's selected model
    model_to_use = model_override or st.session_state.model
    # Use explicitly passed style or the current session state style
    _style = answer_style or st.session_state.get("answer_style", "Normal")

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "images": [], "sources": [],
         "lang_version": st.session_state.get("_lang_version", 0)}
    )

    with st.chat_message("assistant"):
        # Retrieval
        with st.spinner("🔍 Searching your documents…"):
            t0 = time.time()
            mode = st.session_state.search_mode
            all_chunks = st.session_state.vector_store.chunks

            if _is_toc_query(search_query):
                # Document structure / TOC: send ONLY section titles (tiny prompt → fast)
                chunks = _build_toc_chunks(all_chunks)
            elif _is_list_all_query(search_query):
                # "list all X": scan EVERY chunk that mentions the target noun
                # so that enumerated data (e.g. measures on pages 59-70) is never missed
                target = _extract_list_target(search_query)
                chunks = _scan_all_chunks_for(target, all_chunks)
            elif mode == "Keyword":
                chunks = keyword_search(search_query, all_chunks, top_k=6)
            elif mode == "Hybrid":
                sem = st.session_state.vector_store.search(search_query, st.session_state.api_key, top_k=6)
                kw  = keyword_search(search_query, all_chunks, top_k=6)
                seen_texts = set()
                chunks = []
                for c in sem + kw:
                    if c["text"] not in seen_texts:
                        seen_texts.add(c["text"])
                        chunks.append(c)
                chunks = chunks[:8]
            else:
                chunks = st.session_state.vector_store.search(search_query, st.session_state.api_key, top_k=6)
            # Deduplicate near-identical chunks (≥ 90% text similarity)
            chunks = _deduplicate_chunks(chunks)
            t_search = time.time() - t0

        # Usage callback: accumulates per-key per-model token counts in session state
        def _on_usage(usage_dict: dict, model_name: str) -> None:
            store = st.session_state.setdefault("api_usage", {})
            ki = usage_dict.get("key_index", 0)
            key_store = store.setdefault(ki, {})
            m_store = key_store.setdefault(model_name, {"prompt": 0, "completion": 0, "total": 0})
            m_store["prompt"]     += usage_dict.get("prompt_tokens", 0)
            m_store["completion"] += usage_dict.get("completion_tokens", 0)
            m_store["total"]      += usage_dict.get("total_tokens", 0)

        # Answer — follow-up questions come back in the same call, no extra delay
        with st.spinner("💬 Generating answer…"):
            t1 = time.time()
            answer, follow_ups, actual_model = get_answer(
                question, chunks,
                st.session_state.api_key,
                model_to_use,
                st.session_state.language,
                include_followups=True,
                chat_history=chat_history if chat_history else None,
                answer_style=_style,
                # comprehensive=True ONLY for TOC queries — not for "list all X" queries
                comprehensive=_is_toc_query(search_query),
                list_all=_is_list_all_query(search_query),
                extra_keys=st.session_state.get("extra_api_keys") or None,
                usage_callback=_on_usage,
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
        "lang_version":    st.session_state.get("_lang_version", 0),
    })
    # Append both turns to conv_history (persistent, survives Clear Chat)
    st.session_state.conv_history.append({"role": "user",      "content": question})
    st.session_state.conv_history.append({"role": "assistant", "content": answer})


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
            key="lang_selectbox",
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            # Language change: reset conv_history so old-language turns aren't sent
            st.session_state["conv_history"] = []
            st.session_state["_lang_version"] = st.session_state.get("_lang_version", 0) + 1
            st.rerun()

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
        st.caption("✏️ Answer length")
        _cur_style_sb = st.session_state.get("answer_style", "Normal")
        _sb1, _sb2, _sb3 = st.columns(3)
        for _col, _label, _val, _icon in [
            (_sb1, "Short",  "Short",  "📝"),
            (_sb2, "Normal", "Normal", "💬"),
            (_sb3, "Long",   "Long",   "📚"),
        ]:
            if _col.button(
                f"{_icon} {_label}",
                key=f"sidebar_style_{_val}",
                use_container_width=True,
                type="primary" if _cur_style_sb == _val else "secondary",
            ):
                st.session_state.answer_style = _val
                st.rerun()

    st.divider()

    _key_from_secrets = bool(
        (st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "")
        or os.environ.get("GROQ_API_KEY", "")
    )
    with st.expander("🔑 API Settings", expanded=not bool(st.session_state.api_key)):
        if _key_from_secrets:
            st.success("✅ API key configured (server-side)")
            # Show how many backup keys are loaded from secrets
            _n_extra_secrets = len([
                1 for suffix in ["_2","_3","_4"]
                if (st.secrets.get(f"GROQ_API_KEY{suffix}","") if hasattr(st,"secrets") else "")
                   or os.environ.get(f"GROQ_API_KEY{suffix}","")
            ])
            if _n_extra_secrets:
                st.caption(f"✅ {_n_extra_secrets} backup key(s) loaded from server secrets")
        else:
            new_key = st.text_input(
                "Groq API Key (primary)", type="password",
                value=st.session_state.api_key, placeholder="gsk_…",
                help="Get a free key at https://console.groq.com",
            )
            if new_key != st.session_state.api_key:
                st.session_state.api_key = new_key
            if st.session_state.api_key:
                st.success("✅ Primary key configured")
            else:
                st.info("👉 [Get free Groq key →](https://console.groq.com)")

            # Backup keys for automatic rotation when daily limit is hit
            st.caption(
                "💡 **Backup keys** — if the primary key hits its daily limit, "
                "the app automatically tries these in order. "
                "Each free Groq account gets ~100k tokens/day."
            )
            _cur_extras = st.session_state.get("extra_api_keys", [])
            # Show up to 3 backup key slots
            _new_extras: List[str] = []
            for _slot in range(1, 4):
                _existing_val = _cur_extras[_slot - 1] if _slot - 1 < len(_cur_extras) else ""
                _bk = st.text_input(
                    f"Backup key #{_slot}", type="password",
                    value=_existing_val, placeholder="gsk_…",
                    key=f"extra_key_{_slot}",
                )
                if _bk:
                    _new_extras.append(_bk)
            if _new_extras != _cur_extras:
                st.session_state.extra_api_keys = _new_extras
            if _new_extras:
                st.caption(f"✅ {len(_new_extras)} backup key(s) ready")

        # ── API Usage / Rate Limit Dashboard ──────────────────────────────
        st.markdown("#### 📊 API Usage")
        _all_keys = [st.session_state.api_key] + (st.session_state.get("extra_api_keys") or [])
        _all_keys = [k for k in _all_keys if k]   # remove blanks

        # Session token usage (accumulated from real calls this session)
        _session_usage = st.session_state.get("api_usage", {})
        if _session_usage:
            with st.expander("🪙 Session token usage", expanded=False):
                for ki, models_data in sorted(_session_usage.items()):
                    key_label = "Primary key" if ki == 0 else f"Backup key #{ki}"
                    st.markdown(f"**{key_label}**")
                    for model_name, counts in sorted(models_data.items()):
                        short_model = model_name.split("/")[-1]
                        st.caption(
                            f"`{short_model}` — "
                            f"↑ {counts['prompt']:,} prompt  "
                            f"↓ {counts['completion']:,} completion  "
                            f"= **{counts['total']:,} total tokens**"
                        )
        else:
            st.caption("No API calls made yet this session.")

        # Live rate-limit probe button
        if _all_keys:
            if st.button("🔍 Check live rate limits", use_container_width=True,
                         help="Makes a 1-token probe call per key × model to read Groq's "
                              "rate-limit headers (requests/tokens remaining this minute)."):
                _probe_models = [
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                ]
                with st.spinner("Probing Groq rate limits…"):
                    _probe_results = probe_key_limits(_all_keys, _probe_models)
                st.session_state["_probe_results"] = _probe_results

        _probe_results = st.session_state.get("_probe_results", [])
        if _probe_results:
            # Group by key_index for a clean display
            _by_key: dict = {}
            for r in _probe_results:
                _by_key.setdefault(r["key_index"], []).append(r)
            for ki, rows in sorted(_by_key.items()):
                key_label = "Primary key" if ki == 0 else f"Backup key #{ki}"
                key_preview = rows[0]["key_preview"]
                with st.expander(f"🔑 {key_label}  `{key_preview}`", expanded=(ki == 0)):
                    for r in rows:
                        short_model = r["model"].split("/")[-1]
                        if r.get("error"):
                            st.error(f"`{short_model}` — ⚠️ {r['error']}")
                            continue
                        # Requests bar
                        req_pct = r.get("req_used_pct")
                        req_rem = r.get("req_remaining")
                        req_lim = r.get("req_limit")
                        req_rst = r.get("req_reset", "?")
                        # Tokens bar
                        tok_pct = r.get("tok_used_pct")
                        tok_rem = r.get("tok_remaining")
                        tok_lim = r.get("tok_limit")
                        tok_rst = r.get("tok_reset", "?")
                        st.markdown(f"**`{short_model}`**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Requests / min")
                            if req_pct is not None:
                                st.progress(min(req_pct / 100, 1.0))
                                st.caption(f"{req_rem:,} / {req_lim:,} remaining  ·  resets {req_rst}")
                            else:
                                st.caption("n/a")
                        with col2:
                            st.caption("Tokens / min")
                            if tok_pct is not None:
                                st.progress(min(tok_pct / 100, 1.0))
                                st.caption(f"{tok_rem:,} / {tok_lim:,} remaining  ·  resets {tok_rst}")
                            else:
                                st.caption("n/a")
        # ──────────────────────────────────────────────────────────────────

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
                            # New document uploaded → reset conversation history to start fresh
                            st.session_state.conv_history = []
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
        # Clear display messages, feedback, and conversation history — document stays loaded
        st.session_state.messages      = []
        st.session_state.conv_history  = []
        st.session_state.feedback      = {}
        st.rerun()
    if col2.button("🔄 Reset all", use_container_width=True):
        for k in ["messages","conv_history","documents","doc_images","doc_full_text","feedback","bookmarks"]:
            st.session_state[k] = [] if isinstance(st.session_state.get(k, []), list) else {}
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
    if _cur_style != "Normal":
        st.caption(f"✅ **{_cur_style} Answer** mode · change in ⚙️ Settings → Answer length")

_placeholder = (
    "Ask a question about your documents…"
    if st.session_state.documents
    else "Upload a document in the sidebar first…"
)
_ready = bool(st.session_state.api_key and st.session_state.documents)

if user_input := st.chat_input(_placeholder, disabled=not _ready):
    st.session_state["_follow_ups"] = []
    _handle_question(user_input)
