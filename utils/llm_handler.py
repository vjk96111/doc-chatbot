"""
llm_handler.py
Handles all calls to Groq: Q&A, summarisation, follow-up questions,
highlight extraction, and language-aware responses.
"""

import re
import time
from typing import List, Dict

from groq import Groq, RateLimitError, BadRequestError

# Fallback model order — fast 8b is always the last-resort safety net
_FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 — fast + smart
    "openai/gpt-oss-20b",                          # GPT-OSS 20B — very fast
    "llama-3.3-70b-versatile",                     # reliable 70b
    "llama-3.1-8b-instant",                        # last resort: always works
]

# Safe character limit per prompt
_MAX_PROMPT_CHARS = 80_000

# Separator used to split answer from follow-up questions in combined response
_FU_SEP = "---FU---"

# Per-request answer-style instructions injected into the user prompt
_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "Short":  " Keep your answer SHORT and CONCISE — 2–4 sentences maximum, core point only.",
    "Normal": "",
    "Long":   " Give a DETAILED and COMPREHENSIVE answer — cover all relevant aspects thoroughly with examples from the document.",
}

# ─────────────────────────────────── prompts ─────────────────────────────────

_SYSTEM_PROMPT = (
    "You are DocChat, a precise and helpful assistant that answers questions "
    "strictly based on content retrieved from uploaded documents.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context snippets — NEVER use general knowledge.\n"
    "2. If the answer is not present in the context, respond with exactly: "
    "\"This information is not found in the uploaded document.\"\n"
    "   IMPORTANT: Do NOT describe the document structure, page count, or file name "
    "   as a substitute for an actual content answer. If content is missing, say "
    "   the exact phrase above — nothing else.\n"
    "3. On follow-up questions, always refer back to the previous answer and "
    "filter or extend it — do NOT start fresh from the full context.\n"
    "4. When the answer contains multiple items, always use a numbered list or "
    "bullet points — never write them as a flat paragraph.\n"
    "5. Always cite the source document name and page/section number wherever possible.\n"
    "6. If a context snippet mentions a diagram, figure, chart, or table, "
    "acknowledge it — the image will be displayed automatically in the chat.\n"
    "7. Be concise but complete. Do NOT hallucinate facts. Do NOT fabricate page "
    "   ranges or document structure when you have not read the actual content.\n"
    "8. If LANGUAGE appears in the prompt, use ONLY that language for EVERY part "
    "of your response including follow-up questions."
)

def _make_system_prompt(language: str = "English", include_fu: bool = False) -> str:
    """Build a system prompt that bakes the language directly in (more reliable)."""
    lang_note = (
        f"\n\nCRITICAL: You MUST respond ENTIRELY in {language}. "
        "Every sentence, every bullet point, every follow-up question "
        f"MUST be written in {language}. Do not use English."
        if language != "English" else ""
    )
    fu_note = (
        f"\n\nAfter your answer, output a single line containing exactly '---FU---', "
        f"then write exactly 3 short follow-up questions "
        + (f"in {language} " if language != "English" else "")
        + "numbered 1. 2. 3. (each under 12 words)."
        if include_fu else ""
    )
    return _SYSTEM_PROMPT + lang_note + fu_note

_SUMMARY_PROMPT_TEMPLATE = (
    'Provide a structured summary of the document "{doc_name}".\n\n'
    "DOCUMENT CONTENT (may be truncated):\n{text}\n\n"
    "Include:\n"
    "1. **Overview** — 2-3 sentence description of the document's purpose.\n"
    "2. **Key Topics** — bullet list of main subjects covered.\n"
    "3. **Main Findings / Conclusions** — key takeaways or decisions.\n"
    "4. **Notable Data / Visuals** — mention any statistics, tables, or "
    "diagrams referenced in the text."
)


# ─────────────────────────────────── helpers ─────────────────────────────────

def _build_qa_prompt(
    question: str,
    chunks: List[Dict],
    language: str = "English",
    style: str = "Normal",
    include_fu: bool = False,
    comprehensive: bool = False,
    list_all: bool = False,
) -> str:
    if not chunks:
        return (
            f"The user asked: {question}\n\n"
            "No relevant context was found in the uploaded documents. "
            "You MUST respond with exactly: "
            "\"This information is not found in the uploaded document.\""
        )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc_name", "Unknown")
        src = chunk.get("source", "")
        context_blocks.append(
            f"[Context {i} | {doc} | {src}]\n{chunk['text']}"
        )

    context = "\n\n" + ("─" * 60) + "\n\n".join(context_blocks)
    style_instruction = _STYLE_INSTRUCTIONS.get(style, "")
    # Language instruction goes FIRST (most prominent placement for model compliance)
    lang_instruction = (
        f"\nIMPORTANT: Write your ENTIRE response in {language} only. "
        f"Every word must be in {language}.\n"
        if language != "English" else ""
    )
    # FU in user prompt reinforces system prompt instruction
    _fu_lang = f"in {language} " if language != "English" else ""
    fu_instruction = (
        f"\nAfter your answer, output exactly '---FU---' on its own line, "
        f"then 3 follow-up questions {_fu_lang}"
        "numbered 1. 2. 3. (max 12 words each)."
        if include_fu else ""
    )
    comprehensive_instruction = (
        "\nINSTRUCTION: The context above contains a compact table of contents extracted "
        "from the document. List EVERY section and heading you can find — do not omit any. "
        "Organise the output as a structured numbered or bulleted list with brief descriptions."
        if comprehensive else ""
    )
    list_all_instruction = (
        "\nINSTRUCTION: The context above contains ALL document chunks that mention "
        "the requested items. Extract and enumerate EVERY individual item name you find "
        "across ALL context blocks \u2014 do not summarise, do not skip any. "
        "Produce a complete numbered or bulleted list."
        if list_all else ""
    )
    prompt = (
        f"{lang_instruction}"
        f"RETRIEVED CONTEXT FROM DOCUMENTS:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Answer the question based on the context above and cite sources."
        f"{comprehensive_instruction}{list_all_instruction}{style_instruction}{fu_instruction}"
    )
    return prompt[:_MAX_PROMPT_CHARS]


def _groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _call_with_retry(
    client: Groq,
    model_name: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    extra_keys: List[str] = None,
) -> tuple:
    """
    Call Groq with exponential backoff + automatic model fallback on rate limit.
    Tries up to 3 times per model, then falls back to a smaller model.
    If extra_keys are provided, rotates to the next API key when all models
    are exhausted on the current key (daily usage limit reached).
    Returns (content, actual_model_used).
    """
    models_to_try = [model_name] + [m for m in _FALLBACK_MODELS if m != model_name]
    # Build list of clients to try: primary first, then each extra key
    clients_to_try = [client] + [Groq(api_key=k) for k in (extra_keys or []) if k]

    for cli in clients_to_try:
        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = cli.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content, model
                except RateLimitError:
                    if attempt < 2:
                        wait = 10 * (2 ** attempt)  # 10s, 20s
                        time.sleep(wait)
                    else:
                        break  # try next model on same key
                except BadRequestError:
                    break  # model unavailable — try next model
                except Exception as exc:
                    raise exc
        # All models on this key exhausted (rate limited) — loop to next key

    return (
        "⚠️ All API keys and models are currently rate-limited. "
        "Please wait a few minutes or add another Groq API key in ⚙️ Settings.",
        "none",
    )


def _strip_think_blocks(text: str) -> str:
    """Remove <think>…</think> reasoning blocks that Llama 4 / Qwen3 models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_followups(raw: str) -> tuple:
    """
    Split a combined answer+follow-up response on the _FU_SEP separator.
    Handles markdown-wrapped separators and <think> blocks from Llama 4/Qwen3.
    Returns (answer_text, [follow_up_1, follow_up_2, follow_up_3]).
    """
    # 1. Strip chain-of-thought reasoning blocks (Llama 4 Scout, Qwen3)
    cleaned = _strip_think_blocks(raw)

    # 2. Normalize every known separator variation the model might emit
    cleaned = (
        cleaned
        .replace("**---FU---**", "---FU---")
        .replace("`---FU---`", "---FU---")
        .replace("--- FU ---", "---FU---")
        .replace("---fu---", "---FU---")
        .replace("--FU--", "---FU---")
        .replace("– FU –", "---FU---")
        .replace("— FU —", "---FU---")
    )
    # Also catch lines that are ONLY the separator (with surrounding whitespace/dashes)
    cleaned = re.sub(r"(?m)^\s*-{2,}\s*FU\s*-{2,}\s*$", "---FU---", cleaned)

    if "---FU---" not in cleaned:
        return cleaned.strip(), []

    parts = cleaned.split("---FU---", 1)
    answer = parts[0].strip()
    follow_ups: List[str] = []
    for line in parts[1].strip().splitlines():
        line = line.strip()
        # Accept lines starting with a digit followed by . or )
        if line and line[0].isdigit():
            text = line.split(".", 1)[-1].strip().lstrip(")").strip()
            if text and len(text) > 4:   # skip very short noise
                follow_ups.append(text)
    return answer, follow_ups[:3]


def get_answer(
    question: str,
    chunks: List[Dict],
    api_key: str,
    model_name: str = "openai/gpt-oss-120b",
    language: str = "English",
    include_followups: bool = True,
    chat_history: List[Dict] = None,
    answer_style: str = "Normal",
    comprehensive: bool = False,
    list_all: bool = False,
    extra_keys: List[str] = None,
) -> tuple:
    """
    Returns (answer_text, follow_up_questions_list, actual_model_used).
    answer_style: 'Short', 'Normal', or 'Long'.
    extra_keys: additional Groq API keys tried in order when primary key hits daily limit.
    """
    client = _groq_client(api_key)
    # Embed FU instruction in user prompt (more reliable cross-model compliance)
    prompt = _build_qa_prompt(
        question, chunks, language,
        style=answer_style, include_fu=include_followups,
        comprehensive=comprehensive,
        list_all=list_all,
    )
    # Dynamic system prompt carries language + FU instructions baked in
    system = _make_system_prompt(language=language, include_fu=include_followups)

    # Build message list: system + prior conversation + current question
    messages: List[Dict] = [{"role": "system", "content": system}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    # Long answers may need more tokens
    _max_tok = 512 if answer_style == "Short" else (3000 if answer_style == "Long" else 2048)

    raw, actual_model = _call_with_retry(
        client,
        model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=_max_tok,
        extra_keys=extra_keys,
    )
    # Strip <think> blocks before any further processing (Llama 4 Scout, Qwen3)
    raw = _strip_think_blocks(raw)
    if include_followups:
        answer, follow_ups = _parse_followups(raw)
        return answer, follow_ups, actual_model
    return raw.strip(), [], actual_model


def get_document_summary(
    full_text: str,
    doc_name: str,
    api_key: str,
    model_name: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate a structured summary of an entire document.

    Args:
        full_text:  Concatenated text of all chunks.
        doc_name:   Display name of the document.
        api_key:    Groq API key.
        model_name: Groq model identifier.

    Returns:
        Summary string in Markdown.
    """
    client = _groq_client(api_key)
    text = full_text[:20_000] if len(full_text) > 20_000 else full_text
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(doc_name=doc_name, text=text)
    content, _ = _call_with_retry(
        client,
        model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return content


def get_highlight(
    question: str,
    chunks: List[Dict],
    api_key: str,
    model_name: str = "llama-3.1-8b-instant",
) -> str:
    """Extract the single most relevant sentence from the chunks for the question."""
    if not chunks:
        return ""
    combined = " | ".join(c["text"][:400] for c in chunks[:3])
    prompt = (
        f"From the text below, copy the single most relevant sentence that "
        f"directly answers: '{question}'\n\nTEXT: {combined}\n\n"
        "Return ONLY the exact sentence, nothing else."
    )
    client = _groq_client(api_key)
    result, _ = _call_with_retry(
        client, model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=150,
    )
    return result.strip().strip('"')


def keyword_search(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """Pure keyword search — returns chunks containing any word from the query."""
    query_words = set(query.lower().split())
    scored: List[tuple] = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        hits = sum(1 for w in query_words if w in text_lower)
        if hits:
            scored.append((hits, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
