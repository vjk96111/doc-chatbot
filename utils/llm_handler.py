"""
llm_handler.py
Handles all calls to Groq: Q&A over retrieved context and
one-shot document summarisation.
"""

import time
from typing import List, Dict

from groq import Groq, RateLimitError

# Fallback model order when rate-limited (smaller = fewer tokens/min used)
_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
]

# ─────────────────────────────────── prompts ─────────────────────────────────

_SYSTEM_PROMPT = (
    "You are DocChat, a precise and helpful assistant that answers questions "
    "strictly based on content retrieved from uploaded documents.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided context snippets.\n"
    "- Always cite the source document name and page/section number.\n"
    "- If a context snippet mentions a diagram, figure, chart, or table, "
    "acknowledge it — the image will be displayed automatically in the chat.\n"
    "- If the answer is genuinely not in the context, say so clearly — "
    "do NOT hallucinate facts.\n"
    "- Be concise but complete."
)

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

def _build_qa_prompt(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return (
            f"The user asked: {question}\n\n"
            "No relevant content was found in the uploaded documents. "
            "Please inform the user politely."
        )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc_name", "Unknown")
        src = chunk.get("source", "")
        context_blocks.append(
            f"[Context {i} | {doc} | {src}]\n{chunk['text']}"
        )

    context = "\n\n" + ("─" * 60) + "\n\n".join(context_blocks)
    return (
        f"RETRIEVED CONTEXT FROM DOCUMENTS:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Answer the question based on the context above and cite sources."
    )


def _groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ─────────────────────────────────── public API ──────────────────────────────

def _call_with_retry(client: Groq, model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    """
    Call Groq with exponential backoff + automatic model fallback on rate limit.
    Tries up to 3 times per model, then falls back to a smaller model.
    """
    models_to_try = [model_name] + [m for m in _FALLBACK_MODELS if m != model_name]

    for model in models_to_try:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt < 2:
                    wait = 10 * (2 ** attempt)  # 10s, 20s
                    time.sleep(wait)
                else:
                    break  # try next model
            except Exception as exc:
                raise exc  # non-rate-limit errors bubble up immediately

    return (
        "⚠️ The AI service is currently rate-limited (too many requests). "
        "Please wait 30 seconds and try again."
    )


def get_answer(
    question: str,
    chunks: List[Dict],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    client = _groq_client(api_key)
    prompt = _build_qa_prompt(question, chunks)
    return _call_with_retry(
        client,
        model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )


def get_document_summary(
    full_text: str,
    doc_name: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
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
    return _call_with_retry(
        client,
        model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
