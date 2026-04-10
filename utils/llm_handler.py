"""
llm_handler.py
Handles all calls to Groq: Q&A over retrieved context and
one-shot document summarisation.
"""

from typing import List, Dict

from groq import Groq

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

def get_answer(
    question: str,
    chunks: List[Dict],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate a grounded answer from retrieved document chunks.

    Args:
        question:   User's question.
        chunks:     Top-k chunks returned by VectorStore.search().
        api_key:    Groq API key.
        model_name: Groq model identifier.

    Returns:
        Answer string in Markdown.
    """
    client = _groq_client(api_key)
    prompt = _build_qa_prompt(question, chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content


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

    # Groq context window is large, but be conservative for free tier
    text = full_text[:20_000] if len(full_text) > 20_000 else full_text

    prompt = _SUMMARY_PROMPT_TEMPLATE.format(doc_name=doc_name, text=text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content
