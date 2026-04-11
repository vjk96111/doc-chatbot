"""
vector_store.py
FAISS-based vector store for semantic search.
Embeddings are generated locally using sentence-transformers (no API key needed).
Falls back to a pure-NumPy cosine-similarity implementation when FAISS
is not installed (slower but always works).
"""

from typing import List, Dict, Optional

import numpy as np

# Module-level model cache — loaded once per process, reused forever
_EMBED_MODEL = None
# paraphrase-MiniLM-L3-v2: only 17 MB, 2× faster than L6, same quality for RAG
_EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"


def get_embed_model():
    """Return the shared embedding model, loading it once on first call."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from exc
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


class VectorStore:
    """
    In-memory semantic vector store backed by FAISS (or NumPy as fallback).
    Embeddings use sentence-transformers locally — no API key required.

    Usage:
        store = VectorStore()
        store.add_documents(chunks)
        results = store.search("my question", top_k=5)
    """

    def __init__(self) -> None:
        self.chunks: List[Dict] = []
        self._embeddings: List[List[float]] = []
        self._index = None          # FAISS index or numpy array
        self._dimension: Optional[int] = None
        self._use_faiss: Optional[bool] = None  # determined on first build

    # ─────────────────────────────── internal ────────────────────────────────

    def _check_faiss(self) -> bool:
        if self._use_faiss is None:
            try:
                import faiss  # noqa: F401
                self._use_faiss = True
            except ImportError:
                self._use_faiss = False
        return self._use_faiss

    def _get_embed_model(self):
        """Return the shared module-level model (loaded once, reused)."""
        return get_embed_model()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally using sentence-transformers."""
        model = self._get_embed_model()
        vectors = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,          # process 64 chunks at once instead of 1-by-1
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def _rebuild_index(self) -> None:
        if not self._embeddings:
            return
        arr = np.array(self._embeddings, dtype=np.float32)
        self._dimension = arr.shape[1]

        if self._check_faiss():
            import faiss

            # L2-normalise → inner-product == cosine similarity
            faiss.normalize_L2(arr)
            idx = faiss.IndexFlatIP(self._dimension)
            idx.add(arr)
            self._index = idx
        else:
            # Store normalised embeddings for NumPy search
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._index = arr / norms  # (N, D) normalised

    # ─────────────────────────────── public API ──────────────────────────────

    def add_documents(self, chunks: List[Dict], api_key: str = "") -> None:
        """Embed document chunks and add them to the index."""
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        new_embeddings = self._embed_texts(texts)
        self.chunks.extend(chunks)
        self._embeddings.extend(new_embeddings)
        self._rebuild_index()

    def search(self, query: str, api_key: str = "", top_k: int = 5) -> List[Dict]:
        """Return the top-k most relevant chunks for *query*."""
        if not self.chunks:
            return []

        q_emb = self._embed_texts([query])[0]
        q_arr = np.array([q_emb], dtype=np.float32)

        k = min(top_k, len(self.chunks))

        if self._check_faiss() and self._index is not None:
            import faiss

            faiss.normalize_L2(q_arr)
            scores, indices = self._index.search(q_arr, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    chunk = dict(self.chunks[idx])
                    chunk["relevance_score"] = float(score)
                    results.append(chunk)
            return results

        # NumPy fallback
        normed_store = self._index  # already normalised in _rebuild_index
        q_norm = np.linalg.norm(q_arr[0])
        q_unit = q_arr[0] / (q_norm if q_norm else 1)
        sims = normed_store @ q_unit  # (N,)
        top_idx = np.argsort(sims)[-k:][::-1]
        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[idx])
            chunk["relevance_score"] = float(sims[idx])
            results.append(chunk)
        return results

    def clear(self) -> None:
        """Remove all stored chunks and embeddings."""
        self.chunks = []
        self._embeddings = []
        self._index = None
        self._dimension = None

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
