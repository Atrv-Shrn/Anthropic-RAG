"""Dense Qdrant vector store.

Plain ``QdrantVectorStore`` — no ``enable_hybrid``, no sparse model. Sparse/lexical
retrieval is handled separately via ``BM25Retriever`` over the Redis docstore and fused
in ``retrieve/retriever.py``.
"""

from __future__ import annotations


def get_vector_store():
    """Return a dense ``QdrantVectorStore`` bound to the configured collection."""
    raise NotImplementedError("M4")