"""Dense Qdrant vector store (no hybrid, no sparse).

A plain ``QdrantVectorStore`` with ``enable_hybrid=False``. Sparse/lexical retrieval
is handled separately by ``BM25Retriever`` over the Redis docstore and fused in
``retrieve/retriever.py`` — we do not use Qdrant's native sparse/fastembed path.

We construct a ``QdrantClient`` ourselves and pass it in: ``QdrantVectorStore``
requires either a client instance *or* both a url and an api_key, and a local
Qdrant has no api_key.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore

from config.settings import Settings, get_settings


def get_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    """Return a dense ``QdrantVectorStore`` bound to the configured collection."""
    s = settings or get_settings()
    client = QdrantClient(url=s.qdrant_url)
    return QdrantVectorStore(
        collection_name=s.qdrant_collection,
        client=client,
        enable_hybrid=False,
        batch_size=64,
    )