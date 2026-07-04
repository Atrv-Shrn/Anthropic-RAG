"""Hybrid retriever: ``QueryFusionRetriever`` fusing dense (Qdrant) + BM25 (docstore).

Dense via nomic->Qdrant, lexical via ``BM25Retriever`` built from the Redis docstore
nodes (no embedding model needed), fused with reciprocal-rank fusion,
``similarity_top_k`` ~30. No HyDE.
"""

from __future__ import annotations


def build_retriever():
    """Return a ``QueryFusionRetriever`` over the dense + BM25 retrievers."""
    raise NotImplementedError("M5")