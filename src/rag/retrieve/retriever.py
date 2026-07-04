"""Hybrid retriever: ``QueryFusionRetriever`` fusing dense (Qdrant) + BM25 (docstore).

- Dense: ``VectorIndexRetriever`` over the Qdrant-backed ``VectorStoreIndex``
  (nomic embeddings, 768-dim).
- Lexical: ``BM25Retriever`` built from the Redis docstore documents (no embedding
  model needed).
- Fusion: reciprocal-rank fusion (RRF) across the two retrievers,
  ``similarity_top_k`` ~30. ``num_queries=1`` so no LLM-generated query expansion
  is performed — pure fusion, retrieval stays embedding+lexical only. No HyDE.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever

from config.settings import Settings, get_settings
from rag.generate.llm import get_llm
from rag.ingest.embeddings import get_embed_model
from rag.ingest.pipeline import get_docstore
from rag.index.qdrant_store import get_vector_store

TOP_K = 30


def build_retriever(settings: Settings | None = None) -> QueryFusionRetriever:
    """Return a RRF-fused retriever over the dense Qdrant + BM25 docstore retrievers."""
    s = settings or get_settings()

    # Dense: index wrapped around the existing Qdrant collection (no re-index).
    index = VectorStoreIndex.from_vector_store(
        get_vector_store(s), embed_model=get_embed_model(s)
    )
    dense = index.as_retriever(similarity_top_k=TOP_K)

    # Lexical: BM25 over the raw documents in the Redis docstore.
    docstore = get_docstore(s)
    nodes = list(docstore.docs.values())
    if not nodes:
        raise RuntimeError(
            "Redis docstore is empty — run an ingest pass before building the retriever."
        )
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)

    return QueryFusionRetriever(
        retrievers=[dense, bm25],
        # The fusion retriever eagerly resolves an LLM at construction (it would
        # fall back to Settings.llm and try OpenAI, which we don't ship). Pass our
        # Ollama LLM explicitly — with num_queries=1 it is constructed but never
        # invoked, so retrieval stays embedding+lexical only (no cloud call).
        llm=get_llm(s),
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=TOP_K,
        num_queries=1,   # no LLM query expansion — pure RRF fusion
        use_async=False,
    )