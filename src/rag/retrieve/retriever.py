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
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

from config.settings import Settings, get_settings
from rag.generate.llm import get_llm
from rag.ingest.embeddings import get_embed_model
from rag.ingest.pipeline import get_docstore
from rag.index.qdrant_store import get_vector_store

TOP_K = 30

# A "category" maps to the set of ``source_type`` metadata values it covers. This is
# the single source of truth for category scoping — every doc already carries
# ``source_type`` (set at ingest in github_source.py / comments.py). ``all``/``None``
# means no filter (search everything).
CATEGORY_SOURCE_TYPES: dict[str, tuple[str, ...]] = {
    "prs": ("pr",),
    "issues": ("issue",),
    "comments": ("issue_comment", "review_comment"),
    "docs": ("docs",),
}


def resolve_source_types(category: str | None) -> tuple[str, ...] | None:
    """Map a category name to its ``source_type`` tuple (``None`` = no filter).

    ``None`` or ``"all"`` returns ``None`` (search everything). An unknown category
    raises ``ValueError`` rather than silently searching everything.
    """
    if category is None or category == "all":
        return None
    try:
        return CATEGORY_SOURCE_TYPES[category]
    except KeyError:
        raise ValueError(
            f"unknown category {category!r}; expected one of "
            f"{', '.join(sorted(CATEGORY_SOURCE_TYPES))}, 'all', or None"
        ) from None


def build_retriever(
    settings: Settings | None = None,
    source_types: tuple[str, ...] | None = None,
) -> QueryFusionRetriever:
    """Return a RRF-fused retriever over the dense Qdrant + BM25 docstore retrievers.

    When ``source_types`` is given, both arms are scoped to those types: the dense arm
    via a Qdrant metadata filter and the BM25 arm via a filtered node list. Filtering
    only one arm would leak other categories through fusion.
    """
    s = settings or get_settings()

    # Dense: index wrapped around the existing Qdrant collection (no re-index).
    index = VectorStoreIndex.from_vector_store(
        get_vector_store(s), embed_model=get_embed_model(s)
    )
    dense_filters = None
    if source_types:
        dense_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source_type",
                    value=list(source_types),
                    operator=FilterOperator.IN,
                )
            ]
        )
    dense = index.as_retriever(similarity_top_k=TOP_K, filters=dense_filters)

    # Lexical: BM25 over the raw documents in the Redis docstore.
    docstore = get_docstore(s)
    all_nodes = list(docstore.docs.values())
    if not all_nodes:
        raise RuntimeError(
            "Redis docstore is empty — run an ingest pass before building the retriever."
        )
    if source_types:
        wanted = set(source_types)
        nodes = [n for n in all_nodes if n.metadata.get("source_type") in wanted]
    else:
        nodes = all_nodes

    # If the category filter leaves no lexical nodes (whole store is non-empty), run
    # dense-only rather than erroring — the dense arm is still category-scoped.
    if not nodes:
        return QueryFusionRetriever(
            retrievers=[dense],
            llm=get_llm(s),
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=TOP_K,
            num_queries=1,
            use_async=False,
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