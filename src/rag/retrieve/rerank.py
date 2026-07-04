"""Cross-encoder reranker: ``bge-reranker-base`` via ``SentenceTransformerRerank``.

A local cross-encoder re-scores the fused candidate nodes against the query and
keeps the top ``top_n`` (~6). This runs *after* retrieval, before synthesis.
"""

from __future__ import annotations

from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TOP_N = 6


def build_reranker(top_n: int = DEFAULT_TOP_N) -> SentenceTransformerRerank:
    """Return a ``SentenceTransformerRerank`` postprocessor (bge-reranker-base)."""
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=top_n)