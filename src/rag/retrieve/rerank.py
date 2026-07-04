"""Cross-encoder reranker: ``bge-reranker-base`` via ``SentenceTransformerRerank`` -> top_n ~6."""

from __future__ import annotations


def build_reranker(top_n: int = 6):
    """Return a ``SentenceTransformerRerank`` node postprocessor."""
    raise NotImplementedError("M5")