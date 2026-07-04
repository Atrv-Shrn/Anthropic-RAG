"""Ollama ``nomic-embed-text`` wrapper that injects task prefixes.

nomic-embed-text requires ``search_document:`` on document nodes and ``search_query:`` on
queries for good retrieval quality — without the prefixes recall drops materially. This
module wraps ``OllamaEmbedding`` so ingest and retrieval always use the correct prefix.
"""

from __future__ import annotations


def get_embed_model():
    """Return a task-prefix-aware Ollama embedding model (dense, 768-dim)."""
    raise NotImplementedError("M2")


DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "