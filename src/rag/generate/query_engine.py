"""RetrieverQueryEngine with cross-encoder rerank postprocessor, ``response_mode="compact"``.

Compact is the only synthesis mode — no toggle. If quality falls short we change the
code rather than add mode switches. Prompt templates are loaded from
``config/templates/`` by intent.
"""

from __future__ import annotations


def build_query_engine():
    """Return a ``RetrieverQueryEngine`` (hybrid retriever + reranker, compact mode)."""
    raise NotImplementedError("M5")