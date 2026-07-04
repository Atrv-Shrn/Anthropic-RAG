"""IngestionPipeline wiring: Redis docstore + UPSERTS dedup, dense Qdrant vector store.

Redis's CRUD (get/set-by-hash) *is* the upsert/dedup mechanism: with
``docstore_strategy=UPSERTS`` unchanged documents are never re-embedded. The Redis
docstore is also what serves raw docs back to the MCP ``get_documents`` tool.
"""

from __future__ import annotations


def build_pipeline():
    """Return a configured ``IngestionPipeline`` (transforms + docstore + vector store)."""
    raise NotImplementedError("M4")


def run_ingest(documents):
    """Run an ingest pass over ``documents``; return run stats (new vs unchanged)."""
    raise NotImplementedError("M4")