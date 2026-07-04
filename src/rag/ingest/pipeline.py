"""IngestionPipeline wiring: Redis docstore + UPSERTS dedup, dense Qdrant store.

Redis's CRUD (get/set-by-hash) *is* the upsert/dedup mechanism: with
``docstore_strategy=UPSERTS`` the pipeline checks each document's hash against the
Redis docstore — unchanged documents are skipped entirely (no re-split, no
re-embed), and only new or hash-changed documents are transformed and embedded into
Qdrant. The same Redis docstore is what the MCP ``get_documents`` tool reads raw
documents back from.

``run_ingest`` returns the nodes that were actually processed (i.e. newly embedded)
on this pass — re-running with identical documents yields zero, which is the dedup
verification gate.
"""

from __future__ import annotations

from typing import Sequence

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import BaseNode, Document
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore

from config.settings import Settings, get_settings
from rag.ingest.embeddings import get_embed_model
from rag.ingest.splitters import make_splitters
from rag.index.qdrant_store import get_vector_store

DEFAULT_NAMESPACE = "anthropic_rag"


def get_docstore(settings: Settings | None = None, namespace: str = DEFAULT_NAMESPACE) -> RedisDocumentStore:
    """Return a ``RedisDocumentStore`` backed by the configured Redis."""
    s = settings or get_settings()
    return RedisDocumentStore(
        redis_kvstore=RedisKVStore(redis_uri=s.redis_url),
        namespace=namespace,
    )


def build_pipeline(settings: Settings | None = None) -> IngestionPipeline:
    """Return a configured ``IngestionPipeline`` (splitter+embed, Qdrant, Redis UPSERTS)."""
    s = settings or get_settings()
    return IngestionPipeline(
        transformations=[*make_splitters(), get_embed_model(s)],
        vector_store=get_vector_store(s),
        docstore=get_docstore(s),
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )


def run_ingest(
    documents: Sequence[Document],
    settings: Settings | None = None,
    pipeline: IngestionPipeline | None = None,
    show_progress: bool = False,
) -> list[BaseNode]:
    """Run an ingest pass over ``documents``; return the nodes that were embedded.

    Only new / hash-changed documents are transformed and embedded (UPSERTS dedup).
    Re-running with identical documents returns an empty list.
    """
    pipe = pipeline or build_pipeline(settings)
    nodes = pipe.run(documents=list(documents), show_progress=show_progress)
    return nodes