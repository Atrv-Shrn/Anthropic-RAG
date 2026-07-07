"""Ollama ``nomic-embed-text`` embeddings with the required task prefixes.

nomic-embed-text is a contrastive model that *requires* a task prefix for good
retrieval: ``search_document:`` on document nodes and ``search_query:`` on
queries. Without the prefixes, recall drops materially.

LlamaIndex's ``OllamaEmbedding`` already prepends ``text_instruction`` /
``query_instruction`` to every input (see ``_format_text`` / ``_format_query``),
so we set those rather than subclass. The result is that ingest embeds every
node as ``search_document: <text>`` and retrieval embeds every query as
``search_query: <query>`` — exactly the nomic contract.
"""

from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding

from config.settings import Settings, get_settings

# nomic task prefixes — these are the only valid values for nomic-embed-text.
DOCUMENT_PREFIX = "search_document:"
QUERY_PREFIX = "search_query:"


def get_embed_model(settings: Settings | None = None) -> OllamaEmbedding:
    """Return a task-prefix-aware dense ``OllamaEmbedding`` (768-dim nomic)."""
    s = settings or get_settings()
    return OllamaEmbedding(
        model_name=s.embed_model,
        base_url=s.ollama_embed_base_url,
        text_instruction=DOCUMENT_PREFIX,  # applied to every document node
        query_instruction=QUERY_PREFIX,    # applied to every retrieval query
        embed_batch_size=64,
    )