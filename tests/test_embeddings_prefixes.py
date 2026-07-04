"""Unit test: the nomic embedding wrapper wires the required task prefixes.

nomic-embed-text needs `search_document:` on nodes and `search_query:` on queries;
getting these wrong silently tanks recall, so assert they are set on the model.
"""

from __future__ import annotations

from config.settings import Settings
from rag.ingest.embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    get_embed_model,
)


def test_prefix_constants():
    assert DOCUMENT_PREFIX == "search_document:"
    assert QUERY_PREFIX == "search_query:"


def test_embed_model_wires_nomic_prefixes():
    s = Settings(embed_model="nomic-embed-text", ollama_embed_base_url="http://localhost:11144")
    model = get_embed_model(s)
    # OllamaEmbedding stores the instructions it prepends to docs vs queries.
    assert model.text_instruction == DOCUMENT_PREFIX
    assert model.query_instruction == QUERY_PREFIX
    assert model.model_name == "nomic-embed-text"
