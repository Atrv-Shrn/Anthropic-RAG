"""Unit tests for source-node -> doc_id extraction (dedup via ref_doc_id)."""

from __future__ import annotations

from types import SimpleNamespace

from rag.evals.eval_harness import _source_texts_and_ids
from rag.server.mcp_server import _source_doc_ids


def _node_with_score(node_id, ref_doc_id, text="chunk text"):
    node = SimpleNamespace(
        node_id=node_id,
        ref_doc_id=ref_doc_id,
        get_content=lambda: text,
    )
    return SimpleNamespace(node=node)


def test_source_doc_ids_prefers_ref_doc_id_and_dedups():
    nodes = [
        _node_with_score("chunk-1", "doc-A"),
        _node_with_score("chunk-2", "doc-A"),  # same parent -> dedup
        _node_with_score("chunk-3", "doc-B"),
    ]
    assert _source_doc_ids(nodes) == ["doc-A", "doc-B"]


def test_source_doc_ids_falls_back_to_node_id():
    node = SimpleNamespace(node=SimpleNamespace(node_id="only-node-id", ref_doc_id=None))
    assert _source_doc_ids([node]) == ["only-node-id"]


def test_source_doc_ids_empty():
    assert _source_doc_ids([]) == []
    assert _source_doc_ids(None) == []


def test_source_texts_and_ids_returns_texts_and_deduped_ids():
    nodes = [
        _node_with_score("c1", "doc-A", text="  hello  "),
        _node_with_score("c2", "doc-A", text="world"),
    ]
    texts, ids = _source_texts_and_ids(nodes)
    assert texts == ["hello", "world"]  # stripped, one per chunk
    assert ids == ["doc-A"]  # deduped parent
