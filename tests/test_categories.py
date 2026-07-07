"""Unit tests for category scoping + raw-docstore browsing.

No network, no cross-encoder: category resolution is pure, and the BM25 node filter
and ``list_documents`` are exercised against ``SimpleNamespace`` fakes (same style as
``tests/test_source_ids.py``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag.retrieve.retriever import (
    CATEGORY_SOURCE_TYPES,
    resolve_source_types,
)
from rag.server import mcp_server


# --- resolve_source_types -------------------------------------------------------

def test_resolve_each_category_maps_to_its_source_types():
    assert resolve_source_types("prs") == ("pr",)
    assert resolve_source_types("issues") == ("issue",)
    assert resolve_source_types("comments") == ("issue_comment", "review_comment")
    assert resolve_source_types("docs") == ("docs",)


def test_resolve_all_and_none_mean_no_filter():
    assert resolve_source_types(None) is None
    assert resolve_source_types("all") is None


def test_resolve_unknown_category_raises_value_error():
    with pytest.raises(ValueError):
        resolve_source_types("bogus")


# --- BM25 node-list filter (the dual-arm lexical filter in build_retriever) ------

def _fake_node(node_id, source_type, text="x", repo="anthropics/repo"):
    return SimpleNamespace(
        node_id=node_id,
        metadata={"source_type": source_type, "repo": repo},
        get_content=lambda: text,
    )


def _bm25_filter(nodes, source_types):
    """Mirror the node-list filter used for the BM25 arm in build_retriever."""
    wanted = set(source_types)
    return [n for n in nodes if n.metadata.get("source_type") in wanted]


def test_bm25_filter_selects_only_the_categorys_source_types():
    nodes = [
        _fake_node("a", "pr"),
        _fake_node("b", "issue"),
        _fake_node("c", "issue_comment"),
        _fake_node("d", "review_comment"),
        _fake_node("e", "docs"),
    ]

    prs = _bm25_filter(nodes, resolve_source_types("prs"))
    assert [n.node_id for n in prs] == ["a"]

    # comments spans both comment source_types
    comments = _bm25_filter(nodes, resolve_source_types("comments"))
    assert sorted(n.node_id for n in comments) == ["c", "d"]


# --- list_documents (raw docstore browse) ----------------------------------------

class _FakeDocstore:
    def __init__(self, nodes_by_id):
        self.docs = nodes_by_id


def _install_fake_docstore(monkeypatch, nodes_by_id):
    fake = _FakeDocstore(nodes_by_id)
    # list_documents calls _get_docstore(s) -> stub it to return our fake.
    monkeypatch.setattr(mcp_server, "_get_docstore", lambda s: fake)


def _mixed_docstore():
    return {
        "pr-1": _fake_node("pr-1", "pr", text="pr one body", repo="anthropics/a"),
        "pr-2": _fake_node("pr-2", "pr", text="pr two body", repo="anthropics/b"),
        "iss-1": _fake_node("iss-1", "issue", text="issue body", repo="anthropics/a"),
        "cmt-1": _fake_node("cmt-1", "issue_comment", text="c", repo="anthropics/a"),
        "doc-1": _fake_node("doc-1", "docs", text="d", repo="anthropics/a"),
    }


def _list_documents(**kwargs):
    """Invoke the registered list_documents tool via a built server."""
    mcp = mcp_server.build_server(settings=SimpleNamespace())
    # FastMCP wraps the function; reach the underlying callable (get_tool is async).
    tool = asyncio.run(mcp.get_tool("list_documents"))
    return tool.fn(**kwargs)


def test_list_documents_no_filter_returns_all(monkeypatch):
    _install_fake_docstore(monkeypatch, _mixed_docstore())
    out = _list_documents()
    assert out["total"] == 5
    assert out["limit"] == 50
    assert out["offset"] == 0
    assert len(out["documents"]) == 5
    assert set(d["doc_id"] for d in out["documents"]) == {
        "pr-1", "pr-2", "iss-1", "cmt-1", "doc-1"
    }


def test_list_documents_category_filter(monkeypatch):
    _install_fake_docstore(monkeypatch, _mixed_docstore())
    out = _list_documents(category="prs")
    assert out["total"] == 2
    assert sorted(d["doc_id"] for d in out["documents"]) == ["pr-1", "pr-2"]


def test_list_documents_repo_filter(monkeypatch):
    _install_fake_docstore(monkeypatch, _mixed_docstore())
    out = _list_documents(repo="anthropics/a")
    assert out["total"] == 4
    assert all(d["metadata"]["repo"] == "anthropics/a" for d in out["documents"])


def test_list_documents_paginates(monkeypatch):
    _install_fake_docstore(monkeypatch, _mixed_docstore())
    out = _list_documents(limit=2, offset=1)
    assert out["total"] == 5  # total is the full match count, not the page size
    assert out["limit"] == 2
    assert out["offset"] == 1
    assert len(out["documents"]) == 2


def test_list_documents_truncates_preview(monkeypatch):
    long_text = "z" * 500
    nodes = {"big": _fake_node("big", "docs", text=long_text)}
    _install_fake_docstore(monkeypatch, nodes)
    out = _list_documents()
    assert len(out["documents"][0]["text_preview"]) == 200


def test_list_documents_unknown_category_raises(monkeypatch):
    _install_fake_docstore(monkeypatch, _mixed_docstore())
    with pytest.raises(ValueError):
        _list_documents(category="nope")


def test_category_map_is_the_single_source_of_truth():
    # Guard against silent drift between the map and the documented categories.
    assert set(CATEGORY_SOURCE_TYPES) == {"prs", "issues", "comments", "docs"}
