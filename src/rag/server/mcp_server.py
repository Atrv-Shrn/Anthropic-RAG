"""FastMCP Streamable HTTP server exposing the RAG tools.

- ``answer(query)`` -> runs the full retrieval + rerank + compact-generation pipeline
  and returns the grounded answer plus the source document IDs (parent doc_ids the
  ``get_documents`` tool can resolve).
- ``answer_prs`` / ``answer_issues`` / ``answer_comments`` / ``answer_docs`` -> the same
  pipeline scoped to a single content category.
- ``list_documents(...)`` -> browse the raw Redis docstore (paginated, optional
  category/repo filter) without needing IDs first.
- ``get_documents(doc_ids)`` -> raw documents for those IDs straight from the Redis
  docstore (no retrieval, no generation).

Bound to ``0.0.0.0`` so it is LAN-reachable now and publicly reachable once hosted on
EC2 (open the port in the security group + set ``MCP_AUTH_TOKEN`` for bearer auth —
no pipeline code change).
"""

from __future__ import annotations

import threading
from typing import Any

from fastmcp import FastMCP

from config.settings import Settings, get_settings
from rag.ingest.pipeline import get_docstore
from rag.retrieve.retriever import resolve_source_types

# Module-level singletons — built once and reused across calls (building a query engine
# loads the bge cross-encoder and connects to the stores). Engines are cached per
# category (each loads its own cross-encoder), keyed by category name ("all", "prs",
# ...). A lock guards lazy init so concurrent first requests don't double-build an
# engine (which would load the cross-encoder twice — a memory spike on a small host).
_query_engines: dict[str, Any] = {}
_docstore = None
_engine_lock = threading.Lock()


def _get_query_engine(category: str | None = None):
    key = category or "all"
    engine = _query_engines.get(key)
    if engine is None:
        with _engine_lock:
            engine = _query_engines.get(key)  # re-check inside the lock
            if engine is None:
                from rag.generate.query_engine import build_query_engine

                engine = build_query_engine(
                    source_types=resolve_source_types(category)
                )
                _query_engines[key] = engine
    return engine


def invalidate_query_engine() -> None:
    """Drop all cached query engines so the next request rebuilds them.

    The BM25 (lexical) retriever is built from a snapshot of the docstore at engine
    build time; after the scheduler ingests new documents, every cached engine's BM25
    arm is stale. Clearing the cache after an ingest makes the next ``answer*`` rebuild
    over the fresh docstore so lexical retrieval sees newly-ingested content.
    """
    with _engine_lock:
        _query_engines.clear()


def _get_docstore(settings: Settings):
    global _docstore
    if _docstore is None:
        _docstore = get_docstore(settings)
    return _docstore


def _source_doc_ids(source_nodes) -> list[str]:
    """Extract unique parent doc_ids from source nodes (dense chunks -> ref_doc_id)."""
    ids: list[str] = []
    seen: set[str] = set()
    for n in source_nodes or []:
        node = n.node
        pid = getattr(node, "ref_doc_id", None) or node.node_id
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def build_server(settings: Settings | None = None) -> FastMCP:
    """Return a configured FastMCP server with the two tools registered."""
    s = settings or get_settings()
    mcp = FastMCP(
        name="anthropic-rag",
        instructions=(
            "Grounded Q&A over the Anthropic GitHub organization (docs + "
            "issues/PRs/comments). Use `answer` for grounded answers over everything, "
            "or `answer_prs` / `answer_issues` / `answer_comments` / `answer_docs` to "
            "scope a question to one content type. Use `list_documents` to browse the "
            "raw source docs (paginated, optional category/repo filter) and "
            "`get_documents` to fetch raw docs by ID (IDs come from `answer*` or "
            "`list_documents`)."
        ),
    )

    def _run_answer(query: str, category: str | None) -> dict[str, Any]:
        qe = _get_query_engine(category)
        response = qe.query(query)
        return {
            "answer": str(response),
            "source_doc_ids": _source_doc_ids(response.source_nodes),
        }

    @mcp.tool
    def answer(query: str) -> dict[str, Any]:
        """Answer a question about the Anthropic GitHub org, grounded in ingested docs.

        Searches everything (docs, issues, PRs, comments). Returns the answer text and
        the source document IDs it was grounded on.
        """
        return _run_answer(query, None)

    @mcp.tool
    def answer_prs(query: str) -> dict[str, Any]:
        """Answer a question grounded only in pull requests."""
        return _run_answer(query, "prs")

    @mcp.tool
    def answer_issues(query: str) -> dict[str, Any]:
        """Answer a question grounded only in issues."""
        return _run_answer(query, "issues")

    @mcp.tool
    def answer_comments(query: str) -> dict[str, Any]:
        """Answer a question grounded only in comments (issue + PR review comments)."""
        return _run_answer(query, "comments")

    @mcp.tool
    def answer_docs(query: str) -> dict[str, Any]:
        """Answer a question grounded only in documentation files (.md/.rst/.ipynb)."""
        return _run_answer(query, "docs")

    @mcp.tool
    def list_documents(
        category: str | None = None,
        repo: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse raw source documents in the Redis docstore (no retrieval, no LLM).

        Filter by ``category`` (``prs``/``issues``/``comments``/``docs``, or ``None``
        for all) and/or exact ``repo`` (full ``org/repo``, e.g.
        ``anthropics/anthropic-sdk-python``). Paginate with ``limit``/``offset``. Returns
        ``total`` (matching count), the ``limit``/``offset`` used, and a ``documents``
        list of ``{doc_id, metadata, text_preview}`` (preview truncated to ~200 chars).
        Pass a returned ``doc_id`` to ``get_documents`` for the full text.
        """
        wanted = resolve_source_types(category)
        wanted_set = set(wanted) if wanted else None
        ds = _get_docstore(s)

        matched = []
        for doc_id, node in ds.docs.items():
            meta = node.metadata or {}
            if wanted_set is not None and meta.get("source_type") not in wanted_set:
                continue
            if repo is not None and meta.get("repo") != repo:
                continue
            matched.append((doc_id, node, meta))

        total = len(matched)
        page = matched[offset : offset + limit] if limit >= 0 else matched[offset:]
        documents = [
            {
                "doc_id": doc_id,
                "metadata": dict(meta),
                "text_preview": node.get_content()[:200],
            }
            for doc_id, node, meta in page
        ]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "documents": documents,
        }

    @mcp.tool
    def get_documents(doc_ids: list[str]) -> dict[str, Any]:
        """Fetch raw documents for the given IDs straight from the Redis docstore."""
        ds = _get_docstore(s)
        out: dict[str, Any] = {}
        for did in doc_ids:
            node = ds.get_document(did, raise_error=False)
            if node is None:
                out[did] = None
            else:
                out[did] = {
                    "text": node.get_content(),
                    "metadata": dict(node.metadata or {}),
                }
        return {"documents": out}

    return mcp


class BearerAuthMiddleware:
    """Minimal ASGI bearer-token gate (used when ``MCP_AUTH_TOKEN`` is set)."""

    _www_auth = b'{"error":"unauthorized"}'

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if headers.get("authorization") == f"Bearer {self.token}":
            return await self.app(scope, receive, send)
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"www-authenticate", b"Bearer"), (b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": self._www_auth})


def run() -> None:
    """Build the server and serve over Streamable HTTP (with bearer auth if configured)."""
    import uvicorn

    s = get_settings()
    mcp = build_server(s)

    # http_app gives us the ASGI app; we wrap it with the bearer gate when a token
    # is set (mcp.run() does not expose ASGI middleware). Run via uvicorn directly.
    if s.mcp_auth_token:
        app = mcp.http_app(transport="streamable-http")
        wrapped = BearerAuthMiddleware(app, s.mcp_auth_token)
        uvicorn.run(wrapped, host=s.mcp_host, port=s.mcp_port)
    else:
        mcp.run(transport="streamable-http", host=s.mcp_host, port=s.mcp_port)