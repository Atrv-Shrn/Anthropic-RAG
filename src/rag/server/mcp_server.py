"""FastMCP Streamable HTTP server exposing exactly two tools.

- ``answer(query)`` -> runs the full retrieval + rerank + compact-generation pipeline
  and returns the grounded answer plus the source document IDs (parent doc_ids the
  ``get_documents`` tool can resolve).
- ``get_documents(doc_ids)`` -> raw documents for those IDs straight from the Redis
  docstore (no retrieval, no generation).

Bound to ``0.0.0.0`` so it is LAN-reachable now and publicly reachable once hosted on
EC2 (open the port in the security group + set ``MCP_AUTH_TOKEN`` for bearer auth —
no pipeline code change).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from config.settings import Settings, get_settings
from rag.ingest.pipeline import get_docstore

# Module-level singletons — built once at server startup and reused across calls
# (building the query engine loads the bge cross-encoder and connects to the stores).
_query_engine = None
_docstore = None


def _get_query_engine():
    global _query_engine
    if _query_engine is None:
        from rag.generate.query_engine import build_query_engine

        _query_engine = build_query_engine()
    return _query_engine


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
            "issues/PRs/comments). Use `answer` for grounded answers; use "
            "`get_documents` to fetch raw source docs by the IDs `answer` returns."
        ),
    )

    @mcp.tool
    def answer(query: str) -> dict[str, Any]:
        """Answer a question about the Anthropic GitHub org, grounded in ingested docs.

        Returns the answer text and the source document IDs it was grounded on.
        """
        qe = _get_query_engine()
        response = qe.query(query)
        return {
            "answer": str(response),
            "source_doc_ids": _source_doc_ids(response.source_nodes),
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
    middleware = [BearerAuthMiddleware] if s.mcp_auth_token else None

    # http_app gives us the ASGI app; we wrap it with the bearer gate when a token
    # is set (mcp.run() does not expose ASGI middleware). Run via uvicorn directly.
    if s.mcp_auth_token:
        app = mcp.http_app(transport="streamable-http")
        wrapped = BearerAuthMiddleware(app, s.mcp_auth_token)
        uvicorn.run(wrapped, host=s.mcp_host, port=s.mcp_port)
    else:
        mcp.run(transport="streamable-http", host=s.mcp_host, port=s.mcp_port)