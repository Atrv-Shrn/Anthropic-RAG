"""FastMCP Streamable HTTP server exposing exactly two tools.

- ``answer(query)`` -> runs the full retrieval + generation pipeline and returns the
  grounded answer plus the source document IDs.
- ``get_documents(doc_ids)`` -> raw documents for those IDs straight from the Redis
  docstore (no regeneration, no retrieval).

Bound to ``0.0.0.0`` so it is LAN-reachable now and publicly reachable once hosted on
EC2 (just open the port + set a bearer token — no pipeline code change).
"""

from __future__ import annotations


def build_server():
    """Return a configured FastMCP server with the two tools registered."""
    raise NotImplementedError("M6")


def run() -> None:
    """Build the server and serve over Streamable HTTP."""
    raise NotImplementedError("M6")