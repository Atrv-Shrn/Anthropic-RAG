"""APScheduler hourly job -> run incremental ingest.

An interval trigger fires every hour, running the corpus fetch + ingestion pipeline over
only what changed since each stream's watermark. The scheduler lives inside the app
container alongside the MCP server.
"""

from __future__ import annotations


def start_scheduler(interval_minutes: int = 60):
    """Start the APScheduler hourly incremental-ingest job (blocking the current loop)."""
    raise NotImplementedError("M6")