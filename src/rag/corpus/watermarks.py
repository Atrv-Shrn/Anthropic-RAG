"""Per-repo / per-stream SQLite watermarks for incremental ingestion.

Schema: ``watermarks(repo TEXT, stream TEXT, last_synced_ts TEXT, PRIMARY KEY(repo, stream))``.
Read before fetch (seed window = max(watermark, now-7d)), write the max seen timestamp
after a successful ingest so the next run only pulls deltas.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data/watermarks.sqlite3")


def get_watermark(repo: str, stream: str, db: Path | str = DEFAULT_DB) -> str | None:
    """Return the last synced ISO timestamp for ``(repo, stream)``, or None."""
    raise NotImplementedError("M3")


def set_watermark(repo: str, stream: str, ts: str, db: Path | str = DEFAULT_DB) -> None:
    """Upsert the watermark for ``(repo, stream)`` to ``ts``."""
    raise NotImplementedError("M3")