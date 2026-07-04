"""Per-repo / per-stream SQLite watermarks for incremental ingestion.

A watermark is the most recent ``updated_at`` timestamp we have already ingested
for a given ``(repo, stream)`` pair. Before each fetch we read it to bound the
window (seed: ``max(watermark, now - 7d)``; subsequent runs: ``watermark``); after
a successful ingest we write back the max ``updated_at`` seen so the next run only
pulls deltas.

Schema: ``watermarks(repo TEXT, stream TEXT, last_synced_ts TEXT,
                     PRIMARY KEY(repo, stream))``. ``stream`` is one of
``docs`` / ``issues_prs`` / ``comments``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DEFAULT_DB = Path("data/watermarks.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    repo            TEXT NOT NULL,
    stream          TEXT NOT NULL,
    last_synced_ts  TEXT NOT NULL,
    PRIMARY KEY (repo, stream)
)
"""


def _connect(db: Path | str) -> sqlite3.Connection:
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def get_watermark(repo: str, stream: str, db: Path | str = DEFAULT_DB) -> str | None:
    """Return the last synced ISO timestamp for ``(repo, stream)``, or ``None``."""
    with _connect(db) as conn:
        row = conn.execute(
            "SELECT last_synced_ts FROM watermarks WHERE repo = ? AND stream = ?",
            (repo, stream),
        ).fetchone()
    return row[0] if row else None


def set_watermark(repo: str, stream: str, ts: str, db: Path | str = DEFAULT_DB) -> None:
    """Upsert the watermark for ``(repo, stream)`` to ``ts`` (ISO 8601 string)."""
    with _connect(db) as conn:
        conn.execute(
            """
            INSERT INTO watermarks (repo, stream, last_synced_ts) VALUES (?, ?, ?)
            ON CONFLICT(repo, stream) DO UPDATE SET last_synced_ts = excluded.last_synced_ts
            """,
            (repo, stream, ts),
        )
        conn.commit()


def all_watermarks(db: Path | str = DEFAULT_DB) -> Iterable[tuple[str, str, str]]:
    """Yield every ``(repo, stream, last_synced_ts)`` row (for diagnostics/evals)."""
    with _connect(db) as conn:
        yield from conn.execute(
            "SELECT repo, stream, last_synced_ts FROM watermarks ORDER BY repo, stream"
        )