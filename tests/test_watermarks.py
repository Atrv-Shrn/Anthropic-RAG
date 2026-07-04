"""Unit tests for the SQLite watermark store (uses an isolated temp DB)."""

from __future__ import annotations

from rag.corpus.watermarks import all_watermarks, get_watermark, set_watermark


def test_get_missing_returns_none(tmp_path):
    db = tmp_path / "wm.sqlite3"
    assert get_watermark("repo", "docs", db) is None


def test_set_then_get_roundtrip(tmp_path):
    db = tmp_path / "wm.sqlite3"
    set_watermark("anthropic-sdk-python", "issues_prs", "2026-07-01T00:00:00Z", db)
    assert get_watermark("anthropic-sdk-python", "issues_prs", db) == "2026-07-01T00:00:00Z"


def test_set_overwrites_same_key(tmp_path):
    db = tmp_path / "wm.sqlite3"
    set_watermark("r", "docs", "2026-07-01T00:00:00Z", db)
    set_watermark("r", "docs", "2026-07-02T00:00:00Z", db)
    assert get_watermark("r", "docs", db) == "2026-07-02T00:00:00Z"


def test_streams_are_independent(tmp_path):
    db = tmp_path / "wm.sqlite3"
    set_watermark("r", "docs", "2026-01-01T00:00:00Z", db)
    set_watermark("r", "issues_prs", "2026-02-02T00:00:00Z", db)
    set_watermark("r", "comments", "2026-03-03T00:00:00Z", db)
    assert get_watermark("r", "docs", db) == "2026-01-01T00:00:00Z"
    assert get_watermark("r", "issues_prs", db) == "2026-02-02T00:00:00Z"
    assert get_watermark("r", "comments", db) == "2026-03-03T00:00:00Z"


def test_all_watermarks_lists_every_pair(tmp_path):
    db = tmp_path / "wm.sqlite3"
    set_watermark("a", "docs", "t1", db)
    set_watermark("b", "comments", "t2", db)
    rows = {(repo, stream): ts for repo, stream, ts in all_watermarks(db)}
    assert rows[("a", "docs")] == "t1"
    assert rows[("b", "comments")] == "t2"
    assert len(rows) == 2
