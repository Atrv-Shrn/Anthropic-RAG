"""Incremental ingest orchestrator — shared by the seed entrypoint and the scheduler.

For each repo in ``config/repos.yaml``:
  1. Fetch issues/PRs and comments windowed by their SQLite watermarks
     (seed window = ``now - 7d``; subsequent runs = the stored watermark).
  2. Fetch the docs snapshot (full — no activity window; UPSERTS dedup skips
     unchanged docs so re-fetching is cheap).
  3. Run the ingestion pipeline (Redis docstore + Qdrant, UPSERTS dedup).
  4. Advance each stream's watermark to the max ``updated_at`` seen.

Returns a per-repo stats dict. Both the one-shot seed (container entrypoint) and
the hourly APScheduler job call ``run_incremental_ingest``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config.repos import load_repos
from config.settings import Settings, get_settings
from rag.corpus import comments as comments_mod
from rag.corpus.comments import COMMENTS_STREAM
from rag.corpus.github_source import (
    DOCS_STREAM,
    ISSUES_PRS_STREAM,
    fetch_docs,
    fetch_issues_prs,
)
from rag.corpus.watermarks import get_watermark, set_watermark

log = logging.getLogger(__name__)


def _max_updated_at(docs: list[Any]) -> str | None:
    """Return the max ``updated_at`` across the docs' metadata, or ``None``."""
    ts = [d.metadata.get("updated_at") for d in docs if d.metadata.get("updated_at")]
    return max(ts) if ts else None


def _advance_watermark(repo: str, stream: str, docs: list[Any]) -> None:
    """Advance ``(repo, stream)`` watermark to the max ``updated_at`` seen, if any."""
    ts = _max_updated_at(docs)
    if ts:
        prev = get_watermark(repo, stream)
        # Only move forward (defensive against out-of-order / clock skew).
        if prev is None or ts > prev:
            set_watermark(repo, stream, ts)


def _ingest_repo(repo: str, org: str, settings: Settings) -> dict[str, int]:
    from rag.ingest.pipeline import run_ingest  # local import to avoid heavy import at module load

    issues_prs = fetch_issues_prs(repo, org=org, settings=settings)
    comments = comments_mod.fetch_review_comments(repo, org=org, settings=settings)
    docs = fetch_docs(repo, org=org, settings=settings)

    documents = issues_prs + comments + docs
    embedded = run_ingest(documents, settings=settings) if documents else []

    # Advance activity-stream watermarks; docs get a last-synced stamp.
    _advance_watermark(repo, ISSUES_PRS_STREAM, issues_prs)
    _advance_watermark(repo, COMMENTS_STREAM, comments)
    if docs:
        set_watermark(repo, DOCS_STREAM, datetime.now(timezone.utc).isoformat())

    return {
        "docs": len(docs),
        "issues_prs": len(issues_prs),
        "comments": len(comments),
        "embedded": len(embedded),
    }


def run_incremental_ingest(
    repos: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, dict[str, int]]:
    """Run an incremental ingest pass across the curated repos.

    ``repos`` defaults to the full list in ``config/repos.yaml``. Returns a
    per-repo stats dict ``{repo: {docs, issues_prs, comments, embedded}}``.
    """
    s = settings or get_settings()
    cfg_org, cfg_repos = load_repos()
    org = s.github_org or cfg_org
    repos = repos or cfg_repos

    stats: dict[str, dict[str, int]] = {}
    for repo in repos:
        try:
            stats[repo] = _ingest_repo(repo, org, s)
            log.info("ingested %s: %s", repo, stats[repo])
        except Exception:
            # One repo failing should not abort the whole pass.
            log.exception("FAILED to ingest %s", repo)
            stats[repo] = {"docs": 0, "issues_prs": 0, "comments": 0, "embedded": 0, "error": 1}
    return stats


def seed(settings: Settings | None = None) -> dict[str, dict[str, int]]:
    """One-shot full seed — identical to an incremental pass (UPSERTS dedup)."""
    return run_incremental_ingest(settings=settings)