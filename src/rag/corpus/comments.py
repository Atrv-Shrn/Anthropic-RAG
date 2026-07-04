"""Fetch PR review + inline comments the issues reader doesn't cover.

The ``GitHubRepositoryIssuesReader`` exposes issue/PR bodies and top-level comments but
misses review threads and inline review comments. This thin fetcher fills that gap.
"""

from __future__ import annotations


def fetch_review_comments(repo: str, org: str | None = None, since: str | None = None) -> list:
    """Fetch PR review + inline comments (windowed by ``since``) for a repo."""
    raise NotImplementedError("M3")