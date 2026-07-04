"""GitHub corpus source — docs-only file reader + issues/PRs reader.

Scope (see docs/SPEC.md): docs + issues/PRs/comments only. We do NOT ingest source
code / whole codebases — that bloats Qdrant. ``GithubRepositoryReader`` is filtered to
documentation extensions/dirs + READMEs; ``GitHubRepositoryIssuesReader`` covers
issues and PRs, windowed by per-stream watermarks (7-day seed window, then delta).
"""

from __future__ import annotations


def fetch_docs(repo: str, org: str | None = None) -> list:
    """Fetch documentation documents for a repo via ``GithubRepositoryReader``."""
    raise NotImplementedError("M3")


def fetch_issues_prs(repo: str, org: str | None = None, since: str | None = None) -> list:
    """Fetch issues + PRs (windowed by ``since`` ISO timestamp) for a repo."""
    raise NotImplementedError("M3")