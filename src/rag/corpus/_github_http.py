"""Thin GitHub REST API helper (httpx) shared by the issues/PRs and comments fetchers.

We use this instead of the LlamaHub ``GitHubRepositoryIssuesReader`` because that
reader exposes no ``since`` parameter and does not return ``updated_at`` — both of
which we need for the spec's ``updated_at``-based watermark windowing. The docs
fetch still uses ``GithubRepositoryReader`` (its extension filter is the docs-only
scoping guard).

Auth: a Bearer token from ``GITHUB_TOKEN`` if set (5000 req/hr); otherwise
unauthenticated (60 req/hr — fine for small seed fetches, not for the hourly job).
"""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from config.settings import Settings, get_settings

GITHUB_API = "https://api.github.com"
# How many seconds to wait for a single GitHub API call.
DEFAULT_TIMEOUT = 30.0


def _headers(settings: Settings | None = None) -> dict[str, str]:
    s = settings or get_settings()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if s.github_token:
        headers["Authorization"] = f"Bearer {s.github_token}"
    return headers


def paginate(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    settings: Settings | None = None,
    per_page: int = 100,
    max_pages: int = 200,
) -> Iterator[Any]:
    """Yield items from a paginated GitHub list endpoint.

    ``path`` is the path component after ``https://api.github.com`` (e.g.
    ``/repos/{owner}/{repo}/issues``). Follows ``Link: rel="next"`` until exhausted
    or ``max_pages`` is hit (safety cap so a runaway loop can't hammer the API).
    """
    s = settings or get_settings()
    params = dict(params or {})
    params.setdefault("per_page", per_page)

    url: str | None = f"{GITHUB_API}{path}"
    pages = 0
    with httpx.Client(
        headers=_headers(s), timeout=DEFAULT_TIMEOUT, follow_redirects=True
    ) as client:
        while url and pages < max_pages:
            resp = client.get(url, params=params if pages == 0 else None)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            yield from data
            pages += 1
            # After the first request, 'next' link already carries query params.
            url = _next_link(resp.headers.get("link", ""))


def _next_link(link_header: str) -> str | None:
    """Parse the ``rel="next"`` URL out of a GitHub ``Link`` header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            # '<url>; rel="next"'
            start = part.find("<")
            end = part.find(">")
            if start != -1 and end != -1:
                return part[start + 1 : end]
    return None