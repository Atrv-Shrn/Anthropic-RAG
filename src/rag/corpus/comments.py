"""Fetch GitHub comments the LlamaHub issues reader does not cover.

Two comment streams, both supporting ``since`` for watermark windowing:

- **Issue comments** (``/repos/{o}/{r}/issues/comments``): top-level comments on
  issues AND PRs — the issue/PR body itself is fetched in ``github_source`` but
  the conversation under it is not.
- **PR review / inline comments** (``/repos/{o}/{r}/pulls/comments``): review
  threads and inline code comments — these never appear in the issues endpoint.

Both are merged into one ``comments`` watermark stream.
"""

from __future__ import annotations

import re

from llama_index.core.schema import Document

from config.settings import Settings, get_settings
from rag.corpus._github_http import paginate, resolve_repo
from rag.corpus.github_source import _seed_since
from rag.corpus.watermarks import get_watermark

COMMENTS_STREAM = "comments"

# Metadata keys excluded from embedded text (kept for attribution/filtering).
EMBED_EXCLUDED_KEYS = [
    "repo", "stream", "source_type", "comment_id", "parent_number",
    "created_at", "updated_at", "source", "author", "path", "line",
]

# /repos/{owner}/{repo}/issues/123  ->  123
_ISSUE_NUM = re.compile(r"/issues/(\d+)$")
# /repos/{owner}/{repo}/pulls/123   ->  123
_PR_NUM = re.compile(r"/pulls/(\d+)$")


def _num_from_url(url: str, pattern: re.Pattern[str]) -> int | None:
    m = pattern.search(url or "")
    return int(m.group(1)) if m else None


def fetch_review_comments(
    repo: str,
    org: str | None = None,
    since: str | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Fetch issue-level + PR review/inline comments updated at or after ``since``."""
    s = settings or get_settings()
    org = org or s.github_org
    if since is None:
        since = get_watermark(repo, COMMENTS_STREAM) or _seed_since()

    # Resolve canonical owner/name so the request URLs don't 301 (a redirect on a
    # renamed/transferred repo drops the ?since= query string, breaking windowing).
    # Attribution keeps the configured org/repo.
    info = resolve_repo(org, repo, settings=s)
    owner, name = info.owner, info.name
    full = f"{org}/{repo}"
    docs: list[Document] = []

    # Issue-level comments (cover both issues and PRs).
    for item in paginate(
        f"/repos/{owner}/{name}/issues/comments",
        params={"sort": "created", "direction": "desc", "since": since},
        settings=s,
    ):
        cid = item.get("id")
        if cid is None:
            continue
        parent = _num_from_url(item.get("issue_url", ""), _ISSUE_NUM)
        docs.append(_comment_doc(
            full, cid, item,
            source_type="issue_comment",
            parent_number=parent,
        ))

    # PR review / inline comments.
    for item in paginate(
        f"/repos/{owner}/{name}/pulls/comments",
        params={"sort": "created", "direction": "desc", "since": since},
        settings=s,
    ):
        cid = item.get("id")
        if cid is None:
            continue
        parent = _num_from_url(item.get("pull_request_url", ""), _PR_NUM)
        docs.append(_comment_doc(
            full, cid, item,
            source_type="review_comment",
            parent_number=parent,
            path=item.get("path"),
            line=item.get("line") or item.get("original_line"),
        ))

    return docs


def _comment_doc(
    full: str,
    cid: int,
    item: dict,
    *,
    source_type: str,
    parent_number: int | None,
    path: str | None = None,
    line: int | None = None,
) -> Document:
    body = item.get("body") or ""
    author = (item.get("user") or {}).get("login")
    extra = {
        "repo": full,
        "stream": COMMENTS_STREAM,
        "source_type": source_type,
        "comment_id": cid,
        "parent_number": parent_number,
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "source": item.get("html_url"),
        "author": author,
    }
    if path is not None:
        extra["path"] = path
    if line is not None:
        extra["line"] = line
    return Document(
        doc_id=f"{full}/comment/{cid}",
        text=body.strip(),
        excluded_embed_metadata_keys=list(EMBED_EXCLUDED_KEYS),
        metadata=extra,
    )