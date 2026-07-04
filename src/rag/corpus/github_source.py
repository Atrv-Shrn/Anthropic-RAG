"""GitHub corpus source: docs-only file reader + issues/PRs (watermark-windowed).

Scope (see docs/SPEC.md): docs + issues/PRs/comments only — **no source code** in
the corpus. Two fetchers:

- ``fetch_docs``: LlamaHub ``GithubRepositoryReader`` with an extension INCLUDE
  filter on documentation types (``.md .mdx .rst .txt .ipynb``). This is the main
  scoping guard — every other extension (``.py .ts .js ...``) is excluded, so no
  source codebases leak into Qdrant. READMEs anywhere in the tree are caught
  automatically.

- ``fetch_issues_prs``: GitHub REST ``/issues`` endpoint (returns issues AND pull
  requests, since PRs are issues) via ``_github_http.paginate``, with
  ``sort=updated`` and ``since=<watermark>`` so only items updated at or after the
  watermark come back. We go direct to the API (not the LlamaHub issues reader)
  because that reader exposes no ``since`` and does not return ``updated_at``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from llama_index.core.schema import Document
from llama_index.readers.github import GithubClient, GithubRepositoryReader

from config.settings import Settings, get_settings
from rag.corpus._github_http import paginate
from rag.corpus.watermarks import get_watermark

# Documentation extensions we ingest. Everything else (source code, configs, etc.)
# is excluded — this keeps whole codebases out of Qdrant.
DOC_EXTENSIONS = [".md", ".mdx", ".rst", ".txt", ".ipynb"]

# Streams used as watermark keys.
DOCS_STREAM = "docs"
ISSUES_PRS_STREAM = "issues_prs"

# Metadata keys we attach to every document. These are kept for source attribution
# and filtering but excluded from the *embedded* text so they don't pollute the
# nomic vectors (they remain visible to the LLM for citation via MetadataMode.LLM).
EMBED_EXCLUDED_KEYS = [
    "repo", "stream", "source_type", "number", "state",
    "created_at", "updated_at", "source", "labels",
]

# Seed window for the activity streams when no watermark exists yet.
SEED_WINDOW_DAYS = 7


def _seed_since() -> str:
    """ISO timestamp for ``now - SEED_WINDOW_DAYS`` (the seed activity window)."""
    return (datetime.now(timezone.utc) - timedelta(days=SEED_WINDOW_DAYS)).isoformat()


def fetch_docs(repo: str, org: str | None = None, settings: Settings | None = None) -> list[Document]:
    """Fetch documentation documents for ``org/repo`` (docs extensions only)."""
    s = settings or get_settings()
    org = org or s.github_org
    client = GithubClient(github_token=s.github_token or None, verbose=False)
    reader = GithubRepositoryReader(
        github_client=client,
        owner=org,
        repo=repo,
        use_parser=False,
        filter_file_extensions=(DOC_EXTENSIONS, GithubRepositoryReader.FilterType.INCLUDE),
        verbose=False,
        fail_on_error=False,
    )
    docs = reader.load_data(branch="main")
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("repo", f"{org}/{repo}")
        d.metadata.setdefault("stream", DOCS_STREAM)
        d.metadata.setdefault("source_type", "docs")
        d.excluded_embed_metadata_keys = list(EMBED_EXCLUDED_KEYS)
    return docs


def fetch_issues_prs(
    repo: str,
    org: str | None = None,
    since: str | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Fetch issues + PRs for ``org/repo`` updated at or after ``since``.

    ``since`` defaults to the seed window (``now - 7d``) when no watermark is set.
    """
    s = settings or get_settings()
    org = org or s.github_org
    if since is None:
        since = get_watermark(repo, ISSUES_PRS_STREAM) or _seed_since()

    path = f"/repos/{org}/{repo}/issues"
    params = {"state": "all", "sort": "updated", "direction": "desc", "since": since}

    docs: list[Document] = []
    for item in paginate(path, params=params, settings=s):
        # GitHub returns PRs in the issues endpoint too; they carry a pull_request key.
        is_pr = "pull_request" in item
        number = item.get("number")
        if number is None:
            continue
        updated_at = item.get("updated_at") or item.get("created_at")
        # Defensive: API `since` is inclusive, but filter again client-side.
        if updated_at and updated_at < since:
            continue
        title = item.get("title") or ""
        body = item.get("body") or ""
        doc = Document(
            doc_id=f"{org}/{repo}#{number}",
            text=f"{title}\n{body}".strip(),
            excluded_embed_metadata_keys=list(EMBED_EXCLUDED_KEYS),
            metadata={
                "repo": f"{org}/{repo}",
                "stream": ISSUES_PRS_STREAM,
                "source_type": "pr" if is_pr else "issue",
                "number": number,
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "updated_at": updated_at,
                "source": item.get("html_url"),
                "labels": [lbl.get("name") for lbl in item.get("labels") or []],
            },
        )
        docs.append(doc)
    return docs