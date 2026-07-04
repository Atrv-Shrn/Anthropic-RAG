"""Prune the corpus down to the repos currently in config/repos.yaml.

Removes every doc (and its vectors) whose `repo` metadata is not in the configured
keep-set, from both Qdrant and the Redis docstore, and clears their watermarks. Idempotent
and safe to re-run. Run detached for large corpora.

    uv run --no-sync python scripts/prune_repos.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("prune")

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    from qdrant_client import QdrantClient, models

    from config.repos import load_repos
    from config.settings import get_settings
    from rag.corpus.watermarks import DEFAULT_DB, _connect, all_watermarks
    from rag.ingest.pipeline import get_docstore

    s = get_settings()
    org, repo_names = load_repos()
    keep = {f"{org}/{r}" for r in repo_names}
    log.info("keep-set: %s", sorted(keep))

    ds = get_docstore(s)
    # Generous timeout: deletes can be slow when the box is under load (e.g. a
    # concurrent image build), and a filter-delete rewrites segments.
    c = QdrantClient(url=s.qdrant_url, timeout=300)

    # Which repos are present but not kept?
    present = {(n.metadata or {}).get("repo") for n in ds.docs.values()}
    remove_repos = {r for r in present if r not in keep and r is not None}
    log.info("removing repos: %s", sorted(remove_repos))

    # 1. Delete their vectors from Qdrant (by repo payload).
    before = c.get_collection(s.qdrant_collection).points_count
    for repo in remove_repos:
        c.delete(
            collection_name=s.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo))]
                )
            ),
        )
    after = c.get_collection(s.qdrant_collection).points_count
    log.info("qdrant points: %d -> %d", before, after)

    # 2. Delete their docs from the Redis docstore.
    victims = [did for did, n in ds.docs.items() if (n.metadata or {}).get("repo") in remove_repos]
    for did in victims:
        try:
            ds.delete_ref_doc(did, raise_error=False)
        except Exception:  # noqa: BLE001
            pass
        try:
            ds.delete_document(did, raise_error=False)
        except Exception:  # noqa: BLE001
            pass
    log.info("docstore docs: removed %d, now %d", len(victims), len(ds.docs))

    # 3. Clear watermarks for removed repos.
    conn = _connect(DEFAULT_DB)
    removed_wm = 0
    for repo, stream, _ in all_watermarks():
        if f"{org}/{repo}" in remove_repos or repo in {r.split('/')[-1] for r in remove_repos}:
            conn.execute("DELETE FROM watermarks WHERE repo=? AND stream=?", (repo, stream))
            removed_wm += 1
    conn.commit()
    conn.close()
    log.info("watermarks removed: %d", removed_wm)

    # 4. Report final state.
    from collections import Counter

    final = Counter((n.metadata or {}).get("repo") for n in ds.docs.values())
    log.info("FINAL corpus: qdrant=%d docstore=%d repos=%s",
             c.get_collection(s.qdrant_collection).points_count, len(ds.docs), dict(final))
    log.info("PRUNE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
