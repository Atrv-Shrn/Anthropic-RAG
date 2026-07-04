"""One-off: seed docs (only) for the given repos, one repo at a time, persisting
after each. Run detached so it isn't bound to a short task-lifetime window:

    uv run --no-sync python scripts/seed_docs.py skills anthropic-cookbook courses ...

Docs answer the golden-set questions; the huge issue streams (esp. claude-code's
7k+/7d) are intentionally out of scope for this quick corpus fill.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root (for `config`) and `src` (for `rag`) are importable when run
# as a detached script from any CWD.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("seed_docs")


def main(argv: list[str]) -> int:
    from config.settings import get_settings
    from rag.corpus.github_source import DOCS_STREAM, fetch_docs
    from rag.corpus.watermarks import set_watermark
    from rag.ingest.pipeline import run_ingest

    s = get_settings()
    repos = argv or []
    for repo in repos:
        t0 = time.time()
        try:
            docs = fetch_docs(repo, org=s.github_org, settings=s)
            emb = run_ingest(docs, settings=s) if docs else []
            set_watermark(repo, DOCS_STREAM, datetime.now(timezone.utc).isoformat())
            log.info("SEEDED %s docs=%d embedded=%d in %.1fs", repo, len(docs), len(emb), time.time() - t0)
        except Exception:
            log.exception("FAILED %s", repo)
    log.info("ALL DONE: %s", repos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
