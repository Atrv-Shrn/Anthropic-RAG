"""Trim config/golden_set.json to the repos currently in config/repos.yaml (+ negatives).

Keeps every negative (out-of-scope) item and every in-scope item whose repo is still
in the configured keep-set; drops the rest. Writes a .bak backup first. Idempotent.

    uv run --no-sync python scripts/trim_golden_set.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GOLDEN = _ROOT / "config" / "golden_set.json"


def main() -> int:
    from config.repos import load_repos

    _org, repo_names = load_repos()
    keep = set(repo_names)

    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = data["items"]
    kept = [i for i in items if i.get("negative") or i.get("repo") in keep]
    dropped = [i for i in items if i not in kept]

    seeded = [i for i in kept if not i.get("negative")]
    negatives = [i for i in kept if i.get("negative")]
    print(f"keep-set repos: {sorted(keep)}")
    print(f"total {len(items)} -> kept {len(kept)} (seeded {len(seeded)}, negatives {len(negatives)}); dropped {len(dropped)}")

    # Backup once (don't clobber an existing .bak from a prior run).
    bak = GOLDEN.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(GOLDEN, bak)
        print(f"backup written: {bak}")

    data["items"] = kept
    GOLDEN.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {GOLDEN} with {len(kept)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
