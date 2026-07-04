"""Verify the incremental hourly tick: watermarks before -> trigger_now() -> after.

Confirms the scheduler's job (rag.scheduler.hourly.trigger_now, which runs
run_incremental_ingest) fetches only items newer than each stream's watermark, embeds
just the new ones (UPSERTS dedup), and advances the watermarks. Run detached.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    from rag.corpus.watermarks import all_watermarks
    from rag.scheduler.hourly import trigger_now

    before = {(r, s): ts for r, s, ts in all_watermarks()}
    print("=== WATERMARKS BEFORE ===", flush=True)
    for (r, s), ts in sorted(before.items()):
        print(f"  {r}/{s}: {ts}", flush=True)

    print("\n=== TRIGGERING trigger_now() ===", flush=True)
    stats = trigger_now()

    print("\n=== TICK STATS (per repo) ===", flush=True)
    total_embedded = 0
    for repo, st in stats.items():
        print(f"  {repo}: {st}", flush=True)
        total_embedded += st.get("embedded", 0)

    print("\n=== WATERMARKS AFTER ===", flush=True)
    advanced = 0
    for r, s, ts in sorted(all_watermarks()):
        changed = ts != before.get((r, s))
        advanced += int(changed)
        print(f"  {r}/{s}: {ts} {'(ADVANCED)' if changed else '(unchanged)'}", flush=True)

    print(
        f"\nTICK VERIFIED: {total_embedded} new nodes embedded, "
        f"{advanced} watermark(s) advanced.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
