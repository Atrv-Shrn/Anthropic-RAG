"""Run both judge suites (RAGAS + native LlamaIndex) over an existing predictions.json.

Loads the shared prediction records collected earlier and passes records= to both
runners so no pipeline queries are re-issued — only the judge-LLM calls happen here.
Writes data/evals/ragas_baseline.json and native_baseline.json and prints a summary.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("judges")

PREDS = Path("data/evals/predictions.json")


def main() -> int:
    records = json.loads(PREDS.read_text(encoding="utf-8"))
    log.info("loaded %d prediction records", len(records))

    from rag.evals.run_ragas import run_ragas

    t0 = time.time()
    ragas_baseline = run_ragas(records=records)
    log.info("RAGAS done in %.1fs", time.time() - t0)

    from rag.evals.run_native_evals import run_native_evals

    t1 = time.time()
    native_baseline = run_native_evals(records=records)
    log.info("native done in %.1fs", time.time() - t1)

    print("\n=================== FULL EVAL SUMMARY ===================")
    print(f"records: {len(records)}")
    print("\n--- RAGAS in_scope ---")
    print(json.dumps(ragas_baseline["in_scope"], indent=2))
    print("--- RAGAS negative ---")
    print(json.dumps(ragas_baseline["negative"], indent=2))
    print("\n--- NATIVE in_scope (pass rates) ---")
    print(json.dumps(native_baseline["in_scope"], indent=2))
    print("--- NATIVE negative (pass rates) ---")
    print(json.dumps(native_baseline["negative"], indent=2))
    print("========================================================")
    print("ALL JUDGES DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
