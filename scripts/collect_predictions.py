"""Checkpointed prediction collection over the full golden set.

Writes each prediction to data/evals/predictions.json as it completes, so a kill
(task-lifetime limit) costs at most the in-flight item and the run resumes by
skipping ids already present. Query engine is built once and reused.
"""

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
log = logging.getLogger("collect")

from rag.evals.eval_harness import _source_texts_and_ids
from rag.evals.golden_set import load_golden_set
from rag.generate.query_engine import build_query_engine

OUT = Path("data/evals/predictions.json")


def load_done() -> dict:
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text(encoding="utf-8"))
            return {r["id"]: r for r in data}
        except Exception:
            return {}
    return {}


def main() -> int:
    items = load_golden_set()
    done = load_done()
    log.info("golden %d items; %d already done", len(items), len(done))

    qe = build_query_engine()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, it in enumerate(items, 1):
        if it.id in done:
            continue
        rec = {
            "id": it.id, "query": it.query, "reference": it.reference,
            "negative": it.negative, "answer": None,
            "retrieved_contexts": [], "source_doc_ids": [], "error": None,
        }
        t0 = time.time()
        try:
            resp = qe.query(it.query)
            rec["answer"] = str(resp).strip()
            texts, ids = _source_texts_and_ids(resp.source_nodes)
            rec["retrieved_contexts"] = texts
            rec["source_doc_ids"] = ids
        except Exception as exc:
            log.exception("query failed: %s", it.id)
            rec["error"] = repr(exc)
        done[it.id] = rec
        # checkpoint after every item (preserve golden order)
        ordered = [done[x.id] for x in items if x.id in done]
        OUT.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("[%d/%d] %s  ans_len=%d ctxs=%d  %.1fs",
                 i, len(items), it.id, len(rec["answer"] or ""),
                 len(rec["retrieved_contexts"]), time.time() - t0)

    log.info("ALL PREDICTIONS DONE: %d", len(done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
