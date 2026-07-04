"""Native LlamaIndex evaluators: Faithfulness, Relevancy, Correctness.

Runs the three built-in LlamaIndex evaluators (judge = the same Ollama-Cloud
``deepseek-v4-pro:cloud`` LLM used for generation) over the golden set and records
per-item pass/fail + feedback alongside the RAGAS baseline so the two are
cross-comparable.

- ``FaithfulnessEvaluator``: is the answer supported by the retrieved context?
- ``RelevancyEvaluator``  : is the answer relevant to the query (given the context)?
- ``CorrectnessEvaluator``: does the answer match the ground-truth reference?

Predictions are collected once via ``eval_harness.collect_predictions`` (the same
records RAGAS uses). Baseline written to ``data/evals/native_baseline.json``.

CLI: ``python -m rag.evals.run_native_evals`` (needs OLLAMA_CLOUD_API_KEY + a seeded corpus).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from rag.evals.eval_harness import collect_predictions, write_predictions

log = logging.getLogger(__name__)

EVALS_DIR = Path("data/evals")
NATIVE_BASELINE_PATH = EVALS_DIR / "native_baseline.json"
PREDICTIONS_PATH = EVALS_DIR / "predictions.json"

METRIC_NAMES = ("faithfulness", "relevancy", "correctness")


def _pass_rate(rows: list[dict[str, Any]], metric: str) -> float:
    vals = [r.get(metric) for r in rows if r.get(metric) is not None]
    if not vals:
        return 0.0
    return round(sum(1 for v in vals if v) / len(vals), 4)


def run_native_evals(
    records: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Faithfulness/Relevancy/Correctness over the golden set; return a baseline dict."""
    from llama_index.core.evaluation import (
        CorrectnessEvaluator,
        FaithfulnessEvaluator,
        RelevancyEvaluator,
    )

    from rag.generate.llm import get_llm

    s = settings or get_settings()
    if records is None:
        records = collect_predictions(settings=s)
        write_predictions(records, str(PREDICTIONS_PATH))

    llm = get_llm(s)
    faithfulness_ev = FaithfulnessEvaluator(llm=llm)
    relevancy_ev = RelevancyEvaluator(llm=llm)
    correctness_ev = CorrectnessEvaluator(llm=llm)

    per_item: list[dict[str, Any]] = []
    for r in records:
        entry: dict[str, Any] = {
            "id": r["id"],
            "query": r["query"],
            "negative": r["negative"],
            "answer": r["answer"],
            "error": r["error"],
            "num_contexts": len(r["retrieved_contexts"]),
            "faithfulness": None,
            "relevancy": None,
            "correctness": None,
            "feedback": {"faithfulness": None, "relevancy": None, "correctness": None},
        }
        if r.get("error") or r["answer"] is None:
            per_item.append(entry)
            continue

        q, ans, ctxs, ref = r["query"], r["answer"], r["retrieved_contexts"], r["reference"]
        try:
            f = faithfulness_ev.evaluate(query=q, response=ans, contexts=ctxs)
            entry["faithfulness"] = bool(f.passing)
            entry["feedback"]["faithfulness"] = f.feedback
        except Exception as exc:  # noqa: BLE001
            log.exception("faithfulness failed: %s", r["id"])
            entry["feedback"]["faithfulness"] = f"eval error: {exc!r}"
        try:
            rel = relevancy_ev.evaluate(query=q, response=ans, contexts=ctxs)
            entry["relevancy"] = bool(rel.passing)
            entry["feedback"]["relevancy"] = rel.feedback
        except Exception as exc:  # noqa: BLE001
            log.exception("relevancy failed: %s", r["id"])
            entry["feedback"]["relevancy"] = f"eval error: {exc!r}"
        try:
            cor = correctness_ev.evaluate(query=q, response=ans, reference=ref)
            entry["correctness"] = bool(cor.passing)
            entry["feedback"]["correctness"] = cor.feedback
        except Exception as exc:  # noqa: BLE001
            log.exception("correctness failed: %s", r["id"])
            entry["feedback"]["correctness"] = f"eval error: {exc!r}"
        per_item.append(entry)

    in_scope = [e for e in per_item if not e["negative"]]
    negatives = [e for e in per_item if e["negative"]]

    def agg(group: list[dict[str, Any]]) -> dict[str, float]:
        return {m: _pass_rate(group, m) for m in METRIC_NAMES}

    baseline = {
        "n_records": len(records),
        "n_evaluated": sum(1 for e in per_item if e["faithfulness"] is not None),
        "n_in_scope": len(in_scope),
        "n_negative": len(negatives),
        "overall": agg(per_item),
        "in_scope": agg(in_scope),
        "negative": agg(negatives),
        "per_item": per_item,
    }

    out = Path(out_path) if out_path else NATIVE_BASELINE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("native baseline written to %s", out)
    log.info("=== native baseline ===")
    log.info("in-scope (n=%d): %s", baseline["n_in_scope"], baseline["in_scope"])
    log.info("negative  (n=%d): %s", baseline["n_negative"], baseline["negative"])
    return baseline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run_native_evals()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())