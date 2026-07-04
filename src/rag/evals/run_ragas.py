"""RAGAS eval runner: LLM-judged metrics + cheap non-LLM metrics.

LLM metrics (faithfulness, answer_relevancy, context_precision, context_recall) use
``deepseek-v4-pro:cloud`` as the judge (the same Ollama-Cloud LLM the pipeline uses for
generation, via ``ragas.llms.LlamaIndexLLMWrapper``) and local ``nomic-embed-text``
embeddings for the embedding-based metrics (via ``ragas.embeddings.LlamaIndexEmbeddingsWrapper``).

Non-LLM metrics (ROUGE-L, BLEU-1..4, nomic-cosine semantic similarity) come from
``rag.evals.non_llm_metrics`` and need no judge.

The run collects predictions once via ``eval_harness.collect_predictions``, runs RAGAS
over them, then layers the non-LLM scores on top, and writes a baseline JSON to
``data/evals/ragas_baseline.json``. In-scope and negative (out-of-scope) items are
reported separately so we can see whether the system correctly declines negatives.

CLI: ``python -m rag.evals.run_ragas``  (needs OLLAMA_CLOUD_API_KEY + a seeded corpus).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import statistics as stats
import sys
import types
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from rag.evals.eval_harness import collect_predictions, write_predictions
from rag.evals.non_llm_metrics import non_llm_scores

log = logging.getLogger(__name__)


def _ensure_ragas_importable() -> None:
    """Work around ragas eagerly importing ``langchain_community.chat_models.vertexai``.

    ragas (0.4.x) imports ``ChatVertexAI`` from ``langchain_community.chat_models.vertexai``
    at module load, but langchain-community >=0.4 removed that submodule (VertexAI moved to
    the ``langchain-google-vertexai`` partner package). This pipeline never uses VertexAI —
    the judge is our LlamaIndex/Ollama-Cloud LLM — so a stub class lets ragas finish
    importing. The shim is a no-op when the real module is present, so it is safe across
    langchain-community versions and ages out automatically once ragas lazy-imports.
    """
    mod = "langchain_community.chat_models.vertexai"
    if mod in sys.modules or importlib.util.find_spec(mod) is not None:
        return
    stub = types.ModuleType(mod)

    class ChatVertexAI:  # pragma: no cover - placeholder, never instantiated here
        """Stub; ragas only needs the name importable for its LLM factory registry."""

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod] = stub

EVALS_DIR = Path("data/evals")
RAGAS_BASELINE_PATH = EVALS_DIR / "ragas_baseline.json"
PREDICTIONS_PATH = EVALS_DIR / "predictions.json"

# Phrases that indicate the system is (correctly) declining to answer from the corpus.
_REFUSAL_MARKERS = (
    "i don't have", "i do not have", "no information", "not in the corpus",
    "not answerable", "cannot answer", "can't answer", "outside the", "out of scope",
    "not present", "isn't in", "is not in", "don't have information",
)


def _refused(answer: str | None) -> bool:
    if not answer:
        return True
    a = answer.lower()
    return any(m in a for m in _REFUSAL_MARKERS)


def _build_dataset(records: list[dict[str, Any]]):
    """Build a RAGAS ``EvaluationDataset`` from prediction records."""
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = []
    for r in records:
        # RAGAS needs a response and at least empty contexts. Skip records that errored
        # at retrieval time (no answer to judge).
        if r.get("error") or r.get("answer") is None:
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["query"],
                retrieved_contexts=r["retrieved_contexts"] or [],
                response=r["answer"],
                reference=r["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return round(stats.fmean(vals), 4) if vals else 0.0


def run_ragas(
    records: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the RAGAS suite + non-LLM metrics and return a baseline dict.

    ``records`` defaults to a fresh ``collect_predictions()`` run (queries the live
    pipeline). The baseline is written to ``out_path`` (default ``data/evals/ragas_baseline.json``).
    """
    _ensure_ragas_importable()
    from ragas import evaluate
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.llms import LlamaIndexLLMWrapper
    # Use the legacy ragas.metrics classes (not ragas.metrics.collections): the new
    # "collections" metrics require an InstructorLLMWrapper, but we wrap our existing
    # LlamaIndex Ollama-Cloud LLM with LlamaIndexLLMWrapper, which only the legacy
    # metric implementations accept. The legacy path is deprecated for removal in
    # ragas v1.0 but functional in 0.4.x; suppress the import-time deprecation noise.
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

    from rag.generate.llm import get_llm
    from rag.ingest.embeddings import get_embed_model

    s = settings or get_settings()
    if records is None:
        records = collect_predictions(settings=s)
        write_predictions(records, str(PREDICTIONS_PATH))

    embed_model = get_embed_model(s)
    evaluator_llm = LlamaIndexLLMWrapper(llm=get_llm(s))
    evaluator_emb = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]
    # RAGAS keys its result dataframe columns by each metric's `.name` (snake_case,
    # e.g. "faithfulness"), NOT the class name. Read the columns by `.name` — using the
    # class name silently misses every column and zeros out all LLM metrics.
    metric_names = [m.name for m in metrics]

    dataset = _build_dataset(records)
    log.info("RAGAS dataset: %d samples (of %d records)", len(dataset), len(records))

    ragas_result = evaluate(dataset=dataset, metrics=metrics)
    df = ragas_result.to_pandas()
    missing = [n for n in metric_names if n not in df.columns]
    if missing:
        log.warning("RAGAS df missing expected metric columns %s; have %s",
                    missing, list(df.columns))

    # Map RAGAS scores back to records by row order (RAGAS preserves input sample order;
    # we only skipped records that errored at retrieval). Preserve NaN as None rather
    # than coercing to 0.0 — a real judge failure must not masquerade as a zero score.
    def _num(v: Any) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f  # drop NaN

    ragas_scores: list[dict[str, float | None]] = []
    for _, row in df.iterrows():
        ragas_scores.append({name: _num(row.get(name)) for name in metric_names})

    # Layer non-LLM scores per record and assemble the full per-item breakdown.
    per_item: list[dict[str, Any]] = []
    ragas_idx = 0
    for r in records:
        entry: dict[str, Any] = {
            "id": r["id"],
            "query": r["query"],
            "negative": r["negative"],
            "answer": r["answer"],
            "error": r["error"],
            "source_doc_ids": r["source_doc_ids"],
            "num_contexts": len(r["retrieved_contexts"]),
            "refused": _refused(r["answer"]),
        }
        if r.get("error") or r["answer"] is None:
            entry["ragas"] = {}
            entry["non_llm"] = {}
        else:
            entry["ragas"] = ragas_scores[ragas_idx] if ragas_idx < len(ragas_scores) else {}
            ragas_idx += 1
            entry["non_llm"] = non_llm_scores(r["answer"], r["reference"], embed_model)
        per_item.append(entry)

    in_scope = [e for e in per_item if not e["negative"]]
    negatives = [e for e in per_item if e["negative"]]

    def _agg(group: list[dict[str, Any]], key: str) -> dict[str, float]:
        """Mean of every numeric field found under ``e[key]`` across the group.

        Generic on purpose: with ``key="ragas"`` it aggregates the RAGAS metric
        scores; with ``key="non_llm"`` it aggregates the ROUGE/BLEU/semantic scores.
        The two key sets are disjoint, so merging ``{**_agg(g, "ragas"), **_agg(g,
        "non_llm")}`` never lets one clobber the other.
        """
        out: dict[str, float] = {}
        if not group:
            return out
        keys: set[str] = set()
        for e in group:
            keys.update((e.get(key) or {}).keys())
        for k in keys:
            out[k] = _mean([e.get(key, {}).get(k) for e in group if e.get(key)])
        return out

    baseline = {
        "n_records": len(records),
        "n_evaluated": len(dataset),
        "n_in_scope": len(in_scope),
        "n_negative": len(negatives),
        "ragas_metrics": metric_names,
        "overall": {**_agg(per_item, "ragas"), **_agg(per_item, "non_llm")},
        "in_scope": {**_agg(in_scope, "ragas"), **_agg(in_scope, "non_llm")},
        "negative": {
            **_agg(negatives, "ragas"),
            **_agg(negatives, "non_llm"),
            "refusal_rate": _mean([1.0 if e["refused"] else 0.0 for e in negatives]),
        },
        "per_item": per_item,
    }

    out = Path(out_path) if out_path else RAGAS_BASELINE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("RAGAS baseline written to %s", out)
    _log_summary(baseline)
    return baseline


def _log_summary(b: dict[str, Any]) -> None:
    log.info("=== RAGAS baseline ===")
    log.info("in-scope (n=%d): %s", b["n_in_scope"], b["in_scope"])
    log.info("negative  (n=%d): %s (refusal_rate=%.2f)",
             b["n_negative"], b["negative"], b["negative"].get("refusal_rate", 0.0))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    run_ragas()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())