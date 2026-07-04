"""Shared eval harness: run the pipeline over the golden set and collect predictions.

Both ``run_ragas.py`` and ``run_native_evals.py`` need the same per-question artifacts:
the generated answer and the retrieved context chunks. Running retrieval once here and
reusing the records keeps the two eval suites comparable and avoids double work.

A record is a plain dict with:
    id, query, reference, negative, answer, retrieved_contexts (list[str]),
    source_doc_ids (list[str]), error (str | None)
"""

from __future__ import annotations

import logging
from typing import Any

from rag.evals.golden_set import GoldenItem, load_golden_set

log = logging.getLogger(__name__)


def _source_texts_and_ids(source_nodes) -> tuple[list[str], list[str]]:
    """Extract (chunk texts, parent doc_ids) from a response's source nodes."""
    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for n in source_nodes or []:
        node = n.node
        texts.append(node.get_content().strip())
        pid = getattr(node, "ref_doc_id", None) or node.node_id
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return texts, ids


def collect_predictions(
    items: list[GoldenItem] | None = None,
    settings: Any | None = None,
    query_engine: Any | None = None,
) -> list[dict[str, Any]]:
    """Run the query engine over each golden item and collect predictions.

    The query engine is built once (it loads the reranker and connects to the stores)
    and reused across items. A failing query is recorded with ``error`` set and does
    not abort the run.
    """
    from rag.generate.query_engine import build_query_engine

    if items is None:
        items = load_golden_set()
    qe = query_engine or build_query_engine(settings)

    records: list[dict[str, Any]] = []
    for it in items:
        rec: dict[str, Any] = {
            "id": it.id,
            "query": it.query,
            "reference": it.reference,
            "negative": it.negative,
            "answer": None,
            "retrieved_contexts": [],
            "source_doc_ids": [],
            "error": None,
        }
        try:
            response = qe.query(it.query)
            rec["answer"] = str(response).strip()
            texts, ids = _source_texts_and_ids(response.source_nodes)
            rec["retrieved_contexts"] = texts
            rec["source_doc_ids"] = ids
        except Exception as exc:  # noqa: BLE001 - eval must survive a single failure
            log.exception("query failed: %s", it.id)
            rec["error"] = repr(exc)
        records.append(rec)
    return records


def write_predictions(records: list[dict[str, Any]], path: str) -> None:
    """Persist prediction records to JSON (for offline / out-of-band eval)."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")