"""Hand-curated golden Q/A set (~50-100 pairs, including negative cases).

The canonical set lives in ``config/golden_set.json`` as a list of items, each:

    {
      "id":         stable identifier,
      "query":      the question asked of the pipeline,
      "reference":  the ideal/ground-truth answer (used by correctness + answer
                    relevancy; for negatives it describes what the system *should*
                    do rather than a factual answer),
      "repo":       the Anthropic repo that grounds it (null for out-of-scope),
      "tags":       free-form labels,
      "negative":   true => out-of-scope; the system should NOT ground an answer
    }

Both ``run_ragas.py`` and ``run_native_evals.py`` load through this module so the
RAGAS baseline and the native-LLamaIndex baseline share identical ground truth and
are directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_SET_PATH = Path("config/golden_set.json")


@dataclass(frozen=True)
class GoldenItem:
    """One golden Q/A pair."""

    id: str
    query: str
    reference: str
    repo: str | None
    tags: tuple[str, ...] = field(default_factory=tuple)
    negative: bool = False

    @property
    def is_in_scope(self) -> bool:
        return not self.negative


def _from_dict(raw: dict) -> GoldenItem:
    return GoldenItem(
        id=raw["id"],
        query=raw["query"],
        reference=raw["reference"],
        repo=raw.get("repo"),
        tags=tuple(raw.get("tags", []) or []),
        negative=bool(raw.get("negative", False)),
    )


def load_golden_set(path: str | Path | None = None) -> list[GoldenItem]:
    """Load and return the curated golden Q/A pairs (in file order)."""
    p = Path(path) if path else GOLDEN_SET_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    return [_from_dict(i) for i in items]


def write_golden_set(items: list[GoldenItem], path: str | Path | None = None) -> Path:
    """Serialize golden items back to JSON (used when re-curating programmatically)."""
    p = Path(path) if path else GOLDEN_SET_PATH
    payload = {
        "_comment": (
            "Hand-curated golden Q/A set for the Anthropic GitHub RAG pipeline. "
            "Shared by run_ragas.py and run_native_evals.py."
        ),
        "items": [
            {
                "id": it.id,
                "query": it.query,
                "reference": it.reference,
                "repo": it.repo,
                "tags": list(it.tags),
                "negative": it.negative,
            }
            for it in items
        ],
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


if __name__ == "__main__":
    # `python -m rag.evals.golden_set` -> quick inventory.
    items = load_golden_set()
    neg = [i for i in items if i.negative]
    print(f"golden set: {len(items)} items ({len(items) - len(neg)} in-scope, {len(neg)} negative)")
    repos: dict[str | None, int] = {}
    for it in items:
        repos[it.repo] = repos.get(it.repo, 0) + 1
    for r, n in sorted(repos.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        print(f"  {r}: {n}")