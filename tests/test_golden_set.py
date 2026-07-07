"""Unit tests for the golden-set loader/writer and the shipped golden_set.json."""

from __future__ import annotations

import json

from rag.evals.golden_set import (
    GoldenItem,
    load_golden_set,
    write_golden_set,
)


def _sample(tmp_path):
    data = {
        "items": [
            {
                "id": "q1",
                "query": "How do I stream?",
                "reference": "Use messages.stream.",
                "repo": "anthropic-sdk-python",
                "tags": ["sdk", "streaming"],
                "negative": False,
            },
            {
                "id": "neg1",
                "query": "Capital of France?",
                "reference": "Out of scope.",
                "repo": None,
                "tags": ["out-of-scope"],
                "negative": True,
            },
        ]
    }
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_parses_items(tmp_path):
    items = load_golden_set(_sample(tmp_path))
    assert len(items) == 2
    assert isinstance(items[0], GoldenItem)
    assert items[0].id == "q1"
    assert items[0].tags == ("sdk", "streaming")
    assert items[0].is_in_scope is True
    assert items[1].negative is True
    assert items[1].is_in_scope is False
    assert items[1].repo is None


def test_roundtrip_write_then_load(tmp_path):
    items = load_golden_set(_sample(tmp_path))
    out = tmp_path / "written.json"
    write_golden_set(items, out)
    reloaded = load_golden_set(out)
    assert [i.id for i in reloaded] == [i.id for i in items]
    assert reloaded[0].tags == items[0].tags


def test_shipped_golden_set_is_valid():
    """The real config/golden_set.json must load and be internally consistent."""
    items = load_golden_set()  # default path
    assert len(items) >= 10
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids)), "golden set has duplicate ids"
    for it in items:
        assert it.query.strip()
        assert it.reference.strip()
        # Negatives are out-of-scope (repo None); in-scope items name a repo.
        if it.negative:
            assert it.repo is None
        else:
            assert it.repo is not None
