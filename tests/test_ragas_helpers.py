"""Unit tests for RAGAS runner pure helpers: _mean and _refused."""

from __future__ import annotations


from rag.evals.run_ragas import _mean, _refused


def test_mean_basic():
    assert _mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_ignores_none_and_nan():
    assert _mean([1.0, None, 3.0]) == 2.0
    assert _mean([2.0, float("nan"), 4.0]) == 3.0


def test_mean_empty_is_zero():
    assert _mean([]) == 0.0
    assert _mean([None, float("nan")]) == 0.0


def test_mean_rounds_to_four_places():
    # (1/3) -> 0.3333
    assert _mean([0.0, 1.0, 0.0]) == round(1 / 3, 4)


def test_refused_detects_decline_phrases():
    assert _refused("I don't have information about that.") is True
    assert _refused("The context does not contain that; it is out of scope.") is True
    assert _refused("I cannot answer from the provided context.") is True


def test_refused_on_empty_or_none():
    assert _refused(None) is True
    assert _refused("") is True


def test_not_refused_on_real_answer():
    assert _refused("Use client.messages.stream(...) to stream events.") is False
