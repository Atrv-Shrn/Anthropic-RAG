"""Unit tests for the dependency-free eval metrics (ROUGE-L, BLEU, cosine)."""

from __future__ import annotations

import math

from rag.evals.non_llm_metrics import (
    _cosine,
    bleu,
    non_llm_scores,
    rouge_l,
    semantic_similarity,
)


def test_rouge_l_identical_is_one():
    r = rouge_l("the quick brown fox", "the quick brown fox")
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f"] == 1.0


def test_rouge_l_disjoint_is_zero():
    r = rouge_l("alpha beta gamma", "delta epsilon zeta")
    assert r["f"] == 0.0


def test_rouge_l_partial_overlap():
    # LCS of "a b c d" vs "a c d" is "a c d" (len 3).
    r = rouge_l("a b c d", "a c d")
    assert r["recall"] == 1.0  # all 3 reference tokens covered
    assert 0.0 < r["precision"] < 1.0  # answer has an extra token
    assert 0.0 < r["f"] < 1.0


def test_rouge_l_empty_inputs():
    assert rouge_l("", "something")["f"] == 0.0
    assert rouge_l("something", "")["f"] == 0.0


def test_bleu_identical_high():
    b = bleu("the quick brown fox jumps", "the quick brown fox jumps")
    assert b["bleu_1"] == 1.0
    assert b["bleu_4"] == 1.0


def test_bleu_disjoint_zero():
    b = bleu("alpha beta gamma delta", "one two three four")
    assert b["bleu_1"] == 0.0


def test_bleu_brevity_penalty_shrinks_short_answer():
    # A very short answer vs a long reference should be penalised on bleu_1.
    b = bleu("the", "the quick brown fox jumps over the lazy dog")
    assert b["bleu_1"] < 1.0


def test_bleu_empty():
    b = bleu("", "x y z")
    assert all(v == 0.0 for v in b.values())


def test_cosine_orthogonal_and_parallel():
    assert _cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert math.isclose(_cosine([1, 0, 0], [2, 0, 0]), 1.0)
    assert _cosine([0, 0, 0], [1, 1, 1]) == 0.0  # zero vector guard


class _StubEmbed:
    def get_text_embedding(self, text: str) -> list[float]:
        # deterministic: identical text -> identical vector
        return [float(len(text)), 1.0, 0.0]


def test_semantic_similarity_uses_embed_model():
    s = semantic_similarity("same", "same", _StubEmbed())
    assert math.isclose(s, 1.0)


def test_semantic_similarity_handles_missing_model():
    assert semantic_similarity("a", "b", None) == 0.0


def test_semantic_similarity_survives_embed_failure():
    class _Boom:
        def get_text_embedding(self, text):
            raise RuntimeError("embed down")

    assert semantic_similarity("a", "b", _Boom()) == 0.0


def test_non_llm_scores_bundles_all_keys():
    scores = non_llm_scores("the quick brown fox", "the quick brown fox", _StubEmbed())
    assert scores["rouge_l_f"] == 1.0
    assert "bleu_1" in scores and "bleu_4" in scores
    assert "semantic_similarity" in scores


def test_non_llm_scores_without_embed_omits_semantic():
    scores = non_llm_scores("a b c", "a b c", None)
    assert "semantic_similarity" not in scores
    assert scores["rouge_l_f"] == 1.0
