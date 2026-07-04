"""Non-LLM eval metrics: ROUGE-L, BLEU-n, and embedding cosine semantic similarity.

These complement RAGAS's LLM-judged metrics with cheap, deterministic scores that need
no judge model. ROUGE-L and BLEU are pure-Python (no extra deps); semantic similarity
reuses the same local nomic embeddings the pipeline uses for retrieval so the score is
meaningful and consistent with the rest of the stack.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

# --- tokenization -----------------------------------------------------------

_PUNCT = set(".,;:!?\"'`()[]{}<>|/\\@#$%^&*~+=_\n\r\t")


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if t and t not in _PUNCT]


# --- ROUGE-L (LCS based) -----------------------------------------------------


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(answer: str, reference: str) -> dict[str, float]:
    """ROUGE-L F1 (recall, precision, f) over whitespace tokens."""
    a, b = _tokens(answer), _tokens(reference)
    if not a or not b:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    lcs = _lcs_length(a, b)
    precision = lcs / len(a)
    recall = lcs / len(b)
    f = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f": f}


# --- BLEU-n (with brevity penalty) ------------------------------------------


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(answer: str, reference: str, max_n: int = 4) -> dict[str, float]:
    """BLEU-1..max_n with brevity penalty against a single reference."""
    a, b = _tokens(answer), _tokens(reference)
    out: dict[str, float] = {}
    if not a or not b:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    bp = 1.0 if len(a) >= len(b) else math.exp(1 - len(b) / len(a))
    for n in range(1, max_n + 1):
        a_counts = _ngram_counts(a, n)
        b_counts = _ngram_counts(b, n)
        if not a_counts:
            out[f"bleu_{n}"] = 0.0
            continue
        matches = sum(min(c, b_counts.get(ng, 0)) for ng, c in a_counts.items())
        total = sum(a_counts.values())
        precision = matches / total if total else 0.0
        out[f"bleu_{n}"] = bp * precision
    return out


# --- semantic similarity (nomic cosine) --------------------------------------


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if not nu or not nv:
        return 0.0
    return dot / (nu * nv)


def semantic_similarity(answer: str, reference: str, embed_model: Any) -> float:
    """Cosine similarity between nomic embeddings of answer and reference.

    Returns 0.0 (rather than raising) if the embed model is unreachable so a single
    embedding failure can't abort the whole eval run.
    """
    if not answer or not reference or embed_model is None:
        return 0.0
    # Both are plain text -> _get_text_embedding uses the search_document: prefix
    # (the document instruction). Cosine is prefix-invariant, so this is fine.
    try:
        a = embed_model.get_text_embedding(answer)
        b = embed_model.get_text_embedding(reference)
    except Exception:  # noqa: BLE001 - eval must survive an embed-model outage
        return 0.0
    return _cosine(a, b)


def non_llm_scores(answer: str, reference: str, embed_model: Any | None = None) -> dict[str, float]:
    """Bundle ROUGE-L, BLEU, and (optionally) semantic similarity for one pair."""
    scores: dict[str, float] = {**{f"rouge_l_{k}": v for k, v in rouge_l(answer, reference).items()},
                                **bleu(answer, reference)}
    if embed_model is not None:
        scores["semantic_similarity"] = semantic_similarity(answer, reference, embed_model)
    return scores