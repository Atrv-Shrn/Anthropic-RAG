"""RAGAS eval runner: LLM metrics + non-LLM metrics.

LLM metrics (faithfulness, answer relevancy, context precision/recall) use deepseek as
judge and local nomic embeddings; non-LLM metrics cover semantic similarity and
ROUGE/BLEU-style overlap. Records a baseline against the golden set.
"""

from __future__ import annotations


def run_ragas():
    """Run the RAGAS suite against the golden set and record metrics."""
    raise NotImplementedError("M7")