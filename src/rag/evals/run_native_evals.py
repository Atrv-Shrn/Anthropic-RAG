"""Native LlamaIndex evaluators: Faithfulness, Relevancy, Correctness.

Runs the three LlamaIndex evaluators (deepseek as judge) over the golden set and records
results alongside the RAGAS baseline for cross-comparison.
"""

from __future__ import annotations


def run_native_evals():
    """Run the LlamaIndex Faithfulness/Relevancy/Correctness evaluators + record."""
    raise NotImplementedError("M7")