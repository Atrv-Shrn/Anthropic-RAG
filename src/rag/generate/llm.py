"""Ollama Cloud LLM (deepseek-v4-pro:cloud) for generation + RAGAS judge.

The LlamaIndex ``Ollama`` class targets a local daemon by default; cloud needs a custom
``base_url`` and a ``Bearer $OLLAMA_CLOUD_API_KEY`` header on the underlying client.
Low temperature (0.1). The exact model tag is env-configurable so we can confirm the
Ollama Cloud tag at build without code changes.
"""

from __future__ import annotations


def get_llm():
    """Return an ``Ollama`` LLM wired to Ollama Cloud with bearer auth."""
    raise NotImplementedError("M2")