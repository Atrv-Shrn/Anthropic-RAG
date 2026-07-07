"""Anthropic GitHub RAG pipeline.

Live, grounded Q&A over the Anthropic GitHub organization (docs + issues/PRs/comments).
Ingestion, retrieval, synthesis and evals are built on LlamaIndex; the corpus scraping,
live wiring (watermarks + scheduler), Docker packaging and MCP surface live here.
"""

__version__ = "0.1.0"