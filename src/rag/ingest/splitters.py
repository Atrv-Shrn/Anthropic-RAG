"""Node splitters for the corpus (docs + issues/PRs/comments).

Primary path: ``MarkdownNodeParser`` (structural split on headers/lists) followed by
``SentenceSplitter`` (size enforcement). This works for both markdown docs and plain
issue/comment text — MarkdownNodeParser is a no-op on non-markdown, and
SentenceSplitter chunks everything to a sane size.

``CodeSplitter`` is intentionally NOT in the default path: we do not ingest source
codebases, so fenced code blocks inside docs are simply treated as text by the
SentenceSplitter. If we ever need language-aware chunking for stray code in docs,
we can add it here — but the spec keeps the pipeline simple.
"""

from __future__ import annotations

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def make_splitters(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list:
    """Return the ordered transformation list: markdown split -> sentence split."""
    return [
        MarkdownNodeParser(),
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    ]