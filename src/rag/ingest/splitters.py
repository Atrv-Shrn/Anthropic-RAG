"""Node splitters.

Primary path is ``MarkdownNodeParser`` / ``SentenceSplitter`` since the corpus is docs +
issues/PRs/comments. ``CodeSplitter`` is kept only as a rare fallback for stray fenced
code blocks inside docs (with a ``SentenceSplitter`` fallback); it is not a core concern
because we do not ingest whole codebases.
"""

from __future__ import annotations


def make_splitters() -> dict:
    """Return the routed splitter set keyed by node kind."""
    raise NotImplementedError("M4")