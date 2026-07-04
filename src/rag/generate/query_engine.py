"""RetrieverQueryEngine: hybrid retriever + cross-encoder rerank, ``compact`` only.

Synthesis is ``compact`` only — no mode toggle (if quality falls short we change the
code, not add switches). The QA prompt template is loaded from
``config/templates/`` (default ``default_qa.txt``); intent-keyed templates can be
added later without touching the engine by passing a different ``template_name``.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from config.settings import Settings, get_settings
from rag.generate.llm import get_llm
from rag.retrieve.rerank import build_reranker
from rag.retrieve.retriever import build_retriever

TEMPLATES_DIR = Path("config/templates")


def load_qa_template(template_name: str = "default_qa") -> PromptTemplate:
    """Load a QA prompt template from ``config/templates/<name>.txt``."""
    path = TEMPLATES_DIR / f"{template_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"QA template not found: {path}")
    return PromptTemplate(path.read_text(encoding="utf-8"))


def build_query_engine(
    settings: Settings | None = None,
    template_name: str = "default_qa",
    rerank_top_n: int = 6,
) -> RetrieverQueryEngine:
    """Return a compact-mode ``RetrieverQueryEngine`` (hybrid retrieve -> rerank -> LLM)."""
    s = settings or get_settings()
    return RetrieverQueryEngine.from_args(
        retriever=build_retriever(s),
        llm=get_llm(s),
        response_mode=ResponseMode.COMPACT,
        text_qa_template=load_qa_template(template_name),
        node_postprocessors=[build_reranker(top_n=rerank_top_n)],
    )