"""Generation LLM: Ollama Cloud (``deepseek-v4-pro:cloud``) with bearer auth.

LlamaIndex's ``Ollama`` class targets a *local* daemon by default. Ollama Cloud is
the same API surface served from ``https://ollama.com`` — we point ``base_url`` there
and send ``Authorization: Bearer <OLLAMA_CLOUD_API_KEY>`` via the class's ``headers``
field, which the underlying ``ollama.Client`` forwards on every request.

References:
- https://docs.ollama.com/api/authentication (Bearer auth on ollama.com)
- https://docs.ollama.com/cloud

The model tag is env-configurable (``GEN_MODEL``) so the exact cloud tag can be
confirmed/changed without a code edit. Low temperature (0.1) for grounded answers.
"""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from config.settings import Settings, get_settings

DEFAULT_REQUEST_TIMEOUT = 120.0


def get_llm(settings: Settings | None = None) -> Ollama:
    """Return an ``Ollama`` LLM wired to Ollama Cloud with bearer auth."""
    s = settings or get_settings()
    headers: dict[str, str] = {}
    if s.ollama_cloud_api_key:
        headers["Authorization"] = f"Bearer {s.ollama_cloud_api_key}"
    return Ollama(
        model=s.gen_model,
        base_url=s.ollama_cloud_base_url,
        headers=headers or None,
        temperature=0.1,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        # deepseek-v4-pro:cloud may or may not expose tool-calling; we don't use
        # function calling in this pipeline, so leave the class default.
    )