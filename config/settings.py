"""Env-driven configuration for the Anthropic RAG pipeline.

All runtime knobs are read from environment variables (optionally a local `.env`).
`pydantic-settings` handles parsing; `get_settings()` is a cached singleton so we
only parse once per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for every env-driven value in the stack."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- GitHub corpus ----
    github_token: str = Field(default="", description="GitHub PAT for repo read access + rate limit.")
    github_org: str = Field(default="anthropics", description="GitHub organization to ingest.")

    # ---- Local Ollama embeddings (dense, nomic) ----
    ollama_embed_base_url: str = Field(default="http://localhost:11434")
    embed_model: str = Field(default="nomic-embed-text")

    # ---- Ollama Cloud: generation + RAGAS judge ----
    ollama_cloud_base_url: str = Field(default="https://ollama.com")
    ollama_cloud_api_key: str = Field(default="", description="Bearer token for Ollama Cloud.")
    gen_model: str = Field(default="deepseek-v4-pro:cloud")

    # ---- Stores ----
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="anthropic_rag")
    redis_url: str = Field(default="redis://localhost:6379")

    # ---- MCP server ----
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=8000)
    mcp_auth_token: str = Field(
        default="",
        description="Optional bearer token; required when exposing beyond localhost (e.g. EC2).",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()