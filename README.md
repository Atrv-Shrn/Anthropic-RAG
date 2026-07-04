# anthropic-rag

Live RAG pipeline over the **Anthropic GitHub organization** — grounded Q&A for AI
agents over docs + issues/PRs/comments (not source code). Stays fresh via hourly
incremental ingestion, served over an MCP server.

## Stack

- **Corpus**: curated Anthropic repos (see `config/repos.yaml`) — docs + issues/PRs/comments only.
- **Embeddings**: local Ollama `nomic-embed-text` (dense, 768-dim, task-prefixed).
- **Generation + RAGAS judge**: `deepseek-v4-pro:cloud` via Ollama Cloud.
- **Retrieval**: hybrid — dense (Qdrant) + BM25 (Redis docstore), RRF fusion, `bge-reranker-base` rerank.
- **Synthesis**: `compact` only.
- **Stores**: Qdrant (vectors) · Redis (docstore + UPSERTS dedup) · SQLite (watermarks).
- **Serving**: FastMCP over Streamable HTTP — two tools: `answer(query)`, `get_documents(doc_ids)`.
- **Live wiring**: APScheduler hourly incremental ingest.

## Quickstart (local Docker)

```bash
cp .env.example .env       # fill in GITHUB_TOKEN + OLLAMA_CLOUD_API_KEY
uv sync
docker compose up -d       # qdrant, redis, ollama(nomic), app
```

## Hosting for many users

Same stack on AWS EC2: open the MCP port in the security group + set `MCP_AUTH_TOKEN`.
No pipeline code change — env + security group + auth only.