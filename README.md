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

## CLI

```bash
python -m rag           # seed once, then scheduler + MCP server (default container flow)
python -m rag seed      # one-shot incremental ingest, then exit
python -m rag serve     # scheduler + MCP server only (corpus already seeded)
```

## Scripts

Operational helpers in `scripts/` (run with `uv run --no-sync python scripts/<name>.py`):

- `seed_docs.py <repo>...` — docs-only ingest for the named repos, one at a time.
- `prune_repos.py` — prune Qdrant + Redis + watermarks down to the repos in `config/repos.yaml`.
- `trim_golden_set.py` — trim `config/golden_set.json` to the configured repos (+ negatives).
- `collect_predictions.py` — run the query engine over the golden set (checkpointed).
- `run_judges.py` — run RAGAS + native evaluators over collected predictions.

## Evals

Golden set: `config/golden_set.json` (hand-curated Q/A incl. out-of-scope negatives).

```bash
uv sync --extra evals
uv run python scripts/collect_predictions.py   # -> data/evals/predictions.json
uv run python scripts/run_judges.py            # -> data/evals/{ragas,native}_baseline.json
```

Metrics: RAGAS (faithfulness, answer relevancy, context precision/recall) with the
deepseek judge + local nomic embeddings, plus dependency-free ROUGE-L / BLEU / cosine;
and native LlamaIndex Faithfulness/Relevancy/Correctness. In-scope, unseeded, and
negative (refusal-rate) items are reported separately.

## Tests

```bash
uv run python -m pytest        # 41 unit tests (pure logic; external services mocked)
```