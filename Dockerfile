# App image: pipeline + scheduler + MCP server.
# Heavy (torch + sentence-transformers for the bge reranker); build once, reuse.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Build tools are not required (all wheels), but git is needed by some llama-index
# metadata steps and curl helps diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps first (cache layer). Only the main dependency group is installed;
# the evals/dev extras are intentionally excluded from the runtime image. The
# cache mount persists downloaded wheels across rebuilds, and torch resolves to
# the CPU wheel index on linux (see [tool.uv.sources] in pyproject.toml) — no
# multi-GB nvidia CUDA wheels.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# Pre-download the bge cross-encoder reranker into the image's HF cache so the first
# answer() call after a container start doesn't pay a one-time HuggingFace download
# (observed to stall the first request otherwise). HF_HOME fixes the cache location.
ENV HF_HOME=/app/.hf_cache
RUN uv run --no-project python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# Load the reranker purely from the baked cache at runtime. Without offline mode
# sentence-transformers still makes HF metadata HEAD/GET calls on every load (checking
# for updates), which added ~90s of latency to the first query on a slow connection.
# The weights are already in the image, so offline is both correct and much faster.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Copy the package + config (secrets come from env at runtime, never baked in),
# then install the project itself so `uv run --no-sync` does zero work at startup.
COPY src ./src
COPY config ./config
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Expose the MCP Streamable HTTP port.
EXPOSE 8000

# Default flow: serve immediately, seed in the background (see rag/__main__.py).
# --no-sync: the env is fully built above; don't resolve/install at container start.
CMD ["uv", "run", "--no-sync", "python", "-m", "rag"]