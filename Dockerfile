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
# the evals/dev extras are intentionally excluded from the runtime image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the package + config (secrets come from env at runtime, never baked in).
COPY src ./src
COPY config ./config
COPY README.md ./

# Expose the MCP Streamable HTTP port.
EXPOSE 8000

# Default flow: seed once, then scheduler + MCP server (see rag/__main__.py).
CMD ["uv", "run", "python", "-m", "rag"]