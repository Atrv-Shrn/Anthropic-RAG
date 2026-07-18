"""Container entrypoint: ``python -m rag``.

Subcommands:
  seed   — one-shot incremental ingest pass, then exit.
  serve  — start the hourly scheduler + the MCP server (no seed).
  (default, no arg) — serve immediately, run the seed in a background thread
  (the standard container flow: the MCP server must be reachable in seconds,
  not after the multi-hour first embed pass).

Examples:
  python -m rag            # scheduler + MCP server, seed in background
  python -m rag seed       # seed only, blocking (e.g. manual refresh)
  python -m rag serve      # scheduler + MCP server only (no seed)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("rag")


def _cmd_seed() -> int:
    from rag.orchestrator import seed

    stats = seed()
    for repo, s in stats.items():
        log.info("seed %s: %s", repo, s)
    return 0


def _background_seed() -> None:
    """Seed without blocking the server; refresh the query engine when done."""
    try:
        from rag.orchestrator import seed

        stats = seed()
        for repo, s in stats.items():
            log.info("background seed %s: %s", repo, s)
        # Same pattern as scheduler/hourly._job: drop the cached query engine so
        # the next answer() sees the freshly-seeded corpus.
        from rag.server.mcp_server import invalidate_query_engine

        invalidate_query_engine()
        log.info("background seed complete — query engine refreshed")
    except Exception:
        log.exception("background seed failed (server keeps running; hourly ingest will retry)")


def _cmd_serve() -> int:
    from rag.scheduler.hourly import start_scheduler
    from rag.server.mcp_server import run as run_server

    s = get_settings()
    start_scheduler()
    log.info("MCP server starting on %s:%d (auth=%s)", s.mcp_host, s.mcp_port, bool(s.mcp_auth_token))
    run_server()  # blocks
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Anthropic GitHub RAG pipeline")
    parser.add_argument("command", nargs="?", default="all", choices=["all", "seed", "serve"])
    args = parser.parse_args(argv)

    if args.command == "seed":
        return _cmd_seed()
    if args.command == "all":
        log.info("seeding in background — answers may be partial until the seed completes")
        threading.Thread(target=_background_seed, name="seed", daemon=True).start()
    return _cmd_serve()


if __name__ == "__main__":
    sys.exit(main())