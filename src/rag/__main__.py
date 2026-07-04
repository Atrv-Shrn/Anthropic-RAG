"""Container entrypoint: ``python -m rag``.

Subcommands:
  seed   — one-shot incremental ingest pass, then exit.
  serve  — start the hourly scheduler + the MCP server (no seed).
  (default, no arg) — seed once, then serve (the standard container flow: a
  one-shot seed before the scheduler starts).

Examples:
  python -m rag            # seed, then scheduler + MCP server
  python -m rag seed       # seed only (e.g. first boot / manual refresh)
  python -m rag serve      # scheduler + MCP server only (already seeded)
"""

from __future__ import annotations

import argparse
import logging
import sys

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

    if args.command in ("all", "seed"):
        code = _cmd_seed()
        if code or args.command == "seed":
            return code
    return _cmd_serve()


if __name__ == "__main__":
    sys.exit(main())