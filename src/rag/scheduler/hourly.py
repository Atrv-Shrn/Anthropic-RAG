"""APScheduler hourly incremental-ingest job (runs in-container alongside the MCP server).

Uses a non-blocking ``BackgroundScheduler`` so the MCP server can occupy the
foreground (``mcp.run`` blocks). The job fires every ``interval_minutes`` (default
60) and calls ``run_incremental_ingest``. A manual trigger is exposed for the seed
entrypoint and for tests.
"""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from rag.orchestrator import run_incremental_ingest

log = logging.getLogger(__name__)

JOB_ID = "incremental-ingest"
_scheduler: BackgroundScheduler | None = None


def _job() -> None:
    try:
        stats = run_incremental_ingest()
        log.info("hourly ingest complete: %s", stats)
        # Drop the server's cached query engine so the next answer() rebuilds BM25
        # over the freshly-ingested docs (the lexical arm is a build-time snapshot).
        try:
            from rag.server.mcp_server import invalidate_query_engine

            invalidate_query_engine()
        except Exception:  # noqa: BLE001 - server module may not be loaded (seed-only runs)
            pass
    except Exception:
        log.exception("hourly ingest job failed")


def trigger_now() -> dict[str, Any]:
    """Run one incremental ingest pass immediately (synchronous)."""
    return run_incremental_ingest()


def start_scheduler(interval_minutes: int = 60) -> BackgroundScheduler:
    """Start the hourly incremental-ingest scheduler (non-blocking)."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        _job,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    sched.start()
    _scheduler = sched
    log.info("scheduler started: incremental ingest every %d min", interval_minutes)
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None