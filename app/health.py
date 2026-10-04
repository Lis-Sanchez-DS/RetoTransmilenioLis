"""Job heartbeat helpers used to detect a silently-stopped collector.

Drift detection (see app/drift.py) only ever runs from collector.yml. If that
workflow gets disabled, cancelled, or quietly fails, nothing else notices -
submissions.yml would keep submitting against an increasingly stale model
forever, with no signal outside of manually checking the Actions tab.

Each successful collector run records a row in `ops.job_runs`. submit_xgboost
checks that row's freshness on every cycle and turns a long silence into a
loud, unmissable failure instead: it lets the submission it already built go
out first, then raises so the job step (and eventually the run) shows failed
in Actions.
"""

import json
from datetime import datetime, timezone

from app.db import connection

# 3x the collector's 30-minute cadence, to tolerate one missed/delayed run
# before treating it as suspicious rather than ordinary jitter.
HEARTBEAT_STALE_AFTER_MINUTES = 90


class CollectorStale(RuntimeError):
    """Raised when the collector hasn't reported a successful run recently."""


def record_job_run(job_name: str, status: str, started_at: datetime) -> None:
    with connection() as conn:
        conn.execute(
            """INSERT INTO ops.job_runs (job_name, status, started_at, finished_at)
               VALUES (%s, %s, %s, now())""",
            (job_name, status, started_at),
        )


def check_collector_heartbeat() -> None:
    with connection() as conn:
        row = conn.execute(
            """SELECT finished_at FROM ops.job_runs
               WHERE job_name = 'collector' AND status = 'ok'
               ORDER BY finished_at DESC LIMIT 1"""
        ).fetchone()
    if row is None:
        return  # no heartbeat recorded yet (fresh database) - nothing to compare against
    finished_at = row[0]
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - finished_at).total_seconds() / 60
    if age_minutes > HEARTBEAT_STALE_AFTER_MINUTES:
        raise CollectorStale(
            f"El collector no reporta una corrida exitosa hace {age_minutes:.0f} min "
            f"(límite {HEARTBEAT_STALE_AFTER_MINUTES} min). Puede estar deshabilitado o fallando."
        )


LEDGER_KEY = "last_submission"


def record_submission(cycle_id: str, details: dict, overwrite: bool = True) -> None:
    """Persists which cycle was last delivered (and its receipt, model and commit)
    in `collector_state` so an independent watchdog (.github/workflows/watchdog.yml)
    and the other submitter can tell a cycle is already delivered. One row, upserted.
    With overwrite=False an existing entry for the SAME cycle (the richer receipt
    written by the job that actually posted) is left alone."""
    value = json.dumps({"cycle_id": cycle_id, **details})
    same_cycle = f'{{"cycle_id": {json.dumps(cycle_id)}%'
    with connection() as conn:
        conn.execute(
            """INSERT INTO collector_state (state_key, cursor_value, updated_at)
               VALUES (%s, %s, now())
               ON CONFLICT (state_key) DO UPDATE
               SET cursor_value = EXCLUDED.cursor_value, updated_at = now()
               WHERE %s OR collector_state.cursor_value NOT LIKE %s""",
            (LEDGER_KEY, value, overwrite, same_cycle),
        )


def last_submission_cycle() -> str | None:
    """cycle_id of the last delivered cycle recorded in the ledger, if any."""
    with connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM collector_state WHERE state_key = %s", (LEDGER_KEY,)
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0]).get("cycle_id")
    except (ValueError, AttributeError):
        return None
