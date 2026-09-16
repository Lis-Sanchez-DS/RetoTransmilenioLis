import json
import os
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.db import connection

API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
POLL_SECONDS = int(os.getenv("COLLECTOR_POLL_SECONDS", "120"))


def get_cursor() -> str | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM collector_state WHERE state_key = 'observations'"
        ).fetchone()
    return row[0] if row else None


def collect_last_15(fetcher=None) -> dict:
    """Collect exactly the next 15 stream records and persist the cursor."""
    cursor = get_cursor()
    query = {"limit": 15}
    if cursor:
        query["cursor"] = cursor
    if fetcher is None:
        request = Request(f"{API_URL}/v1/stream/observations?{urlencode(query)}")
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    else:
        payload = fetcher(cursor=cursor, limit=15)

    records = payload.get("observations", payload.get("data", []))
    if len(records) > 15:
        records = records[:15]
    next_cursor = payload.get("next_cursor")
    if next_cursor is None:
        raise RuntimeError("The stream response did not include next_cursor")

    with connection() as conn:
        with conn.transaction():
            for item in records:
                conn.execute(
                    """INSERT INTO "Temp" (station_id, observed_at, demand)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (station_id, observed_at) DO NOTHING""",
                    (item["station_id"], item["observed_at"], item["demand"]),
                )
            conn.execute(
                """INSERT INTO collector_state (state_key, cursor_value, updated_at)
                   VALUES ('observations', %s, now())
                   ON CONFLICT (state_key) DO UPDATE
                   SET cursor_value = EXCLUDED.cursor_value, updated_at = now()""",
                (next_cursor,),
            )
    return {"collected": len(records), "cursor": next_cursor}


def run() -> None:
    result = collect_last_15()
    print(f"Collected {result['collected']} observations; cursor={result['cursor']}", flush=True)


if __name__ == "__main__":
    run()
