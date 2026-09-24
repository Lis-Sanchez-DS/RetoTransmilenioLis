import base64
import json
import os
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np

from app.db import connection
from app.drift import check_and_retrain
from app.features import temporal_features
from app.health import record_job_run
from app.net import with_retries, UpstreamUnavailable

API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PAGE_SIZE = min(max(int(os.getenv("COLLECTOR_PAGE_SIZE", "5000")), 1), 5000)
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
LAGS = (1, 2, 4, 96, 672)


def get_cursor() -> str | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM collector_state WHERE state_key = 'observations'"
        ).fetchone()
    return row[0] if row else None


def predict_records(records: list[dict]) -> list[tuple[dict, float]]:
    """Predict each new observation using its station's saved XGBoost model."""
    if not records:
        return []
    station_ids = sorted({item["station_id"] for item in records})
    with connection() as conn:
        # "Temp" holds every observation collected since a station's last
        # drift retrain (see drift.py); only drifted stations ever get
        # folded into "Original Data". Both tables must be read here or a
        # non-drifted station's lag lookups fall back on stale seed data
        # once a run's own batch is too small to bridge the gap itself.
        original_rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" '
            'WHERE station_id = ANY(%s)',
            (station_ids,),
        ).fetchall()
        temp_rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Temp" '
            'WHERE station_id = ANY(%s)',
            (station_ids,),
        ).fetchall()
    history = {
        (station, observed_at): float(demand)
        for station, observed_at, demand in original_rows + temp_rows
    }
    models = {
        station_id: joblib.load(os.path.join(MODEL_DIR, f"xgboost_{station_id}.joblib"))
        for station_id in station_ids
    }
    predictions = []
    for item in records:
        observed_at = item["observed_at"]
        ts = observed_at if isinstance(observed_at, datetime) else datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        features = []
        for lag in LAGS:
            key = (item["station_id"], ts - timedelta(minutes=15 * lag))
            if key not in history:
                raise RuntimeError(f"Missing lag {lag} for station {item['station_id']} at {ts.isoformat()}")
            features.append(history[key])
        features.extend(temporal_features(ts))
        prediction = max(0.0, float(models[item["station_id"]].predict(np.array([features]))[0]))
        predictions.append((item, round(prediction, 4)))
        history[(item["station_id"], ts)] = float(item["demand"])
    return predictions


def _synthetic_cursor(record: dict) -> str:
    """Build a resume cursor for when the server hands back next_cursor: null.

    That happens on every run once the collector is caught up to the live
    edge - the normal steady state, not a rare corner case - and the API
    issues no bookmark for it. Verified against the live API: encoding the
    last record's own (released_at, observed_at, station_id) the same way
    reproduces the server's own next_cursor byte-for-byte, and round-trips
    correctly when sent back as ?cursor=. Without this, every single run
    forever would silently re-fetch and re-process the entire stream from
    scratch instead of just what's new - which is what was happening before
    this fix (collector_state never accumulated a single saved row).

    This relies on the server's cursor encoding being what it is today, not
    on anything the API contract (`cursor` is documented as an opaque
    `string | null`) actually promises - see _fetch_page's fallback for what
    happens if a saved cursor (real or synthetic) ever stops being accepted.
    """
    payload = [
        record["released_at"].replace("Z", "+00:00"),
        record["observed_at"].replace("Z", "+00:00"),
        record["station_id"],
    ]
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _fetch_page(fetcher, cursor: str | None, retry_without_cursor: bool = True) -> dict:
    """Fetch one page. If a saved cursor - real or synthetic - gets rejected by
    the server, fall back to a full refetch instead of failing the run: no
    worse than the un-fixed behavior, and self-heals once it succeeds and a
    fresh valid cursor gets saved again.
    """
    query = {"limit": PAGE_SIZE}
    if cursor:
        query["cursor"] = cursor
    if fetcher is not None:
        return fetcher(cursor=cursor, limit=PAGE_SIZE)

    def _get():
        request = Request(f"{API_URL}/v1/stream/observations?{urlencode(query)}")
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    try:
        return with_retries(_get)
    except urllib.error.HTTPError as exc:
        if cursor and retry_without_cursor and 400 <= exc.code < 500:
            print(f"AVISO: el cursor guardado fue rechazado ({exc.code}); se reintenta sin cursor.", flush=True)
            return _fetch_page(fetcher, None, retry_without_cursor=False)
        raise


def collect_new_data(fetcher=None) -> dict:
    """Collect all pages released after the saved cursor into Temp.

    Per the API contract, `next_cursor: null` means the stream has been
    drained up to the current live edge; it is not an error, and it's the
    normal outcome of every run once caught up - see _synthetic_cursor for
    how a resume point still gets saved in that case.
    """
    cursor = get_cursor()
    saved_cursor = cursor
    total = 0
    pages = 0
    while True:
        payload = _fetch_page(fetcher, cursor)

        records = payload.get("data", payload.get("observations", []))
        next_cursor = payload.get("next_cursor")
        pages += 1
        if not records:
            break

        predicted_records = predict_records(records)
        resume_cursor = next_cursor if next_cursor is not None else _synthetic_cursor(records[-1])

        with connection() as conn:
            with conn.transaction():
                for item, prediction in predicted_records:
                    conn.execute(
                        """INSERT INTO "Temp" (station_id, observed_at, demand, prediction)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (station_id, observed_at) DO NOTHING""",
                        (item["station_id"], item["observed_at"], item["demand"], prediction),
                    )
                if resume_cursor != saved_cursor:
                    conn.execute(
                        """INSERT INTO collector_state (state_key, cursor_value, updated_at)
                           VALUES ('observations', %s, now())
                           ON CONFLICT (state_key) DO UPDATE
                           SET cursor_value = EXCLUDED.cursor_value, updated_at = now()""",
                        (resume_cursor,),
                    )
                    saved_cursor = resume_cursor
        total += len(records)
        if next_cursor is None or next_cursor == cursor:
            break
        cursor = next_cursor
    return {"collected": total, "pages": pages, "cursor": saved_cursor}


collect_last_15 = collect_new_data


def run() -> None:
    started_at = datetime.now(timezone.utc)
    try:
        result = collect_new_data()
    except UpstreamUnavailable as exc:
        print(f"AVISO: {exc}", flush=True)
        sys.exit(75)
    print(f"Collected {result['collected']} observations in {result['pages']} pages; cursor={result['cursor']}", flush=True)
    retrained = check_and_retrain()
    if retrained:
        print(f"Modelos reentrenados por drift: {', '.join(retrained)}", flush=True)
    else:
        print("Sin drift detectado.", flush=True)
    # Heartbeat: submit_xgboost.py checks this to notice a collector that
    # went silent (disabled, cancelled, stuck failing) instead of quietly
    # submitting against an aging model forever.
    record_job_run("collector", "ok", started_at)


if __name__ == "__main__":
    run()
