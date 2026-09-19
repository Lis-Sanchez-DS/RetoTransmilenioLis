import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np

from app.db import connection
from app.features import temporal_features

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
        rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" '
            'WHERE station_id = ANY(%s)',
            (station_ids,),
        ).fetchall()
    history = {(station, observed_at): float(demand) for station, observed_at, demand in rows}
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


def collect_new_data(fetcher=None) -> dict:
    """Collect all pages released after the saved cursor into Temp."""
    cursor = get_cursor()
    total = 0
    pages = 0
    while True:
        query = {"limit": PAGE_SIZE}
        if cursor:
            query["cursor"] = cursor
        if fetcher is None:
            request = Request(f"{API_URL}/v1/stream/observations?{urlencode(query)}")
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        else:
            payload = fetcher(cursor=cursor, limit=PAGE_SIZE)

        records = payload.get("data", payload.get("observations", []))
        next_cursor = payload.get("next_cursor")
        if not records and next_cursor is None:
            pages += 1
            break
        if next_cursor is None:
            raise RuntimeError("The stream response did not include next_cursor")
        previous_cursor = cursor
        if records and next_cursor == previous_cursor:
            raise RuntimeError("The stream response returned an unchanged cursor")
        predicted_records = predict_records(records)

        with connection() as conn:
            with conn.transaction():
                for item, prediction in predicted_records:
                    conn.execute(
                        """INSERT INTO "Temp" (station_id, observed_at, demand, prediction)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (station_id, observed_at) DO NOTHING""",
                        (item["station_id"], item["observed_at"], item["demand"], prediction),
                    )
                conn.execute(
                    """INSERT INTO collector_state (state_key, cursor_value, updated_at)
                       VALUES ('observations', %s, now())
                       ON CONFLICT (state_key) DO UPDATE
                       SET cursor_value = EXCLUDED.cursor_value, updated_at = now()""",
                    (next_cursor,),
                )
        total += len(records)
        pages += 1
        cursor = next_cursor
        if not records or next_cursor == previous_cursor:
            break
    return {"collected": total, "pages": pages, "cursor": cursor}


collect_last_15 = collect_new_data


def run() -> None:
    result = collect_new_data()
    print(f"Collected {result['collected']} observations in {result['pages']} pages; cursor={result['cursor']}", flush=True)


if __name__ == "__main__":
    run()
