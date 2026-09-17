import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.db import connection

API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")


def load_all() -> int:
    station_request = Request(f"{API_URL}/v1/stations")
    with urlopen(station_request, timeout=30) as response:
        station_payload = json.load(response)
    stations = station_payload.get("data", station_payload.get("stations", station_payload))
    with connection() as conn:
        with conn.transaction():
            for station in stations:
                conn.execute(
                    """INSERT INTO stations (station_id, name, latitude, longitude)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (station_id) DO UPDATE SET name = EXCLUDED.name,
                       latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude""",
                    (station["station_id"], station.get("name", station.get("station_name")), station.get("latitude"), station.get("longitude")),
                )

    cursor = None
    total = 0
    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        request = Request(f"{API_URL}/v1/observations?{urlencode(params)}")
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        records = payload.get("data", payload.get("observations", []))
        if not records:
            break
        with connection() as conn:
            with conn.transaction():
                for item in records:
                    conn.execute(
                        'INSERT INTO "Original Data" (station_id, observed_at, demand) '
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (item["station_id"], item["observed_at"], item["demand"]),
                    )
        total += len(records)
        next_cursor = payload.get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    print(f"Loaded {total} observations into Original Data", flush=True)
    return total


if __name__ == "__main__":
    load_all()
