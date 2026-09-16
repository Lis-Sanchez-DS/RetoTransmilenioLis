from fastapi import FastAPI

from app.db import connection

app = FastAPI(title="Pulso TransMi API", version="0.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pulso-transmi-api"}


@app.get("/ready")
def ready() -> dict[str, str]:
    with connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "service": "pulso-transmi-api"}


@app.get("/v1/meta")
def meta() -> dict[str, str]:
    return {"version": "0.3.0", "service": "pulso-transmi-api"}


@app.get("/v1/stations")
def stations() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT station_id, name, latitude, longitude FROM stations ORDER BY name"
        ).fetchall()
    return [dict(zip(("station_id", "name", "latitude", "longitude"), row)) for row in rows]


@app.get("/v1/observations")
def observations(limit: int = 100, offset: int = 0) -> list[dict]:
    limit = min(max(limit, 1), 1000)
    offset = max(offset, 0)
    with connection() as conn:
        rows = conn.execute(
            """SELECT station_id, observed_at, demand
               FROM "Original Data" ORDER BY observed_at, station_id
               LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
    return [dict(zip(("station_id", "observed_at", "demand"), row)) for row in rows]
