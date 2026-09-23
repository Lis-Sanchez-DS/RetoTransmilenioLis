"""Genera y envía la primera submission usando los XGBoost guardados."""

from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys

import joblib
import requests

from app.collector import collect_new_data
from app.db import connection
from app.features import temporal_features
from app.net import with_retries, UpstreamUnavailable


BASE_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.getenv("PULSO_API_KEY") or os.getenv("API-KEY-PulsoTransmi")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
LAGS = (1, 2, 4, 96, 672)


def api_get(path: str) -> dict:
    def _get():
        response = requests.get(
            f"{BASE_URL}{path}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    return with_retries(_get)


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return value if len(value) == 40 else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_utc(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def predict_cycle_targets(targets: list[dict], history: dict, models: dict) -> list[dict]:
    """Predict every target, recursing forward through each station's horizons.

    Each station gets 4 targets (+15/+30/+45/+60 min from data_cutoff). The
    model was trained so lag_k means "demand k*15 minutes before the exact
    timestamp being predicted" - true only for the +15min target if lags are
    read straight off data_cutoff, since the shorter lags for +30/+45/+60min
    fall on timestamps that haven't been observed yet. So each station's
    targets are predicted in chronological order, feeding each prediction
    back into that station's local history as the stand-in value for the
    not-yet-observed timestamp its own lags need next.
    """
    by_station: dict[str, list[dict]] = {}
    for target in targets:
        by_station.setdefault(target["station_id"], []).append(target)

    predictions = []
    for station_id, station_targets in by_station.items():
        local_history = dict(history)
        model = models[station_id]
        for target in sorted(station_targets, key=lambda t: t["target_at"]):
            timestamp = _parse_utc(target["target_at"])
            features = []
            for lag in LAGS:
                key = (station_id, timestamp - timedelta(minutes=15 * lag))
                if key not in local_history:
                    raise RuntimeError(f"No hay suficiente historia para {station_id} en lag {lag}")
                features.append(local_history[key])
            features.extend(temporal_features(timestamp))
            value = max(0.0, float(model.predict([features])[0]))
            local_history[(station_id, timestamp)] = value
            predictions.append(
                {
                    "station_id": station_id,
                    "target_at": target["target_at"],
                    "value": round(value, 3),
                }
            )
    return predictions


def submit_current_cycle() -> dict | None:
    if not API_KEY:
        raise RuntimeError("Define PULSO_API_KEY antes de ejecutar el script.")
    # Pull the freshest data first, exactly like the reference student loop
    # (collect, then check the cycle). Without this, a station's history can
    # lag the cycle's data_cutoff by up to a collector cycle whenever
    # collection and submission run as separate, independently-scheduled jobs.
    collect_new_data()
    identity = api_get("/v1/me")
    try:
        cycle = api_get("/v1/forecast-cycles/current")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    with connection() as conn:
        # "Temp" holds every observation collected since the last drift
        # retrain for a station (see drift.py); only drifted stations ever
        # get folded into "Original Data". Both tables must be read here or
        # any non-drifted station's history goes stale the moment new data
        # stops landing in "Original Data".
        original = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" WHERE observed_at <= %s',
            (cycle["data_cutoff"],),
        ).fetchall()
        recent = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Temp" WHERE observed_at <= %s',
            (cycle["data_cutoff"],),
        ).fetchall()
    history = {
        (station_id, _parse_utc(observed_at)): float(demand)
        for station_id, observed_at, demand in original + recent
    }

    station_ids = sorted({target["station_id"] for target in cycle["targets"]})
    models = {
        station_id: joblib.load(os.path.join(MODEL_DIR, f"xgboost_{station_id}.joblib"))
        for station_id in station_ids
    }
    predictions = predict_cycle_targets(cycle["targets"], history, models)

    run_id = f"xgboost-{cycle['cycle_id']}"
    model_info = {
        "version": "xgboost-per-station:1.0",
        "trained_at": datetime.fromtimestamp(
            os.path.getmtime(os.path.join(MODEL_DIR, f"xgboost_{predictions[0]['station_id']}.joblib")),
            timezone.utc,
        ).isoformat(),
        "training_data_end": cycle["data_cutoff"],
    }
    commit = git_commit()
    if commit:
        model_info["git_commit"] = commit
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": run_id,
        "data_cutoff": cycle["data_cutoff"],
        "model": model_info,
        "predictions": predictions,
    }

    def _post():
        response = requests.post(
            f"{BASE_URL}/v1/submissions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Idempotency-Key": run_id,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    receipt = with_retries(_post)
    result = {
        "student": identity["display_name"],
        "submission_id": receipt["submission_id"],
        "status": receipt["status"],
        "predictions_received": receipt["predictions_received"],
        "expected_predictions": receipt["expected_predictions"],
    }
    print(f"Estudiante: {result['student']}")
    print(f"Entrega: {result['submission_id']}")
    print(f"Estado: {result['status']}")
    print(f"Predicciones: {result['predictions_received']}/{result['expected_predictions']}")
    return result


def main() -> None:
    try:
        submit_current_cycle()
    except UpstreamUnavailable as exc:
        print(f"AVISO: {exc}", flush=True)
        sys.exit(75)


if __name__ == "__main__":
    main()
