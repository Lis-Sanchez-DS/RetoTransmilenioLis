"""Genera y envía la primera submission usando los XGBoost guardados."""

from datetime import datetime, timezone
import math
import os
import subprocess

import joblib
import pandas as pd
import requests

from app.db import connection
from app.features import temporal_features


BASE_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.getenv("PULSO_API_KEY") or os.getenv("API-KEY-PulsoTransmi")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
LAGS = (1, 2, 4, 96, 672)


def api_get(path: str) -> dict:
    response = requests.get(
        f"{BASE_URL}{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return value if len(value) == 40 else None
    except (OSError, subprocess.CalledProcessError):
        return None


def features_for_target(history: pd.DataFrame, target: dict) -> list[float]:
    station = target["station_id"]
    timestamp = pd.Timestamp(target["target_at"])
    values = (
        history.loc[history["station_id"] == station]
        .sort_values("observed_at")["demand"]
        .to_numpy()
    )
    if len(values) < max(LAGS):
        raise RuntimeError(f"No hay suficiente historia para {station}")
    features = [float(values[-lag]) for lag in LAGS]
    features.extend(temporal_features(timestamp.to_pydatetime()))
    return features


def submit_current_cycle() -> dict | None:
    if not API_KEY:
        raise RuntimeError("Define PULSO_API_KEY antes de ejecutar el script.")
    identity = api_get("/v1/me")
    try:
        cycle = api_get("/v1/forecast-cycles/current")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    with connection() as conn:
        rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" '
            'WHERE observed_at <= %s ORDER BY station_id, observed_at',
            (cycle["data_cutoff"],),
        ).fetchall()
    history = pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"])
    history["observed_at"] = pd.to_datetime(history["observed_at"], utc=True)

    predictions = []
    for target in cycle["targets"]:
        model = joblib.load(
            os.path.join(MODEL_DIR, f"xgboost_{target['station_id']}.joblib")
        )
        value = max(0.0, float(model.predict([features_for_target(history, target)])[0]))
        predictions.append(
            {
                "station_id": target["station_id"],
                "target_at": target["target_at"],
                "value": round(value, 3),
            }
        )

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
    receipt = response.json()
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
    submit_current_cycle()


if __name__ == "__main__":
    main()
