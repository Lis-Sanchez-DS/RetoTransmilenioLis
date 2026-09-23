"""Detección de drift por estación y reentrenamiento automático.

El collector guarda en "Temp" cada observación nueva junto con la
predicción que el modelo vigente le habría dado. Ese historial reciente es
la señal de drift: si el modelo lleva muchas observaciones acertando poco
y de forma consistente (no solo un pico ruidoso), se reentrena con el
histórico ampliado y se republica el modelo en Supabase Storage.
"""

import os

import joblib
import pandas as pd
import requests
from xgboost import XGBRegressor

from app.db import connection
from app.features import temporal_features

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
MODEL_BUCKET = "models"
LAGS = (1, 2, 4, 96, 672)

ACCURACY_THRESHOLD = 0.85
STD_THRESHOLD = 0.025
MIN_DATAPOINTS = 50


def _row_accuracy(demand: float, prediction: float) -> float:
    """Per-observation accuracy, protecting the denominator like the project WAPE."""
    denom = max(abs(float(demand)), 1.0)
    return max(0.0, 1.0 - abs(float(demand) - float(prediction)) / denom)


def station_accuracy_stats() -> pd.DataFrame:
    """Mean, std and count of per-observation accuracy in "Temp", per station."""
    with connection() as conn:
        rows = conn.execute(
            'SELECT station_id, demand, prediction FROM "Temp" WHERE prediction IS NOT NULL'
        ).fetchall()
    columns = ["station_id", "mean_accuracy", "std_accuracy", "count"]
    if not rows:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame(rows, columns=["station_id", "demand", "prediction"])
    data["accuracy"] = [
        _row_accuracy(demand, prediction)
        for demand, prediction in zip(data["demand"], data["prediction"])
    ]
    stats = data.groupby("station_id")["accuracy"].agg(
        mean_accuracy="mean", std_accuracy="std", count="count"
    )
    return stats.reset_index()[columns]


def stations_needing_retrain(stats: pd.DataFrame) -> list[str]:
    """Stations with a sustained, low-variance accuracy drop: real drift, not noise."""
    if stats.empty:
        return []
    drifted = stats[
        (stats["count"] > MIN_DATAPOINTS)
        & (stats["mean_accuracy"] < ACCURACY_THRESHOLD)
        & (stats["std_accuracy"] < STD_THRESHOLD)
    ]
    return sorted(drifted["station_id"].tolist())


def _training_frame(station_id: str) -> pd.DataFrame:
    """Historical data plus the not-yet-merged Temp observations for this station."""
    with connection() as conn:
        historical = conn.execute(
            'SELECT observed_at, demand FROM "Original Data" WHERE station_id = %s',
            (station_id,),
        ).fetchall()
        recent = conn.execute(
            'SELECT observed_at, demand FROM "Temp" WHERE station_id = %s',
            (station_id,),
        ).fetchall()
    frame = pd.DataFrame(historical + recent, columns=["observed_at", "demand"])
    return frame.drop_duplicates(subset="observed_at").sort_values("observed_at").reset_index(drop=True)


def _train_station_model(station_id: str, frame: pd.DataFrame) -> str:
    y = frame.demand.astype(float)
    lagged = pd.concat({f"lag_{lag}": y.shift(lag) for lag in LAGS}, axis=1)
    calendar = pd.DataFrame(
        [temporal_features(value) for value in frame["observed_at"]],
        columns=["hour_sin", "hour_cos", "week_sin", "week_cos"],
    )
    features = pd.concat([lagged, calendar], axis=1)
    valid = features.notna().all(axis=1)
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_leaves=40,
        grow_policy="lossguide",
        max_depth=0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features.loc[valid], y.loc[valid])
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"xgboost_{station_id}.joblib")
    joblib.dump(model, path)
    return path


def _upload_model(station_id: str, path: str) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY para subir el modelo.")
    with open(path, "rb") as fh:
        payload = fh.read()
    response = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{MODEL_BUCKET}/xgboost/xgboost_{station_id}.joblib",
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "x-upsert": "true",
            "Content-Type": "application/octet-stream",
        },
        data=payload,
        timeout=60,
    )
    response.raise_for_status()


def _promote_temp_to_original(station_id: str) -> int:
    """Once the new model is live, fold the Temp rows into permanent history."""
    with connection() as conn:
        with conn.transaction():
            conn.execute(
                """INSERT INTO "Original Data" (station_id, observed_at, demand)
                   SELECT station_id, observed_at, demand FROM "Temp"
                   WHERE station_id = %s
                   ON CONFLICT (station_id, observed_at) DO NOTHING""",
                (station_id,),
            )
            deleted = conn.execute('DELETE FROM "Temp" WHERE station_id = %s', (station_id,)).rowcount
    return deleted


def check_and_retrain() -> list[str]:
    """Detect drifted stations and retrain + republish their models. Returns the list retrained."""
    stats = station_accuracy_stats()
    drifted = stations_needing_retrain(stats)
    retrained = []
    for station_id in drifted:
        frame = _training_frame(station_id)
        if len(frame) <= max(LAGS):
            continue
        path = _train_station_model(station_id, frame)
        _upload_model(station_id, path)
        merged = _promote_temp_to_original(station_id)
        retrained.append(station_id)
        print(
            f"Drift en {station_id}: reentrenado con {len(frame)} observaciones "
            f"({merged} nuevas incorporadas al histórico), modelo republicado en Supabase.",
            flush=True,
        )
    return retrained
