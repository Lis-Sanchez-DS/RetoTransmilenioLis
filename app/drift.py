"""Detección de drift por estación y reentrenamiento automático.

El collector guarda en "Temp" cada observación nueva junto con la
predicción que el modelo vigente le habría dado. La señal de drift es la
accuracy (la misma métrica del proyecto, 1 - WAPE) de esa estación, medida
solo sobre las últimas ACCURACY_WINDOW_HOURS horas de datos (no sobre todo
lo acumulado en "Temp" desde el último reentrenamiento) - así el disparador
reacciona a cómo está funcionando el modelo *ahora mismo* en vez de diluirse
con un histórico largo que puede mezclar tramos buenos y malos. Con el tick
de ~15 minutos del stream, una ventana de 6 horas deja como máximo ~24
puntos por estación, de ahí que MIN_DATAPOINTS esté fijado bien por debajo
de ese techo.

Cuando una estación dispara el drift, el reentrenamiento reemplaza el
histórico 1:1: por cada punto nuevo incorporado desde "Temp", se descarta
el punto más antiguo de "Original Data" - una ventana deslizante de tamaño
fijo, sin evaluar si el cambio de distribución es "real" o no. (Antes se
usaba una prueba de Kolmogorov-Smirnov para decidir si conservar todo el
histórico o recortarlo; se quitó a propósito para simplificar el
comportamiento a un reemplazo 1:1 siempre.)
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

ACCURACY_THRESHOLD = 0.90
ACCURACY_WINDOW_HOURS = 6
# At the stream's ~15-minute tick, 6h caps out around 24 points/station, so
# this has to sit comfortably under that ceiling or the trigger could never
# fire. 20 leaves room for a handful of missed/late ticks.
MIN_DATAPOINTS = 20

MODEL_PARAMS = {"learning_rate": 0.05, "n_estimators": 400, "max_leaves": 40}


def station_accuracy_stats() -> pd.DataFrame:
    """Accuracy and count per station over the last ACCURACY_WINDOW_HOURS of data.

    Accuracy = max(0, 1 - WAPE), i.e. 1 - sum(|demand - prediction|) / sum(|demand|),
    the same formula as train_comparison_models.py's score() (there scaled by 100).

    The window is anchored to the newest observed_at across both tables rather
    than wall-clock now(): the competition runs on a virtual clock decoupled
    from real time, so "now" has to mean "as of the freshest data we have."

    A retrain empties "Temp" for the stations it just touched (their rows move
    into "Original Data" - see _promote_temp_to_original). Right after that, a
    just-retrained station's own "Temp" history can be shorter than the full
    window, even though the data itself still exists - it just moved. So this
    reads both tables for the window and lets whichever one holds each point
    fill it in; it's the same rows, just possibly in their new location.
    """
    with connection() as conn:
        rows = conn.execute(
            """WITH combined AS (
                   SELECT station_id, demand, prediction, observed_at FROM "Temp"
                   UNION ALL
                   SELECT station_id, demand, prediction, observed_at FROM "Original Data"
               ),
               cutoff AS (SELECT max(observed_at) - %s::interval AS ts FROM combined)
               SELECT station_id, demand, prediction FROM combined, cutoff
               WHERE prediction IS NOT NULL AND observed_at > cutoff.ts""",
            (f"{ACCURACY_WINDOW_HOURS} hours",),
        ).fetchall()
    columns = ["station_id", "accuracy", "count"]
    if not rows:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame(rows, columns=["station_id", "demand", "prediction"])
    data["abs_error"] = (data["demand"] - data["prediction"]).abs()
    data["abs_demand"] = data["demand"].abs()
    stats = data.groupby("station_id").agg(
        abs_error=("abs_error", "sum"), abs_demand=("abs_demand", "sum"), count=("demand", "size")
    )
    stats["accuracy"] = (1 - stats["abs_error"] / stats["abs_demand"].clip(lower=1)).clip(lower=0)
    return stats.reset_index()[columns]


def stations_needing_retrain(stats: pd.DataFrame) -> list[str]:
    """Stations with enough recent data whose accuracy has dropped below the bar."""
    if stats.empty:
        return []
    drifted = stats[(stats["count"] > MIN_DATAPOINTS) & (stats["accuracy"] < ACCURACY_THRESHOLD)]
    return sorted(drifted["station_id"].tolist())


def _fetch_recent(station_id: str) -> list[tuple]:
    """New (Temp) observations for this station, oldest first."""
    with connection() as conn:
        return conn.execute(
            'SELECT observed_at, demand FROM "Temp" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()


def _training_frame(station_id: str, recent: list[tuple], drop_oldest: int) -> pd.DataFrame:
    """Historical data (minus the oldest `drop_oldest` rows) plus the new Temp observations."""
    with connection() as conn:
        historical = conn.execute(
            'SELECT observed_at, demand FROM "Original Data" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()
    if drop_oldest > 0:
        historical = historical[drop_oldest:]
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
        grow_policy="lossguide",
        max_depth=0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **MODEL_PARAMS,
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


def _promote_temp_to_original(station_id: str, drop_oldest: int) -> int:
    """Once the new model is live: slide the window, then fold Temp into history.

    The new Temp rows are always merged in; `drop_oldest` (always equal to
    the number of new rows, for the 1:1 sliding window) controls how many of
    the oldest historical rows are discarded first.
    """
    with connection() as conn:
        with conn.transaction():
            if drop_oldest > 0:
                conn.execute(
                    """WITH oldest AS (
                           SELECT station_id, observed_at FROM "Original Data"
                           WHERE station_id = %s
                           ORDER BY observed_at ASC
                           LIMIT %s
                       )
                       DELETE FROM "Original Data" o
                       USING oldest
                       WHERE o.station_id = oldest.station_id AND o.observed_at = oldest.observed_at""",
                    (station_id, drop_oldest),
                )
            # `prediction` is carried over too (not just demand): once these rows
            # move into "Original Data", station_accuracy_stats() needs to be able
            # to keep reading them there for the 6h accuracy window - see its
            # docstring for why.
            conn.execute(
                """INSERT INTO "Original Data" (station_id, observed_at, demand, prediction)
                   SELECT station_id, observed_at, demand, prediction FROM "Temp"
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
        recent = _fetch_recent(station_id)
        if not recent:
            continue
        # 1:1 sliding-window replacement: every new point incorporated from
        # "Temp" bumps out the oldest point in "Original Data", no exceptions.
        drop_oldest = len(recent)

        frame = _training_frame(station_id, recent, drop_oldest)
        if len(frame) <= max(LAGS):
            continue
        path = _train_station_model(station_id, frame)
        _upload_model(station_id, path)
        merged = _promote_temp_to_original(station_id, drop_oldest)
        retrained.append(station_id)
        print(
            f"Drift en {station_id}: reentrenado con {len(frame)} observaciones "
            f"({merged} nuevas incorporadas al histórico, {drop_oldest} antiguas descartadas 1:1), "
            "modelo republicado en Supabase.",
            flush=True,
        )
    return retrained
