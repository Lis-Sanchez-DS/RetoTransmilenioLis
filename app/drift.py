"""Detección de drift por estación y reentrenamiento automático.

El collector guarda en "Temp" cada observación nueva junto con la
predicción que el modelo vigente le habría dado. Ese historial reciente es
la señal de drift: si con más de 50 observaciones la accuracy de la
estación (la misma métrica del proyecto, 1 - WAPE) cae por debajo de 0.85,
se evalúa si el patrón de demanda cambió antes de reentrenar.

Esa evaluación compara, con una prueba de Kolmogorov-Smirnov de dos
muestras (alpha=0.10), los 50 puntos históricos más recientes contra los
primeros 50 puntos nuevos (cronológicamente adyacentes, para no confundir
el resultado con la estacionalidad diaria/semanal del proyecto). Si no se
rechaza la hipótesis de que vienen de la misma distribución, la caída de
accuracy se trata como sobreajuste: se conserva todo el histórico. Si se
rechaza, se asume un cambio real de régimen: se descartan las filas más
antiguas de "Original Data" (tantas como puntos nuevos) antes de
reentrenar, para que el modelo no se diluya con datos ya obsoletos. En
ambos casos, una vez publicado el nuevo modelo, los puntos nuevos se
incorporan a "Original Data".
"""

import os

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
from xgboost import XGBRegressor

from app.db import connection
from app.features import temporal_features

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
MODEL_BUCKET = "models"
LAGS = (1, 2, 4, 96, 672)

ACCURACY_THRESHOLD = 0.85
MIN_DATAPOINTS = 50
REGIME_SAMPLE_SIZE = 50
REGIME_TEST_ALPHA = 0.10


def station_accuracy_stats() -> pd.DataFrame:
    """Accuracy and count per station over "Temp", using the project's own metric:

    Accuracy = max(0, 1 - WAPE), i.e. 1 - sum(|demand - prediction|) / sum(|demand|),
    the same formula as train_comparison_models.py's score() (there scaled by 100).
    """
    with connection() as conn:
        rows = conn.execute(
            'SELECT station_id, demand, prediction FROM "Temp" WHERE prediction IS NOT NULL'
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


def _new_data_matches_history(station_id: str, recent: list[tuple]) -> bool:
    """Two-sample KS test: True if we fail to reject 'same distribution' at alpha=0.10.

    Compares the most recent REGIME_SAMPLE_SIZE historical points (adjacent
    in time to the new data, to control for daily/weekly seasonality) against
    the first REGIME_SAMPLE_SIZE new points. Too little data to compare
    reliably defaults to True: keep all history rather than discard it on an
    inconclusive read.
    """
    with connection() as conn:
        reference_rows = conn.execute(
            'SELECT demand FROM "Original Data" WHERE station_id = %s '
            'ORDER BY observed_at DESC LIMIT %s',
            (station_id, REGIME_SAMPLE_SIZE),
        ).fetchall()
    reference = np.array([row[0] for row in reference_rows], dtype=float)
    new_sample = np.array([demand for _, demand in recent[:REGIME_SAMPLE_SIZE]], dtype=float)
    if len(reference) < REGIME_SAMPLE_SIZE or len(new_sample) < REGIME_SAMPLE_SIZE:
        return True
    _, p_value = ks_2samp(reference, new_sample)
    return bool(p_value >= REGIME_TEST_ALPHA)


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


def _promote_temp_to_original(station_id: str, drop_oldest: int) -> int:
    """Once the new model is live: optionally slide the window, then fold Temp into history.

    The new Temp rows are always merged in, regardless of the regime-shift
    verdict; `drop_oldest` only controls whether the oldest historical rows
    are discarded first.
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
        recent = _fetch_recent(station_id)
        if not recent:
            continue
        same_distribution = _new_data_matches_history(station_id, recent)
        drop_oldest = 0 if same_distribution else len(recent)
        print(
            f"{station_id}: KS test ({'misma' if same_distribution else 'distinta'} distribución, "
            f"alpha={REGIME_TEST_ALPHA}); "
            + (
                "se conserva todo el histórico."
                if same_distribution
                else f"se descartarán {drop_oldest} filas antiguas al publicar el nuevo modelo."
            ),
            flush=True,
        )

        frame = _training_frame(station_id, recent, drop_oldest)
        if len(frame) <= max(LAGS):
            continue
        path = _train_station_model(station_id, frame)
        _upload_model(station_id, path)
        merged = _promote_temp_to_original(station_id, drop_oldest)
        retrained.append(station_id)
        note = f", {drop_oldest} antiguas descartadas por cambio de distribución" if drop_oldest else ""
        print(
            f"Drift en {station_id}: reentrenado con {len(frame)} observaciones "
            f"({merged} nuevas incorporadas al histórico{note}), modelo republicado en Supabase.",
            flush=True,
        )
    return retrained
