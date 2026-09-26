"""Detección de drift por estación y reentrenamiento automático.

El collector guarda en "Temp" cada observación nueva junto con la
predicción que el modelo vigente le habría dado. La señal de drift es la
accuracy (la misma métrica del proyecto, 1 - WAPE) de esa estación, medida
solo sobre sus últimos RECENT_CHECKS puntos con predicción registrada (no
sobre una ventana de tiempo fija ni sobre todo lo acumulado en "Temp" desde
el último reentrenamiento) - así el disparador reacciona a cómo está
funcionando el modelo *ahora mismo* en vez de diluirse con un histórico
largo que puede mezclar tramos buenos y malos, y no se queda ciego si el
stream se ralentiza o se detiene (una ventana de tiempo fija podría no
llegar nunca a acumular puntos suficientes; un conteo fijo de chequeos sí).

Cuando una estación dispara el drift, el reentrenamiento SIEMPRE conserva
todo el histórico: los puntos nuevos de "Temp" se incorporan a
"Original Data" y nunca se descarta nada. (Antes hubo dos variantes que sí
recortaban historia - una prueba de Kolmogorov-Smirnov que decidía
conservar todo o recortar, y luego un reemplazo 1:1 fijo sin esa prueba -
ambas se quitaron el 2026-09-24 después de que una comparación directa
mostrara que un modelo entrenado con todo el histórico predice mejor que
uno entrenado solo con una ventana reciente, en 12/12 estaciones. No
reintroducir un recorte de historia sin repetir esa comparación.)
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
# Direct multi-horizon forecasting: one model per +15/+30/+45/+60min target
# instead of one model recursively fed forward. Each horizon's lag features
# are anchored to the same reference point (data_cutoff / the observation
# itself) and are always real observed values - never another horizon's own
# prediction - so error can't compound across horizons the way it did with
# the old recursive approach. See submit_xgboost.py's predict_cycle_targets.
HORIZONS = (1, 2, 3, 4)

ACCURACY_THRESHOLD = 0.85
# Count-based window, not a time window: a station's drift signal is its
# accuracy over its own last RECENT_CHECKS real prediction/actual pairs,
# whichever tick they landed on. MIN_DATAPOINTS requires the window to be
# full before trusting the signal - a station with only 1-2 checks so far
# shouldn't be judged (or retrained) off a handful of points.
RECENT_CHECKS = 4
MIN_DATAPOINTS = RECENT_CHECKS

# Chosen per station via walk-forward CV grid search restricted to the train
# split (2026-09-25): every station's own CV picked a smaller/slower model
# than the previous shared 400-estimator/40-leaf/reg_lambda=1.0 default, but
# reg_lambda itself split station by station (1.0 vs 3.0), and a single
# shared config underperformed each station's own best config on held-out
# test data - so each station keeps its own winning combination rather than
# being forced onto one global default. Mean held-out test accuracy across
# the 12 stations: 85.07% -> 85.37%, with 8/12 stations improving.
DEFAULT_MODEL_PARAMS = {"learning_rate": 0.05, "n_estimators": 400, "max_leaves": 40, "reg_lambda": 1.0}
STATION_MODEL_PARAMS = {
    "02300": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "03000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "05000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "05100": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "06000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "06111": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "07105": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "07107": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "07111": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "09000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "09122": {"learning_rate": 0.05, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "10009": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
}


def _model_params(station_id: str) -> dict:
    """A station outside the tuned set (e.g. a newly added one) falls back to
    the old shared default rather than crashing or silently getting some
    other station's config."""
    return STATION_MODEL_PARAMS.get(station_id, DEFAULT_MODEL_PARAMS)


def has_pending_data() -> bool:
    """Whether "Temp" currently holds any unpromoted observations at all.

    Used as the gate before running drift checks, in place of "did this
    exact collector run insert a new row". A stalled upstream feed can leave
    a drifted station's data sitting in "Temp" indefinitely without any run
    ever inserting a fresh row again - gating on "did this run insert
    something new" then blocks drift/retrain forever even though the
    station's already-known-bad data is sitting right there, unprocessed.
    Gating on "is there anything in Temp at all" still skips redundant
    checks once Temp is genuinely empty (e.g. every station was just
    retrained and promoted), but keeps checking as long as there's
    unprocessed data to evaluate, regardless of whether new rows arrived
    this exact cycle.
    """
    with connection() as conn:
        return conn.execute('SELECT 1 FROM "Temp" LIMIT 1').fetchone() is not None


def station_accuracy_stats() -> pd.DataFrame:
    """Accuracy and count per station over each station's own last RECENT_CHECKS
    predicted/actual pairs.

    Accuracy = max(0, 1 - WAPE), i.e. 1 - sum(|demand - prediction|) / sum(|demand|),
    the same formula as train_comparison_models.py's score() (there scaled by 100).

    A count-based window (not a time window) per station: a station that has
    fallen behind or whose data is arriving irregularly still gets judged on
    its own most recent real checks, rather than a fixed clock window that
    could span very different amounts of real data station to station.

    A retrain empties "Temp" for the stations it just touched (their rows move
    into "Original Data" - see _promote_temp_to_original). Right after that, a
    just-retrained station's own "Temp" history can be shorter than the full
    window, even though the data itself still exists - it just moved. So this
    reads both tables and lets whichever one holds each point fill it in;
    it's the same rows, just possibly in their new location.
    """
    with connection() as conn:
        rows = conn.execute(
            """WITH combined AS (
                   SELECT station_id, demand, prediction, observed_at FROM "Temp"
                   UNION ALL
                   SELECT station_id, demand, prediction, observed_at FROM "Original Data"
               ),
               ranked AS (
                   SELECT station_id, demand, prediction,
                          row_number() OVER (PARTITION BY station_id ORDER BY observed_at DESC) AS rn
                   FROM combined
                   WHERE prediction IS NOT NULL
               )
               SELECT station_id, demand, prediction FROM ranked WHERE rn <= %s""",
            (RECENT_CHECKS,),
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
    drifted = stats[(stats["count"] >= MIN_DATAPOINTS) & (stats["accuracy"] < ACCURACY_THRESHOLD)]
    return sorted(drifted["station_id"].tolist())


def _fetch_recent(station_id: str) -> list[tuple]:
    """New (Temp) observations for this station, oldest first."""
    with connection() as conn:
        return conn.execute(
            'SELECT observed_at, demand FROM "Temp" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()


def _training_frame(station_id: str, recent: list[tuple]) -> pd.DataFrame:
    """All historical data plus the new Temp observations - history only ever grows."""
    with connection() as conn:
        historical = conn.execute(
            'SELECT observed_at, demand FROM "Original Data" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()
    frame = pd.DataFrame(historical + recent, columns=["observed_at", "demand"])
    return frame.drop_duplicates(subset="observed_at").sort_values("observed_at").reset_index(drop=True)


def _train_station_horizon_model(station_id: str, horizon: int, frame: pd.DataFrame) -> str:
    """Train a model that predicts `horizon` steps (15min each) directly ahead.

    Every horizon shares the same anchor: the lag_k feature always reads
    `horizon + k - 1` steps behind the row being predicted, which works out
    to the exact same real, already-observed timestamp regardless of
    horizon (data_cutoff - 15*(k-1) minutes) - never another horizon's own
    prediction. Only the target offset and calendar encoding (computed at
    the row's own real timestamp) change between horizons; h=1 reduces to
    the original single-step model exactly.
    """
    y = frame.demand.astype(float)
    lagged = pd.concat(
        {f"lag_{lag}": y.shift(horizon + lag - 1) for lag in LAGS}, axis=1
    )
    calendar = pd.DataFrame(
        [temporal_features(value) for value in frame["observed_at"]],
        columns=["hour_sin", "hour_cos", "week_sin", "week_cos"],
    )
    features = pd.concat([lagged, calendar], axis=1)
    valid = features.notna().all(axis=1)
    model = XGBRegressor(
        grow_policy="lossguide",
        max_depth=0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **_model_params(station_id),
    )
    model.fit(features.loc[valid], y.loc[valid])
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"xgboost_{station_id}_h{horizon}.joblib")
    joblib.dump(model, path)
    return path


def _upload_model(station_id: str, horizon: int, path: str) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY para subir el modelo.")
    with open(path, "rb") as fh:
        payload = fh.read()
    response = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{MODEL_BUCKET}/xgboost/xgboost_{station_id}_h{horizon}.joblib",
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
    """Once the new model is live: fold Temp into history. History only ever grows -
    no rows are ever dropped from "Original Data" here.
    """
    with connection() as conn:
        with conn.transaction():
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

        frame = _training_frame(station_id, recent)
        if len(frame) <= max(LAGS) + max(HORIZONS):
            continue
        for horizon in HORIZONS:
            path = _train_station_horizon_model(station_id, horizon, frame)
            _upload_model(station_id, horizon, path)
        merged = _promote_temp_to_original(station_id)
        retrained.append(station_id)
        print(
            f"Drift en {station_id}: reentrenados {len(HORIZONS)} horizontes con {len(frame)} observaciones "
            f"({merged} nuevas incorporadas al histórico), modelos republicados en Supabase.",
            flush=True,
        )
    return retrained
