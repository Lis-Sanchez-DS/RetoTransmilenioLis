"""Genera y envía la primera submission usando los XGBoost guardados."""

from datetime import datetime, timedelta, timezone
import os
import sys

import joblib
import requests

from app.collector import collect_new_data
from app.db import connection
from app.drift import LAGS, station_ewma_bias
from app.features import temporal_features
from app.health import check_collector_heartbeat
from app.net import with_retries, UpstreamUnavailable


BASE_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.getenv("PULSO_API_KEY") or os.getenv("API-KEY-PulsoTransmi")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
HORIZONS = (1, 2, 3, 4)


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


def _parse_utc(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def predict_cycle_targets(targets: list[dict], history: dict, models: dict, data_cutoff) -> list[dict]:
    """Predict every target directly - one model per horizon, no recursion.

    Each station gets 4 targets (+15/+30/+45/+60 min from data_cutoff), and
    each horizon has its own model (see drift.py's _train_station_horizon_model).
    Crucially, every horizon's lag_k feature is anchored to the same real,
    already-observed point - data_cutoff - 15*(k-1) minutes - regardless of
    which horizon is being predicted, so the lag features are identical
    across all 4 targets for a station and never depend on another horizon's
    own prediction the way the old recursive approach did.

    Every station/horizon model shares the same LAGS feature set (see
    drift.py's comment on why lag_672 was dropped for all of them on
    2026-09-27), so the feature vector built here always matches what every
    model was actually trained on - no more per-pair lag sets to track. A
    station missing history for one of its lags is skipped only for that
    horizon's target, not the whole station - a gap shouldn't zero out that
    station's other 3 targets, let alone every other station's submission.
    """
    cutoff = _parse_utc(data_cutoff)
    by_station: dict[str, list[dict]] = {}
    for target in targets:
        by_station.setdefault(target["station_id"], []).append(target)

    predictions = []
    for station_id, station_targets in by_station.items():
        lag_values = {}
        for lag in LAGS:
            key = (station_id, cutoff - timedelta(minutes=15 * (lag - 1)))
            if key in history:
                lag_values[lag] = history[key]

        for target in station_targets:
            timestamp = _parse_utc(target["target_at"])
            horizon = round((timestamp - cutoff).total_seconds() / 900)
            model = models.get((station_id, horizon))
            if model is None:
                print(f"AVISO: sin modelo para {station_id} horizonte {horizon}; se omite ese target.", flush=True)
                continue
            missing = [lag for lag in LAGS if lag not in lag_values]
            if missing:
                print(
                    f"AVISO: se omite {station_id} horizonte {horizon} en este ciclo: "
                    f"no hay suficiente historia en lag {missing[0]}",
                    flush=True,
                )
                continue
            features = [lag_values[lag] for lag in LAGS] + temporal_features(timestamp)
            value = max(0.0, float(model.predict([features])[0]))
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
    #
    # This call shares the same "collector_state" cursor with collector.py's
    # own collect_new_data() call, and this one runs 6x more often (every
    # ~5min here vs ~30min there) - so it routinely wins the race and drains
    # each new batch before collector.yml's own check ever sees it, making
    # collector.yml legitimately log "0 new" even while data keeps flowing.
    # Logging the count here (never logged before) makes that race visible
    # instead of something that has to be inferred from timing after the
    # fact - see has_pending_data's docstring for why collector.py's drift
    # check no longer depends on seeing "new" data in its own call.
    collect_result = collect_new_data()
    print(
        f"Collected {collect_result['collected']} observations "
        f"({collect_result['inserted']} new) in {collect_result['pages']} pages; "
        f"cursor={collect_result['cursor']}",
        flush=True,
    )
    identity = api_get("/v1/me")
    try:
        cycle = api_get("/v1/forecast-cycles/current")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    cutoff = _parse_utc(cycle["data_cutoff"])
    station_ids = sorted({target["station_id"] for target in cycle["targets"]})
    # Every target's lag features only ever look up cutoff - 15*(lag-1)
    # minutes for lag in LAGS (predict_cycle_targets, same anchor for every
    # horizon) - a handful of specific timestamps shared by every station in
    # the cycle, never anything else. Pulling the two full tables here
    # instead (as before) meant re-transferring ALL of "Original Data" -
    # tens of thousands of rows and growing - on every ~5min submissions
    # tick, by far the largest source of Supabase DB egress. Filtering by
    # the exact points needed cuts that to at most len(station_ids)*len(LAGS)
    # rows per table.
    needed_timestamps = [cutoff - timedelta(minutes=15 * (lag - 1)) for lag in LAGS]
    with connection() as conn:
        # "Temp" holds every observation collected since the last drift
        # retrain for a station (see drift.py); only drifted stations ever
        # get folded into "Original Data". Both tables must be read here or
        # any non-drifted station's history goes stale the moment new data
        # stops landing in "Original Data".
        original = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" '
            'WHERE station_id = ANY(%s) AND observed_at = ANY(%s)',
            (station_ids, needed_timestamps),
        ).fetchall()
        recent = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Temp" '
            'WHERE station_id = ANY(%s) AND observed_at = ANY(%s)',
            (station_ids, needed_timestamps),
        ).fetchall()
    history = {
        (station_id, _parse_utc(observed_at)): float(demand)
        for station_id, observed_at, demand in original + recent
    }

    models = {
        (station_id, horizon): joblib.load(os.path.join(MODEL_DIR, f"xgboost_{station_id}_h{horizon}.joblib"))
        for station_id in station_ids
        for horizon in HORIZONS
    }
    predictions = predict_cycle_targets(cycle["targets"], history, models, cycle["data_cutoff"])
    if not predictions:
        raise RuntimeError("Ninguna estación tenía suficiente historia para este ciclo; no se envía submission.")

    # Bias correction on top of the raw model output - see station_ewma_bias's
    # docstring. One DB read per station per submission (not per target), and
    # never touches what collector.py stores as "prediction" for drift
    # monitoring, which stays the model's own raw, uncorrected output.
    bias_by_station = {}
    for prediction in predictions:
        station_id = prediction["station_id"]
        if station_id not in bias_by_station:
            bias_by_station[station_id] = station_ewma_bias(station_id)
        prediction["value"] = round(max(0.0, prediction["value"] + bias_by_station[station_id]), 3)

    run_id = f"run-{cycle['cycle_id']}"
    model_info = {
        "version": "v1",
        "trained_at": datetime.fromtimestamp(
            os.path.getmtime(os.path.join(MODEL_DIR, f"xgboost_{station_ids[0]}_h1.joblib")),
            timezone.utc,
        ).isoformat(),
        "training_data_end": cycle["data_cutoff"],
    }
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

    try:
        receipt = with_retries(_post)
    except requests.HTTPError as exc:
        # The forecast cycle window (currently longer than our 5-minute poll
        # interval) can still be "current" on the next loop iteration after
        # we already submitted for it. The server rejects the repeat as a
        # real conflict rather than replaying the original response, so this
        # is an expected steady-state case, not a failure - treat it as a
        # no-op instead of letting it count toward the loop's fail counter.
        if exc.response is not None and exc.response.status_code == 409:
            print(f"Ciclo {cycle['cycle_id']} ya tenía una submission enviada; se omite.", flush=True)
            check_collector_heartbeat()
            return None
        raise
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
    # Checked last, after the submission is already out: a stale collector
    # shouldn't block a submission that otherwise succeeded, but it should
    # still surface as a loud failure so it doesn't go unnoticed.
    check_collector_heartbeat()
    return result


def main() -> None:
    try:
        submit_current_cycle()
    except UpstreamUnavailable as exc:
        print(f"AVISO: {exc}", flush=True)
        sys.exit(75)


if __name__ == "__main__":
    main()
