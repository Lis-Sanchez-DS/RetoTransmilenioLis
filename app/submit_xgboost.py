"""Genera y envía la primera submission usando los XGBoost guardados."""

from datetime import datetime, timedelta, timezone
import os
import sys

import joblib
import requests

from app import context as context_module
from app import regime as regime_module
from app import regime_4h as wave_module
from app import trend_gate
from app.collector import collect_new_data
from app.db import connection
from app.drift import (
    LAGS,
    _station_has_fresh_page_hinkley_alarm,
    blend_horizon_bias,
    station_bias_components,
    station_recent_pairs,
)
from app.features import temporal_features
from app.health import check_collector_heartbeat
from app.net import with_retries, UpstreamUnavailable


BASE_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
API_KEY = os.getenv("PULSO_API_KEY") or os.getenv("API-KEY-PulsoTransmi")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")

# While a station's Page-Hinkley alarm is fresh (a real, recent change point -
# see _station_has_fresh_page_hinkley_alarm), the recursive chain is fed its
# own predictions across a demand level that may already be stale by several
# points, so a badly-off prediction at one horizon compounds into the next.
# Clamping every horizon's value to within PH_ALARM_BAND_PCT of the last real
# observed value (lag_1 at cutoff) only when the alarm is active - never
# otherwise - was backtested across all 12 stations' full history: a tight
# band (~15%) fixes the worst compounding cases (one station's h4 accuracy
# went from 0% to 87%+ during its real alarm cutoffs) but also clips
# legitimately large, real moves on stations whose alarms fire during fast
# but genuine demand swings. 0.5 (50%) was the widest band that still kept
# nearly all of that gain while cutting the one regressor's damage from
# -26pp to -2.6pp at h4 - unlike a proportional shrink toward the anchor
# (rejected: it distorts every alarm-time prediction, not just extreme
# ones), this only touches predictions that are already off by more than the
# band, in either direction, so it helps over- and under-prediction equally.
PH_ALARM_BAND_PCT = 0.5

# Respaldo de la onda de 4h (app/regime_4h.py, 2026-10-02): desde ~2026-09-18
# las estaciones oscilan con periodo de 16 slots y repetir y[objetivo-16]
# rinde ~90% contra ~70% de la cadena. Se activa por (estacion, horizonte)
# solo mientras la onda se verifica en los datos mas recientes y se apaga sola
# ante cualquier otro patron (vuelve al XGBoost estandar). Apagar la bandera
# deja todo exactamente como antes.
USE_4H_WAVE = True

# Capa general (app/regime.py, 2026-10-02): corre DESPUES de la onda de 4h y solo
# sobre las predicciones que esta no reclamo. Aprende cualquier periodo/signo/
# amplitud de los datos recientes y mezcla suavemente con el XGBoost; si ningun
# candidato supera a la cadena por margen y estabilidad, no cambia nada. Apagarla
# deja solo la onda de 4h + XGBoost estandar. APAGADA por defecto (2026-10-02): en el
# replay del pipeline completo con datos reales no mejoro nada (0.0 pp en la onda y
# en las ultimas 6 rondas, -0.07 pp en periodo calmo) porque el unico patron real que
# hay es el de 4h, que ya cubre el respaldo; su valor es solo de seguro ante otros
# patrones y se apoya en pruebas sinteticas. Encenderla es decision del usuario.
USE_GENERIC_REGIME = False

# Compuerta de tendencia (app/trend_gate.py, 2026-10-04): tras terminar la onda
# de 4h la cadena pierde contra la persistencia; donde no manda la onda y la
# persistencia viene ganando, se mezcla 50/50 con una tendencia amortiguada.
# Apagarla deja todo exactamente como antes.
USE_TREND_GATE = True

# Marcador (por job: vive junto a los modelos, que checkout limpia al reiniciar)
# del ultimo ciclo ya enviado. El loop despierta cada 5 min pero un ciclo se
# envia una sola vez: sin esto las otras ~11 vueltas leen ~196 slots por
# estacion solo para descubrir un 409 al final.
LAST_SUBMITTED_MARKER = os.path.join(MODEL_DIR, ".last_submitted_cycle")


def _cycle_already_submitted(cycle_id: str) -> bool:
    try:
        with open(LAST_SUBMITTED_MARKER) as handle:
            return handle.read().strip() == cycle_id
    except OSError:
        return False


def _mark_cycle_submitted(cycle_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(LAST_SUBMITTED_MARKER), exist_ok=True)
        with open(LAST_SUBMITTED_MARKER, "w") as handle:
            handle.write(cycle_id)
    except OSError:
        pass  # solo es una optimizacion: sin marcador se repite el trabajo, nada mas


def _fetch_generic_history(station_ids: list[str], cutoff: datetime) -> dict:
    """Los HISTORY_SLOTS slots que la capa general necesita por estacion, en
    una sola consulta acotada por rango (nunca la tabla completa)."""
    start, end = regime_module.history_range(cutoff)
    with connection() as conn:
        rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" '
            "WHERE station_id = ANY(%s) AND observed_at BETWEEN %s AND %s "
            'UNION ALL SELECT station_id, observed_at, demand FROM "Temp" '
            "WHERE station_id = ANY(%s) AND observed_at BETWEEN %s AND %s",
            (station_ids, start, end, station_ids, start, end),
        ).fetchall()
    return {(station_id, _parse_utc(observed_at)): float(demand) for station_id, observed_at, demand in rows}


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


def lag_timestamps(cutoff: datetime, max_horizon: int = 4) -> list[datetime]:
    """Every real timestamp predict_cycle_targets can read for a cycle: for each
    horizon h and each lag k >= h, the value k*15min before the h-th target
    (cutoff + 15*(h-k)). A handful of exact points (about 11), never a scan."""
    return sorted(
        {
            cutoff + timedelta(minutes=15 * (horizon - lag))
            for horizon in range(1, max_horizon + 1)
            for lag in LAGS
            if lag >= horizon
        }
    )


def _nearest_observation(history: dict, station_id: str, stamp: datetime) -> float | None:
    """Closest recorded value for the station to `stamp` (a gap from a
    quality=missing observation); None only if it has no history at all."""
    best = None
    for (station, observed), value in history.items():
        if station != station_id:
            continue
        distance = abs((observed - stamp).total_seconds())
        if best is None or distance < best[0]:
            best = (distance, value)
    return None if best is None else best[1]


def predict_cycle_targets(
    targets: list[dict],
    history: dict,
    models: dict,
    data_cutoff,
    alarm_flags: dict[str, bool] | None = None,
    context: dict | None = None,
) -> list[dict]:
    """Predict every target by chaining the station's own h1 model forward -
    recursive, one model per station (not one per horizon).

    Replaced the direct-multi-horizon design (2026-09) on 2026-09-29 after a
    held-out backtest across all 12 stations found the dedicated h2/h3/h4
    models were consistently *worse* than just re-applying the h1 model
    recursively - not only on the 4 stations with a fresh demand-level shift,
    but on every single station, often by double digits of accuracy points at
    h4. An oracle test (feeding the chain ground truth instead of its own
    prior predictions) confirmed the direct models' shortfall isn't mostly
    about avoiding compounding error - it's that they lean heavily on
    lag_96 (or, for some stations, mostly on time-of-day) at longer
    horizons, which is a weaker signal than what the h1 model already learned
    from the tightest, freshest lag. Compounding error from feeding the
    chain its own prior predictions is real but small (1-8pp at h4) and
    mostly cancels rather than stacking, since XGBoost's own noise here is
    unbiased rather than systematic.

    For target horizon h, lag_k's feature value is "the demand k*15 minutes
    before the TARGET" (exactly what the h1 model was trained on, where lag_k
    is y.shift(k)):
      - the real, already-observed value at data_cutoff + 15*(h-k) minutes,
        if k >= h (that timestamp is at or before data_cutoff);
      - otherwise, this same chain's own prediction for horizon (h - k) -
        the only case this ever happens is k < h, i.e. lag_1 for h=2,3,4 and
        lag_2 for h=3,4 (lag_4 and lag_96 are always real, since LAGS' next
        value after 2 is 4).
    h1 itself never uses a predicted value, so it's identical to the old
    direct model's own h1 prediction.

    FIXED 2026-10-02: until then the real value for k >= h was read at
    data_cutoff - 15*(k-1) minutes (anchored at the cutoff, as the old direct
    per-horizon models were), which for h >= 2 is h-1 steps STALER than "k
    steps before the target" - lag_2, lag_4 and lag_96 fed the model values
    from the wrong moment. Backtested through the full stack (chain + clamp +
    EWMA/PH, h1 models retrained on each fold's prefix only) the target-anchored
    version improved all three chronological folds (+1.17 / +0.95 / +1.01pp
    station mean, h4 +2 to +3pp) and no station lost more than 0.1pp.

    The old h2/h3/h4 model files are intentionally left untouched in
    Supabase Storage (retraining no longer writes to them) as a cheap
    rollback path - the direct approach's code, not just its artifacts, is
    what would need reviving if this needs to be undone.

    A real lag missing from history (a quality=missing observation) is
    replaced by the station's nearest recorded value instead of skipping the
    station: the server answers 500 to a submission with fewer than the 48
    expected targets. Only a station with no history at all is skipped.

    alarm_flags maps station_id -> whether that station's Page-Hinkley alarm
    is currently fresh (see _station_has_fresh_page_hinkley_alarm); a station
    missing from the map, or a falsy value, is treated as no alarm. See
    PH_ALARM_BAND_PCT above for why an active alarm clamps every horizon's
    value to a band around the last real observed value.
    """
    cutoff = _parse_utc(data_cutoff)
    alarm_flags = alarm_flags or {}
    by_station: dict[str, list[dict]] = {}
    for target in targets:
        by_station.setdefault(target["station_id"], []).append(target)

    predictions = []
    for station_id, station_targets in by_station.items():
        model = models.get(station_id)
        if model is None:
            print(f"AVISO: sin modelo h1 para {station_id}; se omite la estación.", flush=True)
            continue

        targets_by_horizon = {}
        for target in station_targets:
            timestamp = _parse_utc(target["target_at"])
            horizon = round((timestamp - cutoff).total_seconds() / 900)
            targets_by_horizon[horizon] = target

        # (lag, horizon) -> the real value k*15min before that horizon's target,
        # for every lag that is already observed at data_cutoff (lag >= horizon).
        real_lag_values = {}
        filled = []
        for horizon in range(1, max(targets_by_horizon) + 1):
            for lag in LAGS:
                if lag >= horizon:
                    stamp = cutoff + timedelta(minutes=15 * (horizon - lag))
                    key = (station_id, stamp)
                    if key in history:
                        real_lag_values[(lag, horizon)] = history[key]
                    else:
                        substitute = _nearest_observation(history, station_id, stamp)
                        if substitute is None:
                            filled = None
                            break
                        real_lag_values[(lag, horizon)] = substitute
                        filled.append(lag)
            if filled is None:
                break
        if filled is None:
            print(f"AVISO: se omite {station_id} en este ciclo: no hay ninguna observacion", flush=True)
            continue
        if filled:
            print(
                f"AVISO: {station_id} con huecos en lags {sorted(set(filled))}; "
                "se sustituyen por la observacion mas cercana para no dejar el ciclo incompleto.",
                flush=True,
            )

        is_alarm = bool(alarm_flags.get(station_id))
        anchor = real_lag_values[(1, 1)]  # the last real observation, at data_cutoff
        band_lo = anchor * (1 - PH_ALARM_BAND_PCT)
        band_hi = anchor * (1 + PH_ALARM_BAND_PCT)

        # Walk every horizon from 1 up to the highest one actually requested,
        # even ones this cycle didn't ask for - a later horizon's chain can
        # depend on an earlier one's prediction regardless of whether that
        # earlier horizon has its own target in this cycle.
        predicted_by_horizon: dict[int, float] = {}
        for horizon in range(1, max(targets_by_horizon) + 1):
            target_ts = cutoff + timedelta(minutes=15 * horizon)
            feature_values = []
            for lag in LAGS:
                if lag >= horizon:
                    feature_values.append(real_lag_values[(lag, horizon)])
                else:
                    feature_values.append(predicted_by_horizon[horizon - lag])
            features = feature_values + temporal_features(target_ts)
            if context_module.USE_CONTEXT_FEATURES:
                features += context_module.context_features(target_ts, context)
            value = max(0.0, float(model.predict([features])[0]))
            if is_alarm:
                value = min(max(value, band_lo), band_hi)
            predicted_by_horizon[horizon] = value
            if horizon in targets_by_horizon:
                predictions.append(
                    {
                        "station_id": station_id,
                        "target_at": targets_by_horizon[horizon]["target_at"],
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
    # A timeout on the observations stream must not block the submission: the
    # cycle window is short and the history already in the DB is enough to predict.
    try:
        collect_result = collect_new_data()
        print(
            f"Collected {collect_result['collected']} observations "
            f"({collect_result['inserted']} new) in {collect_result['pages']} pages; "
            f"cursor={collect_result['cursor']}",
            flush=True,
        )
    except UpstreamUnavailable as exc:
        print(f"AVISO: no se pudo leer el stream ({exc}); se envia con la historia ya guardada.", flush=True)
    identity = api_get("/v1/me")
    try:
        cycle = api_get("/v1/forecast-cycles/current")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    if _cycle_already_submitted(cycle["cycle_id"]):
        print(f"Ciclo {cycle['cycle_id']} ya enviado por este job; se omite.", flush=True)
        check_collector_heartbeat()
        return None
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
    needed_timestamps = lag_timestamps(cutoff)
    if USE_4H_WAVE:
        # 32 slots mas por estacion: lo que necesita wave_module para decidir.
        needed_timestamps = sorted(set(needed_timestamps) | set(wave_module.history_timestamps(cutoff)))
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
        station_id: joblib.load(os.path.join(MODEL_DIR, f"xgboost_{station_id}_h1.joblib"))
        for station_id in station_ids
    }
    # One bounded read per station (see PH_LOOKBACK_ROWS), same cost class as
    # station_bias_components below - detection only, independent of whether
    # that station's separate PH bias-boost flag is on.
    alarm_flags = {
        station_id: _station_has_fresh_page_hinkley_alarm(station_id)
        for station_id in station_ids
    }
    cycle_context = None
    if context_module.USE_CONTEXT_FEATURES:
        target_times = [_parse_utc(target["target_at"]) for target in cycle["targets"]]
        try:
            cycle_context = context_module.fetch_context(min(target_times), max(target_times))
        except Exception as exc:
            print(f"AVISO: no se pudo leer /v1/context ({exc}); se predice sin contexto.", flush=True)
            cycle_context = {}
    predictions = predict_cycle_targets(
        cycle["targets"], history, models, cycle["data_cutoff"], alarm_flags, cycle_context
    )
    if not predictions:
        raise RuntimeError("Ninguna estación tenía suficiente historia para este ciclo; no se envía submission.")

    # Onda de 4h: reemplaza el valor de la cadena SOLO donde la onda se
    # verifica ahora mismo (wave_module.wave_forecast devuelve None en cualquier
    # otro caso). Cualquier error aqui se degrada al XGBoost estandar: esta capa
    # nunca debe impedir una submission.
    wave_overridden: set[tuple[str, str]] = set()
    if USE_4H_WAVE:
        try:
            for prediction in predictions:
                target_ts = _parse_utc(prediction["target_at"])
                horizon = round((target_ts - cutoff).total_seconds() / 900)
                value = wave_module.wave_forecast(history, prediction["station_id"], cutoff, horizon)
                if value is not None:
                    prediction["value"] = round(value, 3)
                    wave_overridden.add((prediction["station_id"], prediction["target_at"]))
            print(
                f"Onda 4h: {len(wave_overridden)}/{len(predictions)} predicciones reemplazadas "
                f"({len({station for station, _ in wave_overridden})} estaciones); el resto usa el XGBoost estandar.",
                flush=True,
            )
        except Exception as exc:
            print(f"AVISO: capa de onda 4h fallo ({exc!r}); se usa el XGBoost estandar.", flush=True)

    # Capa general: mezcla suave con el XGBoost solo donde la onda de 4h no
    # reclamo la prediccion Y hay evidencia periodica estable y claramente mejor.
    # Se trabaja sobre una copia de seguridad: cualquier error restaura la cadena
    # estandar intacta (esta capa nunca debe impedir ni corromper una submission).
    chain_weight: dict[tuple[str, str], float] = {}
    if USE_GENERIC_REGIME and len(wave_overridden) < len(predictions):
        snapshot = [dict(prediction) for prediction in predictions]
        try:
            generic_history = _fetch_generic_history(station_ids, cutoff)
            chain_weight = regime_module.apply_generic_layer(
                predictions, generic_history, models, cutoff, skip=wave_overridden
            )
            blended = sum(1 for weight in chain_weight.values() if weight < 0.999)
            print(
                f"Capa general: {blended}/{len(chain_weight)} predicciones mezcladas con estructura periodica; "
                "el resto usa el XGBoost estandar.",
                flush=True,
            )
        except Exception as exc:
            for prediction, saved in zip(predictions, snapshot):
                prediction.update(saved)
            chain_weight = {}
            print(f"AVISO: capa general fallo ({exc!r}); se usa el XGBoost estandar.", flush=True)

    # Bias correction on top of the raw model output - see
    # station_bias_components's docstring. One DB read per station per
    # submission (not per target, not per horizon), and never touches what
    # collector.py stores as "prediction" for drift monitoring, which stays
    # the model's own raw, uncorrected output. The correction itself is
    # horizon-aware: horizon 1 (+15min) gets the full boosted bias, later
    # horizons blend back toward the plain/regular bias (see
    # EWMA_DECAY_POWER in drift.py) - computed locally per target from the
    # single fetched (regular, boosted) pair, so this stays one DB read per
    # station regardless of how many of its 4 horizons appear in this cycle.
    bias_components_by_station = {}
    for prediction in predictions:
        station_id = prediction["station_id"]
        if (station_id, prediction["target_at"]) in wave_overridden:
            # El EWMA se mide sobre residuos de la cadena: no aplica a un valor
            # que ya no viene de ella.
            continue
        if station_id not in bias_components_by_station:
            bias_components_by_station[station_id] = station_bias_components(station_id)
        regular_bias, boosted_bias = bias_components_by_station[station_id]
        target_ts = _parse_utc(prediction["target_at"])
        horizon = round((target_ts - cutoff).total_seconds() / 900)
        bias = blend_horizon_bias(regular_bias, boosted_bias, horizon)
        # El sesgo se mide sobre residuos de la cadena: solo aplica a su parte de
        # la mezcla (peso 1.0 si la capa general no toco esta prediccion).
        bias *= chain_weight.get((station_id, prediction["target_at"]), 1.0)
        prediction["value"] = round(max(0.0, prediction["value"] + bias), 3)

    # Compuerta de tendencia: despues del EWMA y solo donde no mando la onda.
    # Cualquier error se degrada al valor ya calculado y nunca bloquea el envio.
    if USE_TREND_GATE:
        try:
            margins: dict[str, float | None] = {}
            blended = 0
            for prediction in predictions:
                station_id = prediction["station_id"]
                if (station_id, prediction["target_at"]) in wave_overridden:
                    continue
                if station_id not in margins:
                    pairs = [
                        (_parse_utc(observed_at), demand, predicted)
                        for observed_at, demand, predicted in station_recent_pairs(
                            station_id, cutoff, trend_gate.GATE_WINDOW
                        )
                    ]
                    margins[station_id] = trend_gate.gate_margin(history, station_id, pairs)
                horizon = round((_parse_utc(prediction["target_at"]) - cutoff).total_seconds() / 900)
                value = trend_gate.blended_value(
                    history, station_id, cutoff, horizon, prediction["value"], margins[station_id]
                )
                if value != prediction["value"]:
                    prediction["value"] = round(value, 3)
                    blended += 1
            print(f"Compuerta de tendencia: {blended}/{len(predictions)} predicciones mezcladas.", flush=True)
        except Exception as exc:
            print(f"AVISO: compuerta de tendencia fallo ({exc!r}); se usan los valores sin mezclar.", flush=True)

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
            _mark_cycle_submitted(cycle["cycle_id"])
            check_collector_heartbeat()
            return None
        raise
    _mark_cycle_submitted(cycle["cycle_id"])
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
