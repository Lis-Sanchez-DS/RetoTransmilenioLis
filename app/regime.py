"""Capa general de estructura periodica, apilada sobre el XGBoost (2026-10-02).

Se ejecuta DESPUES de app/regime_4h.py (que reclama las predicciones donde la
onda de 4h esta intacta) y solo sobre las que ese respaldo dejo sin tocar. Su
trabajo es cubrir patrones parecidos pero no identicos (otro periodo, onda
invertida, otra amplitud) sin degradar el XGBoost cuando no hay estructura.

Por estacion y horizonte compiten tres pronosticos, ponderados suavemente por
su precision (1 - WAPE) en una ventana de validacion reciente:

  M0  la cadena del XGBoost, tal cual se enviaria sin esta capa (ya con el
      recorte Page-Hinkley si hay alarma).
  M1  a + b * y[objetivo-L]: se prueban TODOS los L en [MIN_PERIOD,
      MAX_PERIOD] slots (1.5h-12h); b puede ser negativo (onda invertida) o
      distinto de 1 (cambio de amplitud); `a` absorbe cambios de nivel.
  M2  cadena + a + b * e[objetivo-L], con e = real - cadena(h pasos antes):
      conserva lo que el XGBoost ya sabe y solo suma el error sistematico con
      periodo L. Sin estructura periodica b ~ 0 y M2 ~ M0.

Anti-sobreajuste (lo que la primera version apilada no tenia y la hizo perder
~1 pp en periodo calmo): (a) (a, b) de cada L se ajustan en una ventana de
AJUSTE y se puntuan en una de VALIDACION posterior e independiente; (b) un
candidato solo cuenta si su correlacion tiene el mismo signo y |r| >=
STABLE_CORR en ambas ventanas; (c) solo si supera a la cadena por GATE y
alcanza MIN_ACCURACY; (d) se promedian los TOP_K mejores lags estables en vez
de elegir uno solo. Si ningun candidato pasa, el resultado es EXACTAMENTE el
de la cadena.

Backtest real (ver README, punto 15): perdida en periodo calmo -0.03 pp con el
respaldo de 4h por delante, igual al respaldo en la onda real (89.9%).

Sin estado: todo se recalcula cada ciclo solo con observaciones <=
data_cutoff, asi que si el regimen cambia se readapta y si termina la capa se
apaga sola. El sesgo EWMA se mide sobre residuos de la cadena, por eso el
llamador lo escala por el peso de M0 (apply_generic_layer devuelve ese peso).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from app.drift import LAGS
from app.features import temporal_features

SLOT_MINUTES = 15
MIN_PERIOD = 6  # slots (1.5h); >= MAX_HORIZON para que objetivo-L ya sea real
MAX_PERIOD = 48  # slots (12h)
FIT_WINDOW = 32
VAL_WINDOW = 16
MAX_HORIZON = 4
MIN_ABS_SLOPE = 0.3  # solo M1: evita degenerar en una constante
MAX_ABS_SLOPE = 2.0
LOOKBACK_SLOTS = FIT_WINDOW + VAL_WINDOW + MAX_PERIOD  # ventana de y que usa la mezcla (96)
# La cadena de cada slot de esa ventana, h pasos antes, necesita lag_96 de ese corte:
HISTORY_SLOTS = LOOKBACK_SLOTS + max(LAGS) + MAX_HORIZON


@dataclass(frozen=True)
class GenericParams:
    gate: float = 0.15  # cuanto debe superar el candidato a la cadena en validacion
    min_accuracy: float = 0.80  # precision minima absoluta en validacion
    stable_corr: float = 0.5  # |r| minimo, mismo signo, en ajuste y validacion
    top_k: int = 3
    lam: float = 20.0  # sensibilidad de los pesos a la diferencia de precision
    chain_bonus: float = 0.02  # ventaja inicial de la cadena


DEFAULT_PARAMS = GenericParams()


def history_range(cutoff: datetime) -> tuple[datetime, datetime]:
    """(inicio, fin) de los HISTORY_SLOTS slots que la capa lee por estacion:
    cantidad fija (196), independiente del tamano de la historia acumulada."""
    return cutoff - timedelta(minutes=SLOT_MINUTES * (HISTORY_SLOTS - 1)), cutoff


def series_from_history(history: dict, station_id: str, cutoff: datetime) -> tuple[np.ndarray, list[datetime]] | None:
    """(valores cronologicos, timestamps) de los HISTORY_SLOTS slots que
    terminan en `cutoff`, o None si falta alguno."""
    times = [cutoff - timedelta(minutes=SLOT_MINUTES * k) for k in range(HISTORY_SLOTS - 1, -1, -1)]
    values = []
    for stamp in times:
        value = history.get((station_id, stamp))
        if value is None:
            return None
        values.append(float(value))
    return np.asarray(values), times


def chain_matrix(model, values: np.ndarray, times: list[datetime]) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """La misma cadena recursiva de predict_cycle_targets (sin recorte de
    banda), vectorizada sobre todos los cortes posibles del historial. Una
    prueba (tests/test_regime.py) la compara con la de produccion.

    Devuelve (past, now):
      past[h][i] = prediccion del slot i hecha h pasos antes (NaN si faltan
                   lags), para i en [0, HISTORY_SLOTS)
      now[h]     = prediccion para cutoff + h pasos (cutoff = ultimo slot)."""
    n = len(values)
    first = max(LAGS) - 1  # primer corte con todos los lags reales
    cuts = np.arange(first, n)
    base = times[0]
    cal: dict[int, list[float]] = {}

    def calendar(index: int):
        if index not in cal:
            cal[index] = temporal_features(base + timedelta(minutes=SLOT_MINUTES * index))
        return cal[index]

    preds: dict[int, np.ndarray] = {}
    for h in range(1, MAX_HORIZON + 1):
        columns = []
        for lag in LAGS:
            # Identico a predict_cycle_targets: con lag >= h el valor real es el de
            # "lag pasos antes del objetivo" (cutoff + 15*(h-lag)); con lag < h es
            # la propia prediccion de la cadena para el horizonte (h - lag).
            columns.append(values[cuts + h - lag] if lag >= h else preds[h - lag])
        calendar_cols = np.asarray([calendar(int(c) + h) for c in cuts])
        features = np.column_stack(columns + [calendar_cols[:, k] for k in range(calendar_cols.shape[1])])
        preds[h] = np.maximum(0.0, np.asarray(model.predict(features), dtype=float))
    past = {}
    now = {}
    for h in range(1, MAX_HORIZON + 1):
        arr = np.full(n, np.nan)
        slots = cuts + h
        keep = slots < n
        arr[slots[keep]] = preds[h][keep]
        past[h] = arr
        now[h] = float(preds[h][-1])
    return past, now


def _wape_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return max(0.0, 1.0 - float(np.abs(actual - predicted).sum()) / max(float(np.abs(actual).sum()), 1.0))


def _affine_fit(x: np.ndarray, t: np.ndarray) -> tuple[float, float] | None:
    var = float(np.var(x))
    if var < 1e-9:
        return None
    b = float(np.cov(x, t, bias=True)[0, 1]) / var
    b = min(max(b, -MAX_ABS_SLOPE), MAX_ABS_SLOPE)
    return float(np.mean(t)) - b * float(np.mean(x)), b


def _correlation(x: np.ndarray, t: np.ndarray) -> float:
    if np.std(x) < 1e-9 or np.std(t) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, t)[0, 1])


def _candidates(target: np.ndarray, source: np.ndarray, base: np.ndarray, y: np.ndarray, stable_corr: float) -> list[dict]:
    """Para cada L ajusta  target[i] ~ a + b * source[i-L]  en la ventana de
    ajuste y puntua  base[i] + a + b*source[i-L]  contra y en la de
    validacion. Devuelve todos los lags con (a, b) reajustados con toda la
    ventana y si su correlacion es estable entre ambas ventanas."""
    n = len(y)
    val_idx = np.arange(n - VAL_WINDOW, n)
    fit_idx = np.arange(n - VAL_WINDOW - FIT_WINDOW, n - VAL_WINDOW)
    all_idx = np.arange(n - VAL_WINDOW - FIT_WINDOW, n)
    out = []
    for period in range(MIN_PERIOD, MAX_PERIOD + 1):
        x_fit, x_all = source[fit_idx - period], source[all_idx - period]
        if not (np.isfinite(x_fit).all() and np.isfinite(x_all).all()):
            continue
        fit = _affine_fit(x_fit, target[fit_idx])
        full = _affine_fit(x_all, target[all_idx]) if fit is not None else None
        if fit is None or full is None:
            continue
        val_pred = np.maximum(0.0, base[val_idx] + fit[0] + fit[1] * source[val_idx - period])
        r_fit = _correlation(x_fit, target[fit_idx])
        r_val = _correlation(source[val_idx - period], target[val_idx])
        out.append(
            {
                "period": period,
                "accuracy": _wape_accuracy(y[val_idx], val_pred),
                "a": full[0],
                "b": full[1],
                "stable": r_fit * r_val > 0 and min(abs(r_fit), abs(r_val)) >= stable_corr,
            }
        )
    return out


def blend_station(
    values: np.ndarray,
    times: list[datetime],
    model,
    chain_output: dict[int, float],
    params: GenericParams = DEFAULT_PARAMS,
) -> dict[int, dict] | None:
    """Pronostico mezclado por horizonte para UNA estacion:
    {h: {"value", "weights": {"M0",["M1"],["M2"]}, "periods": {...}}}.

    `chain_output[h]` es lo que la cadena enviaria sin esta capa (puede ya
    traer el recorte por alarma Page-Hinkley); se usa como M0. M2 parte de la
    cadena sin recortar. None si la capa no puede evaluarse."""
    if len(values) != HISTORY_SLOTS:
        return None
    past, chain_now = chain_matrix(model, values, times)
    window = slice(HISTORY_SLOTS - LOOKBACK_SLOTS, HISTORY_SLOTS)
    y = values[window]
    n = len(y)
    val_idx = np.arange(n - VAL_WINDOW, n)
    m1_candidates = [c for c in _candidates(y, y, np.zeros(n), y, params.stable_corr) if abs(c["b"]) >= MIN_ABS_SLOPE]
    out = {}
    for h in range(1, MAX_HORIZON + 1):
        cp = past[h][window]
        if not np.isfinite(cp).all() or h not in chain_output:
            continue
        chain_accuracy = _wape_accuracy(y[val_idx], cp[val_idx])
        scores = {"M0": chain_accuracy + params.chain_bonus}
        forecasts = {"M0": float(chain_output[h])}
        periods: dict[str, list[int]] = {}
        e = y - cp
        for key, candidates in (("M1", m1_candidates), ("M2", _candidates(e, e, cp, y, params.stable_corr))):
            passing = [
                c
                for c in candidates
                if c["stable"] and c["accuracy"] >= chain_accuracy + params.gate and c["accuracy"] >= params.min_accuracy
            ]
            if not passing:
                continue
            top = sorted(passing, key=lambda c: -c["accuracy"])[: params.top_k]
            if key == "M1":
                values_h = [max(0.0, c["a"] + c["b"] * y[n - 1 + h - c["period"]]) for c in top]
            else:
                values_h = [max(0.0, chain_now[h] + c["a"] + c["b"] * e[n - 1 + h - c["period"]]) for c in top]
            forecasts[key] = float(np.mean(values_h))
            scores[key] = float(np.mean([c["accuracy"] for c in top]))
            periods[key] = [c["period"] for c in top]
        best = max(scores.values())
        raw = {k: float(np.exp(params.lam * (s - best))) for k, s in scores.items()}
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}
        out[h] = {
            "value": max(0.0, sum(weights[k] * forecasts[k] for k in weights)),
            "weights": weights,
            "periods": periods,
        }
    return out or None


def apply_generic_layer(
    predictions: list[dict],
    history: dict,
    models: dict,
    cutoff: datetime,
    skip: set[tuple[str, str]] = frozenset(),
    params_by_station: dict[str, GenericParams] | None = None,
) -> dict[tuple[str, str], float]:
    """Mezcla cada prediccion (dict con station_id, target_at, value) no
    incluida en `skip` (las ya reclamadas por la onda de 4h) modificando
    `value` en su lugar. Devuelve {(station_id, target_at): peso de M0}; el
    llamador debe escalar la correccion EWMA por ese peso. Una estacion sin
    historia completa o sin modelo no se toca (peso 1.0, valor intacto)."""
    params_by_station = params_by_station or {}
    by_station: dict[str, list[dict]] = {}
    for prediction in predictions:
        if (prediction["station_id"], prediction["target_at"]) in skip:
            continue
        by_station.setdefault(prediction["station_id"], []).append(prediction)
    chain_weight: dict[tuple[str, str], float] = {}
    for station_id, station_predictions in by_station.items():
        model = models.get(station_id)
        series = series_from_history(history, station_id, cutoff) if model is not None else None
        horizon_of: dict[int, int] = {}
        chain_output: dict[int, float] = {}
        for prediction in station_predictions:
            target_at = prediction["target_at"]
            stamp = target_at if isinstance(target_at, datetime) else datetime.fromisoformat(str(target_at).replace("Z", "+00:00"))
            h = round((stamp - cutoff).total_seconds() / (SLOT_MINUTES * 60))
            horizon_of[id(prediction)] = h
            chain_output[h] = prediction["value"]
        blended = None
        if series is not None:
            blended = blend_station(series[0], series[1], model, chain_output, params_by_station.get(station_id, DEFAULT_PARAMS))
        for prediction in station_predictions:
            key = (station_id, prediction["target_at"])
            h = horizon_of[id(prediction)]
            if blended is not None and h in blended:
                prediction["value"] = round(blended[h]["value"], 3)
                chain_weight[key] = blended[h]["weights"]["M0"]
            else:
                chain_weight[key] = 1.0
    return chain_weight
