"""Capa adaptativa de estructura periodica, apilada sobre el XGBoost (2026-10-02).

EXPERIMENTAL - NO ESTA CABLEADA EN PRODUCCION y no tiene pruebas. Lo que corre es
app/regime_4h.py (ver README, punto 15). En backtest real perdio ~1.1 pp en
periodo calmo y quedo por debajo del respaldo de 4h desde el inicio de la onda.

Contexto: desde ~2026-09-18 11:15 UTC todas las estaciones oscilan con periodo
de 4h (16 slots de 15min, ~81% de la potencia espectral). Los modelos
entrenados sobre historia (lags 1,2,4,96 + calendario 24h/semanal) no pueden
verlo: la cadena recursiva rindio ~70% en esas ventanas mientras repetir el
valor de 16 slots atras rindio ~90%. Antes del inicio la misma regla rendia
12-28%.

Esto NO es un disparador fijo de "onda de 4h" ni un interruptor todo/nada. Por
estacion y horizonte compiten tres pronosticos, ponderados suavemente por su
precision (1 - WAPE) en una ventana de validacion reciente:

  M0  la cadena del XGBoost, tal cual se enviaria sin esta capa.
  M1  y[objetivo-L] reescalado:  a + b * y[objetivo-L]. Se prueban TODOS los L
      en [MIN_PERIOD, MAX_PERIOD] slots (1.5h-12h); b puede ser negativo (onda
      invertida) o distinto de 1 (cambio de amplitud); `a` absorbe niveles.
  M2  cadena + correccion periodica de SU PROPIO error:
      cadena + a + b * e[objetivo-L], con e = real - cadena(h pasos antes).
      Conserva lo que el XGBoost ya sabe (nivel, calendario) y solo suma lo
      que se equivoca de forma sistematica con periodo L. Sin estructura
      periodica b ~ 0 y M2 ~ M0.

Los parametros (a, b) de cada L se ajustan en una ventana de AJUSTE y cada L
se puntua en una ventana de VALIDACION posterior e independiente; gana el
mejor L de cada familia, asi la eleccion entre ~40 lags no esta sobreajustada
en-muestra. Pesos: softmax(LAMBDA * (precision + bono_cadena)). Sin
estructura periodica la cadena domina y nada cambia; si el regimen cambia
(otro periodo, fase invertida) se readapta en ~VAL_WINDOW..FIT_WINDOW slots, y
si termina la capa se apaga sola. No guarda estado: todo se recalcula cada
ciclo solo con observaciones <= data_cutoff.
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
class RegimeParams:
    lam: float = 20.0  # sensibilidad de los pesos a la diferencia de precision
    chain_bonus: float = 0.02  # ventaja inicial de la cadena (evita saltos por ruido)


DEFAULT_PARAMS = RegimeParams()


def history_range(cutoff: datetime) -> tuple[datetime, datetime]:
    """(inicio, fin) de los HISTORY_SLOTS slots que la capa lee por estacion:
    cantidad fija (~196), independiente del tamano de la historia acumulada."""
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
    banda), vectorizada sobre todos los cortes posibles del historial.

    Devuelve (past, now):
      past[h][i] = prediccion del slot i hecha h pasos antes (NaN si faltan
                   lags), para i en [0, HISTORY_SLOTS)
      now[h]     = prediccion para cutoff + h pasos (cutoff = ultimo slot)."""
    n = len(values)
    first = max(LAGS) - 1  # primer corte con todos los lags reales
    cuts = np.arange(first, n)
    base = times[0]
    cal = {}

    def calendar(index: int):
        if index not in cal:
            cal[index] = temporal_features(base + timedelta(minutes=SLOT_MINUTES * index))
        return cal[index]

    preds: dict[int, np.ndarray] = {}
    for h in range(1, MAX_HORIZON + 1):
        columns = []
        for lag in LAGS:
            offset = h - lag
            columns.append(values[cuts + offset] if offset <= 0 else preds[offset])
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


def _best_lag_model(target: np.ndarray, source: np.ndarray, base: np.ndarray, y: np.ndarray, min_slope: float):
    """Para cada L ajusta  target[i] ~ a + b * source[i-L]  en la ventana de
    ajuste, puntua  base[i] + a + b*source[i-L]  contra y en la ventana de
    validacion y devuelve (L, accuracy_val, a_full, b_full) del mejor L, con
    (a_full, b_full) reajustados con toda la ventana; None si ninguno sirve."""
    n = len(y)
    val_idx = np.arange(n - VAL_WINDOW, n)
    fit_idx = np.arange(n - VAL_WINDOW - FIT_WINDOW, n - VAL_WINDOW)
    all_idx = np.arange(n - VAL_WINDOW - FIT_WINDOW, n)
    best = None
    for period in range(MIN_PERIOD, MAX_PERIOD + 1):
        x_fit, x_all = source[fit_idx - period], source[all_idx - period]
        if not (np.isfinite(x_fit).all() and np.isfinite(x_all).all()):
            continue
        fit = _affine_fit(x_fit, target[fit_idx])
        if fit is None or abs(fit[1]) < min_slope:
            continue
        val_pred = np.maximum(0.0, base[val_idx] + fit[0] + fit[1] * source[val_idx - period])
        accuracy = _wape_accuracy(y[val_idx], val_pred)
        if best is None or accuracy > best[1]:
            full = _affine_fit(x_all, target[all_idx])
            if full is not None:
                best = (period, accuracy, full[0], full[1])
    return best


def blend_station(
    values: np.ndarray,
    times: list[datetime],
    model,
    chain_output: dict[int, float],
    params: RegimeParams = DEFAULT_PARAMS,
) -> dict[int, dict] | None:
    """Pronostico mezclado por horizonte para UNA estacion:
    {h: {"value", "weights": {"M0","M1","M2"}, "periods": {...}}}.

    `chain_output[h]` es lo que la cadena enviaria sin esta capa (puede ya
    traer el recorte por alarma Page-Hinkley); se usa como M0. M2 parte de la
    cadena sin recortar. None si la capa no puede evaluarse (la cadena queda
    exactamente igual)."""
    if len(values) != HISTORY_SLOTS:
        return None
    past, chain_now = chain_matrix(model, values, times)
    window = slice(HISTORY_SLOTS - LOOKBACK_SLOTS, HISTORY_SLOTS)
    y = values[window]
    n = len(y)
    val_idx = np.arange(n - VAL_WINDOW, n)
    out = {}
    m1 = _best_lag_model(y, y, np.zeros(n), y, MIN_ABS_SLOPE)  # no depende del horizonte
    for h in range(1, MAX_HORIZON + 1):
        cp = past[h][window]
        if not np.isfinite(cp).all() or h not in chain_output:
            continue
        scores = {"M0": _wape_accuracy(y[val_idx], cp[val_idx]) + params.chain_bonus}
        forecasts = {"M0": float(chain_output[h])}
        periods = {}
        if m1 is not None:
            period, accuracy, a, b = m1
            scores["M1"] = accuracy
            periods["M1"] = period
            forecasts["M1"] = max(0.0, a + b * y[n - 1 + h - period])
        e = y - cp
        m2 = _best_lag_model(e, e, cp, y, 0.0)
        if m2 is not None:
            period, accuracy, a, b = m2
            scores["M2"] = accuracy
            periods["M2"] = period
            forecasts["M2"] = max(0.0, chain_now[h] + a + b * e[n - 1 + h - period])
        top = max(scores.values())
        raw = {k: float(np.exp(params.lam * (s - top))) for k, s in scores.items()}
        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}
        out[h] = {
            "value": max(0.0, sum(weights[k] * forecasts[k] for k in weights)),
            "weights": weights,
            "periods": periods,
        }
    return out or None


def apply_regime(
    predictions: list[dict],
    history: dict,
    models: dict,
    cutoff: datetime,
    params_by_station: dict[str, RegimeParams] | None = None,
) -> dict[tuple[str, str], float]:
    """Para cada prediccion (dict con station_id, target_at, value) devuelve
    {(station_id, target_at): peso de la cadena} y MODIFICA `value` en su
    lugar con la mezcla. El llamador debe escalar la correccion EWMA por ese
    peso: el sesgo se mide sobre residuos de la cadena y solo aplica a su
    parte. Una estacion sin historia completa o sin modelo no se toca (peso
    1.0, valor intacto)."""
    params_by_station = params_by_station or {}
    by_station: dict[str, list[dict]] = {}
    for prediction in predictions:
        by_station.setdefault(prediction["station_id"], []).append(prediction)
    chain_weight: dict[tuple[str, str], float] = {}
    for station_id, station_predictions in by_station.items():
        blended = None
        model = models.get(station_id)
        series = series_from_history(history, station_id, cutoff) if model is not None else None
        if series is not None:
            horizon_of = {}
            chain_output = {}
            for prediction in station_predictions:
                target_at = prediction["target_at"]
                stamp = target_at if isinstance(target_at, datetime) else datetime.fromisoformat(str(target_at).replace("Z", "+00:00"))
                h = round((stamp - cutoff).total_seconds() / (SLOT_MINUTES * 60))
                horizon_of[id(prediction)] = h
                chain_output[h] = prediction["value"]
            blended = blend_station(series[0], series[1], model, chain_output, params_by_station.get(station_id, DEFAULT_PARAMS))
        for prediction in station_predictions:
            key = (station_id, prediction["target_at"])
            h = horizon_of.get(id(prediction)) if blended is not None else None
            if blended is not None and h in blended:
                prediction["value"] = round(blended[h]["value"], 3)
                chain_weight[key] = blended[h]["weights"]["M0"]
            else:
                chain_weight[key] = 1.0
    return chain_weight
