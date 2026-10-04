"""Mezcla con la tendencia cuando la cadena va perdiendo contra la persistencia (2026-10-04).

Tras terminar la onda de 4h (~2026-09-20 12:00 virtual) cada estacion sigue un
movimiento suave propio y la cadena XGBoost tira hacia el nivel habitual de la
hora, asi que sobrepredice una estacion que cae (y subpredice una que sube)
mientras repetir el ultimo valor la supera. Esta capa vigila eso con datos
causales y solo entonces mezcla:

  ACTIVA (por estacion, solo donde no manda la onda de 4h):
    precision de la persistencia y[t-1] -> y[t] menos precision del h1 de la
    cadena (prediccion guardada) en los ultimos GATE_WINDOW (8) slots >=
    GATE_MARGIN (0.08).
  VALOR: GATE_WEIGHT (0.5) * tendencia amortiguada + 0.5 * valor actual
    (cadena + EWMA), con tendencia = y0 + TREND_SLOPE*h*(y0 - y[-2])/2 (>= 0).

Backtest del stack completo (cadena + clamp + onda + EWMA por estacion), media
por estacion: tramo calmo 09-10..09-17 -0.06pp, onda 09-18..09-20 12:00 -0.00pp,
tras la onda +1.22pp (con la salida rapida de la onda, +1.85pp en total). Margen
0.03 daba +1.66 tras la onda pero -0.50 en calma, por eso 0.08. Evidencia fina:
~7 horas de un solo regimen y algunas estaciones pierden (06111 -0.9pp tras la
onda, 05000 -0.6pp en calma); APAGAR con USE_TREND_GATE en submit_xgboost.py.
"""

from datetime import timedelta

GATE_WINDOW = 8
GATE_MARGIN = 0.08
GATE_WEIGHT = 0.5
TREND_SLOPE = 0.6
SLOT = timedelta(minutes=15)


def gate_margin(history: dict, station_id: str, pairs: list[tuple]) -> float | None:
    """persistencia - cadena (precision) sobre los ultimos GATE_WINDOW pares
    (observed_at, demand, prediction), cronologicos. None si falta cualquier dato."""
    pairs = [(at, d, p) for at, d, p in pairs if d is not None and p is not None][-GATE_WINDOW:]
    if len(pairs) < GATE_WINDOW:
        return None
    chain_err = persist_err = demand_sum = 0.0
    for observed_at, demand, prediction in pairs:
        previous = history.get((station_id, observed_at - SLOT))
        if previous is None:
            return None
        chain_err += abs(float(demand) - float(prediction))
        persist_err += abs(float(demand) - previous)
        demand_sum += abs(float(demand))
    denominator = max(demand_sum, 1.0)
    return (1 - persist_err / denominator) - (1 - chain_err / denominator)


def trend_forecast(history: dict, station_id: str, cutoff, horizon: int) -> float | None:
    last = history.get((station_id, cutoff))
    before = history.get((station_id, cutoff - 2 * SLOT))
    if last is None or before is None:
        return None
    return max(0.0, last + TREND_SLOPE * horizon * (last - before) / 2)


def blended_value(history: dict, station_id: str, cutoff, horizon: int, value: float, margin: float | None) -> float:
    """`value` mezclado con la tendencia si la compuerta esta activa; si no, igual."""
    if margin is None or margin < GATE_MARGIN:
        return value
    trend = trend_forecast(history, station_id, cutoff, horizon)
    if trend is None:
        return value
    return GATE_WEIGHT * trend + (1 - GATE_WEIGHT) * value
