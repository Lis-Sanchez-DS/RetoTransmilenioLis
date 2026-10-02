"""Respaldo simple: solo la onda de 4h, con retorno rapido al XGBoost (2026-10-02).

Alternativa conservadora a app/regime.py (la capa que aprende cualquier
periodo y mezcla con el XGBoost). Esta version sabe UNA cosa: desde
~2026-09-18 11:15 UTC las 12 estaciones oscilan con periodo de 16 slots (4h) y
repetir y[objetivo - 16] rinde ~90% contra ~70% de la cadena. Por eso es
un interruptor, no una mezcla, y se apaga ante cualquier senal de que la onda
se rompe:

  ENTRA (todas):
    - precision de y[i] ~ y[i-16] en los ultimos 16 slots >= ENTER (0.80)
    - ... y en los ultimos 8 slots >= ENTER_RECENT (0.75)
    - ... y supera por MARGIN (0.10) a la persistencia al mismo horizonte.
  SALE (cualquiera, se evalua en cada ciclo, sin estado):
    - los ultimos 4 slots caen bajo EXIT_FAST (0.60): corte brusco de la onda
    - o dejan de cumplirse las condiciones de entrada.
  Si falta cualquier dato, o el periodo cambia/se invierte/aparece ruido
  distinto, la precision de lag-16 cae y el resultado es exactamente el del
  XGBoost estandar (cadena + clamp Page-Hinkley + EWMA), sin tocar nada.

Cuando esta activo el valor reemplaza al de la cadena y NO se le suma la
correccion EWMA (se calcula con residuos de la cadena).
"""

from datetime import datetime, timedelta

PERIOD_SLOTS = 16
SLOT_MINUTES = 15
LONG_WINDOW = 16
SHORT_WINDOW = 8
FAST_WINDOW = 4
ENTER = 0.80
ENTER_RECENT = 0.75
EXIT_FAST = 0.60
MARGIN = 0.10
MAX_HORIZON = 4
HISTORY_SLOTS = LONG_WINDOW + PERIOD_SLOTS  # 32 timestamps por estacion


def history_timestamps(cutoff: datetime) -> list[datetime]:
    return [cutoff - timedelta(minutes=SLOT_MINUTES * k) for k in range(HISTORY_SLOTS)]


def _accuracy(history: dict, station_id: str, cutoff: datetime, lag: int, window: int) -> float | None:
    abs_err = abs_dem = 0.0
    for k in range(window):
        stamp = cutoff - timedelta(minutes=SLOT_MINUTES * k)
        actual = history.get((station_id, stamp))
        earlier = history.get((station_id, stamp - timedelta(minutes=SLOT_MINUTES * lag)))
        if actual is None or earlier is None:
            return None
        abs_err += abs(actual - earlier)
        abs_dem += abs(actual)
    return max(0.0, 1.0 - abs_err / max(abs_dem, 1.0))


def wave_active(history: dict, station_id: str, cutoff: datetime, horizon: int) -> bool:
    long_acc = _accuracy(history, station_id, cutoff, PERIOD_SLOTS, LONG_WINDOW)
    short_acc = _accuracy(history, station_id, cutoff, PERIOD_SLOTS, SHORT_WINDOW)
    fast_acc = _accuracy(history, station_id, cutoff, PERIOD_SLOTS, FAST_WINDOW)
    persistence = _accuracy(history, station_id, cutoff, horizon, LONG_WINDOW)
    if None in (long_acc, short_acc, fast_acc, persistence):
        return False
    return (
        fast_acc >= EXIT_FAST
        and long_acc >= ENTER
        and short_acc >= ENTER_RECENT
        and long_acc >= persistence + MARGIN
    )


def wave_forecast(history: dict, station_id: str, cutoff: datetime, horizon: int) -> float | None:
    """y[cutoff + h - 16] (siempre real, h <= 4) si la onda esta activa para
    (estacion, horizonte); None -> usar el XGBoost estandar."""
    if not 1 <= horizon <= MAX_HORIZON or not wave_active(history, station_id, cutoff, horizon):
        return None
    value = history.get((station_id, cutoff + timedelta(minutes=SLOT_MINUTES * (horizon - PERIOD_SLOTS))))
    return None if value is None else max(0.0, float(value))
