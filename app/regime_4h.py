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
    - los ultimos 2 slots caen bajo EXIT_FASTEST (0.60) (2026-10-04: la onda
      termino en ~2026-09-20 12:00 y la ventana de 4 slots tardaba en soltarla;
      el replay del stack completo dio +0.63pp tras el corte y 0.00 antes)
    - o los ultimos 4 slots caen bajo EXIT_FAST (0.60): corte brusco de la onda
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
FASTEST_WINDOW = 2
ENTER = 0.80
ENTER_RECENT = 0.75
EXIT_FAST = 0.60
EXIT_FASTEST = 0.60   # salida rapida: los ultimos 2 slots ya no repiten el periodo anterior
MARGIN = 0.10
MAX_HORIZON = 4
MAX_PERIODS = 6          # periodos previos que se promedian como maximo
PERIOD_BLOCK = 8         # slots que se miran para juzgar si un periodo previo era onda
PERIOD_MIN_ACC = 0.70    # precision lag-16 del bloque para aceptar ese periodo
# Mas lejano: y[c + 4 - 16*6 - 7 - 16] -> 119 timestamps por estacion.
HISTORY_SLOTS = MAX_HORIZON - 1 + MAX_PERIODS * PERIOD_SLOTS + (PERIOD_BLOCK - 1) + PERIOD_SLOTS - MAX_HORIZON + 1


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
    fastest_acc = _accuracy(history, station_id, cutoff, PERIOD_SLOTS, FASTEST_WINDOW)
    persistence = _accuracy(history, station_id, cutoff, horizon, LONG_WINDOW)
    if None in (long_acc, short_acc, fast_acc, fastest_acc, persistence):
        return False
    return (
        fastest_acc >= EXIT_FASTEST
        and fast_acc >= EXIT_FAST
        and long_acc >= ENTER
        and short_acc >= ENTER_RECENT
        and long_acc >= persistence + MARGIN
    )


def _period_was_wave(history: dict, station_id: str, end: datetime) -> bool:
    """El bloque de PERIOD_BLOCK slots que termina en `end` ya repetia el de
    16 slots antes (es decir, ese periodo pertenece a la onda)."""
    acc = _accuracy(history, station_id, end, PERIOD_SLOTS, PERIOD_BLOCK)
    return acc is not None and acc >= PERIOD_MIN_ACC


def wave_forecast(history: dict, station_id: str, cutoff: datetime, horizon: int) -> float | None:
    """Promedio adaptativo de los periodos previos de 4h para (estacion,
    horizonte) si la onda esta activa; None -> usar el XGBoost estandar.

    Copiar un solo periodo (y[t-16]) copia tambien su ruido; promediar varios
    lo reduce (backtest real: ~90% -> ~92.5%). Se promedian hasta MAX_PERIODS
    periodos consecutivos hacia atras, y se corta en el primero que falte o que
    no fuera onda (_period_was_wave), asi que justo tras el inicio de la onda
    solo cuenta lo que realmente oscilaba y nunca se mezcla con el regimen
    anterior. El periodo mas reciente (y[t-16]) es siempre el primero."""
    if not 1 <= horizon <= MAX_HORIZON or not wave_active(history, station_id, cutoff, horizon):
        return None
    target = cutoff + timedelta(minutes=SLOT_MINUTES * horizon)
    values = []
    for period in range(1, MAX_PERIODS + 1):
        source = target - timedelta(minutes=SLOT_MINUTES * PERIOD_SLOTS * period)
        value = history.get((station_id, source))
        if value is None or (period > 1 and not _period_was_wave(history, station_id, source)):
            break
        values.append(float(value))
    if not values:
        return None
    return max(0.0, sum(values) / len(values))
