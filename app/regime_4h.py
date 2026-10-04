"""Respaldo simple: onda de 4h u 8h, con retorno rapido al XGBoost (2026-10-02, ampliado 2026-10-04).

Alternativa conservadora a app/regime.py (la capa que aprende cualquier
periodo y mezcla con el XGBoost). Esta version sabe UNA cosa: desde
~2026-09-18 11:15 UTC las 12 estaciones oscilan con periodo de 16 slots (4h) y
repetir y[objetivo - 16] rinde ~90% contra ~70% de la cadena. Por eso es
un interruptor, no una mezcla, y se apaga ante cualquier senal de que la onda
se rompe:

  2026-10-04: ~2026-09-20 12:00 UTC la onda paso de 16 a 32 slots (8h). La capa
  fija en 16 quedo apagada (0/48 en el ciclo de las 06:00 del 21-sep) mientras
  y[t-32] rinde ~90% en los ultimos 6 ciclos contra ~72-78% de la cadena. Ahora
  se prueban los periodos PERIODS=(16, 32) con las mismas reglas y gana el mas corto
  salvo que el largo sea claramente mejor; los huecos de datos se toleran.

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

PERIOD_SLOTS = 16        # periodo original (4h); se conserva como referencia/compatibilidad
# Periodos candidatos (slots). 2026-10-04: ~2026-09-20 12:00 UTC la onda de 4h
# paso a repetirse cada 32 slots (8h); y[t-32] rinde ~90% en los ultimos 6
# ciclos contra ~72% de la cadena, y la capa fija en 16 quedo apagada.
PERIODS = (16, 32)
PERIOD_SWITCH_MARGIN = 0.05              # un periodo mas largo debe ganar por esto para desplazar al mas corto
# Periodos previos a promediar por periodo. Con 32 solo 1: promediar periodos mas
# viejos mezclaba la era de la onda de 4h (donde lag-32 tambien ajusta, con otra
# amplitud) y bajaba el replay de 90.3% a 68-75%; con la guarda de consistencia
# entre periodos quedaba en 89.4-90.1%, nunca por encima de copiar un periodo.
MAX_PERIODS_BY_PERIOD = {16: 6, 32: 1}
SLOT_MINUTES = 15
LONG_WINDOW = 16
SHORT_WINDOW = 8
FAST_WINDOW = 4
FASTEST_WINDOW = 2
ENTER = 0.80
ENTER_RECENT = 0.75
EXIT_FAST = 0.60
EXIT_FASTEST = 0.60   # salida rapida: los ultimos 2 slots ya no repiten el periodo anterior
MARGIN = 0.05           # 2026-10-04: 0.10 -> 0.05; con onda de 8h la persistencia a +15min ya rinde ~87%
MAX_HORIZON = 4
PERIOD_BLOCK = 8         # slots que se miran para juzgar si un periodo previo era onda
PERIOD_MIN_ACC = 0.70    # precision lag-16 del bloque para aceptar ese periodo
# Mas lejano, por periodo P con M periodos: y[c + 4 - P*M - 7 - P]. Con (16, 6)
# son 119 timestamps por estacion y con (32, 1) son 71; se lee el maximo.
HISTORY_SLOTS = max(
    MAX_HORIZON - 1 + MAX_PERIODS_BY_PERIOD[period] * period + (PERIOD_BLOCK - 1) + period - MAX_HORIZON + 1
    for period in PERIODS
)


def history_timestamps(cutoff: datetime) -> list[datetime]:
    return [cutoff - timedelta(minutes=SLOT_MINUTES * k) for k in range(HISTORY_SLOTS)]


MIN_PAIR_SHARE = 0.75    # fraccion minima de pares (actual, anterior) presentes en una ventana


def _accuracy(history: dict, station_id: str, cutoff: datetime, lag: int, window: int) -> float | None:
    """Precision de y[i] ~ y[i-lag] sobre `window` slots. Los huecos
    (quality=missing no se guardan) se omiten en vez de invalidar la ventana: con
    periodo 32 un solo hueco en 16+32 slots dejaba la capa apagada ~50% del
    tiempo. Se exige MIN_PAIR_SHARE de los pares y siempre el ultimo slot."""
    abs_err = abs_dem = 0.0
    used = 0
    for k in range(window):
        stamp = cutoff - timedelta(minutes=SLOT_MINUTES * k)
        actual = history.get((station_id, stamp))
        earlier = history.get((station_id, stamp - timedelta(minutes=SLOT_MINUTES * lag)))
        if actual is None or earlier is None:
            if k == 0:
                return None
            continue
        abs_err += abs(actual - earlier)
        abs_dem += abs(actual)
        used += 1
    if used < max(1, MIN_PAIR_SHARE * window):
        return None
    return max(0.0, 1.0 - abs_err / max(abs_dem, 1.0))


def _period_accuracies(history: dict, station_id: str, cutoff: datetime, horizon: int, period: int):
    return (
        _accuracy(history, station_id, cutoff, period, LONG_WINDOW),
        _accuracy(history, station_id, cutoff, period, SHORT_WINDOW),
        _accuracy(history, station_id, cutoff, period, FAST_WINDOW),
        _accuracy(history, station_id, cutoff, period, FASTEST_WINDOW),
        _accuracy(history, station_id, cutoff, horizon, LONG_WINDOW),
    )


def _wave_score(history: dict, station_id: str, cutoff: datetime, horizon: int, period: int) -> float | None:
    """Precision de largo plazo del periodo si TODAS las condiciones de entrada
    y salida se cumplen; None si no."""
    long_acc, short_acc, fast_acc, fastest_acc, persistence = _period_accuracies(
        history, station_id, cutoff, horizon, period
    )
    if None in (long_acc, short_acc, fast_acc, fastest_acc, persistence):
        return None
    if (
        fastest_acc >= EXIT_FASTEST
        and fast_acc >= EXIT_FAST
        and long_acc >= ENTER
        and short_acc >= ENTER_RECENT
        and long_acc >= persistence + MARGIN
    ):
        return long_acc
    return None


def active_period(history: dict, station_id: str, cutoff: datetime, horizon: int) -> int | None:
    """El periodo candidato que se verifica ahora mismo, o None. Si varios se
    verifican (una onda de 16 tambien repite a 32) gana el mas corto, salvo que
    uno mas largo sea claramente mas preciso (PERIOD_SWITCH_MARGIN)."""
    best, best_score = None, None
    for period in PERIODS:
        score = _wave_score(history, station_id, cutoff, horizon, period)
        if score is not None and (best_score is None or score > best_score + PERIOD_SWITCH_MARGIN):
            best, best_score = period, score
    return best


def wave_active(history: dict, station_id: str, cutoff: datetime, horizon: int) -> bool:
    return active_period(history, station_id, cutoff, horizon) is not None


def _period_was_wave(history: dict, station_id: str, end: datetime, period: int = PERIOD_SLOTS) -> bool:
    """El bloque de PERIOD_BLOCK slots que termina en `end` ya repetia el de
    `period` slots antes (es decir, ese periodo pertenece a la onda)."""
    acc = _accuracy(history, station_id, end, period, PERIOD_BLOCK)
    return acc is not None and acc >= PERIOD_MIN_ACC


def wave_forecast(history: dict, station_id: str, cutoff: datetime, horizon: int) -> float | None:
    """Promedio adaptativo de los periodos previos (16 o 32 slots, el que se
    verifique) para (estacion, horizonte) si la onda esta activa; None -> usar
    el XGBoost estandar.

    Copiar un solo periodo (y[t-P]) copia tambien su ruido; promediar varios
    lo reduce (backtest real con P=16: ~90% -> ~92.5%). Se promedian hasta
    MAX_PERIODS_BY_PERIOD[P] periodos consecutivos hacia atras, y se corta en el
    primero que falte o que no fuera onda (_period_was_wave), asi que justo tras
    el inicio de la onda solo cuenta lo que realmente oscilaba y nunca se
    mezcla con el regimen anterior. El periodo mas reciente (y[t-P]) es siempre
    el primero."""
    if not 1 <= horizon <= MAX_HORIZON:
        return None
    period = active_period(history, station_id, cutoff, horizon)
    if period is None:
        return None
    target = cutoff + timedelta(minutes=SLOT_MINUTES * horizon)
    values = []
    for k in range(1, MAX_PERIODS_BY_PERIOD[period] + 1):
        source = target - timedelta(minutes=SLOT_MINUTES * period * k)
        value = history.get((station_id, source))
        if value is None or (k > 1 and not _period_was_wave(history, station_id, source, period)):
            break
        values.append(float(value))
    if not values:
        return None
    return max(0.0, sum(values) / len(values))
