import math
from datetime import datetime, timedelta, timezone

from app import regime_4h as w4

CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def history(series, station="A"):
    """series(slot) con el cutoff en slot 0 y el pasado en slots negativos."""
    return {
        (station, CUTOFF + timedelta(minutes=15 * slot)): series(slot)
        for slot in range(-(w4.HISTORY_SLOTS + 8), 1)
    }


def wave(period, amplitude=800.0):
    return lambda slot: 1000.0 + amplitude * math.sin(2 * math.pi * slot / period)


def test_active_on_a_clean_4h_wave_and_repeats_the_period():
    data = history(wave(16))
    for horizon in (1, 2, 3, 4):
        assert w4.wave_active(data, "A", CUTOFF, horizon)
        # onda limpia: promediar periodos previos da lo mismo que copiar uno
        assert abs(w4.wave_forecast(data, "A", CUTOFF, horizon) - wave(16)(horizon - 16)) < 1e-9


def test_averages_previous_periods_to_cancel_noise():
    import random

    rng = random.Random(7)
    noise = {slot: rng.gauss(0, 60) for slot in range(-(w4.HISTORY_SLOTS + 8), 5)}
    data = history(lambda slot: wave(16)(slot) + noise[slot])
    horizon = 2
    expected_periods = [wave(16)(horizon) + noise[horizon - 16 * k] for k in range(1, w4.MAX_PERIODS_BY_PERIOD[16] + 1)]
    value = w4.wave_forecast(data, "A", CUTOFF, horizon)
    assert abs(value - sum(expected_periods) / len(expected_periods)) < 1e-9
    clean = wave(16)(horizon)
    single_error = abs(expected_periods[0] - clean)
    assert abs(value - clean) < single_error  # este ruido concreto: el promedio acerca mas


def test_stops_averaging_at_the_first_period_that_was_not_wave():
    # la onda empieza hace 40 slots: antes hay un nivel plano distinto
    def onset(slot):
        return wave(16)(slot) if slot > -40 else 700.0

    data = history(onset)
    horizon = 1
    value = w4.wave_forecast(data, "A", CUTOFF, horizon)
    assert value is not None
    # solo cuentan los periodos dentro de la onda: y[-15] y y[-31]; y[-47] es nivel plano
    assert abs(value - wave(16)(horizon)) < 1e-9


def test_missing_old_period_just_shortens_the_average():
    data = history(wave(16))
    for k in range(-80, -60):
        data.pop(("A", CUTOFF + timedelta(minutes=15 * k)), None)
    assert abs(w4.wave_forecast(data, "A", CUTOFF, 1) - wave(16)(1)) < 1e-9


def test_off_for_a_6h_wave_a_flat_series_and_non_periodic_noise():
    import random

    rng = random.Random(1)
    noise = {}
    for slot in range(-(w4.HISTORY_SLOTS + 8), 1):
        noise[slot] = rng.uniform(100, 2000)
    for series in (wave(24), lambda slot: 500.0, lambda slot: noise[slot]):
        data = history(series)
        assert all(w4.wave_forecast(data, "A", CUTOFF, h) is None for h in (1, 2, 3, 4))


def test_switches_off_as_soon_as_the_wave_breaks():
    def breaking(slot):
        # onda de 4h hasta hace 4 slots; luego el patron se rompe (nivel plano distinto)
        return wave(16)(slot) if slot <= -4 else 3000.0

    data = history(breaking)
    assert w4.wave_forecast(data, "A", CUTOFF, 2) is None


def test_off_when_the_wave_is_inverted_or_changes_period():
    inverted = lambda slot: wave(16)(slot) if slot <= -8 else 2000.0 - wave(16)(slot)
    changed = lambda slot: wave(16)(slot) if slot <= -8 else wave(24)(slot)
    for series in (inverted, changed):
        assert w4.wave_forecast(history(series), "A", CUTOFF, 3) is None


def test_a_single_gap_is_tolerated_but_a_missing_latest_slot_or_heavy_gaps_are_not():
    data = history(wave(16))
    del data[("A", CUTOFF - timedelta(minutes=15 * 5))]   # un hueco (quality=missing): sigue activa
    assert w4.wave_forecast(data, "A", CUTOFF, 2) is not None
    no_latest = history(wave(16))
    del no_latest[("A", CUTOFF)]
    assert w4.wave_forecast(no_latest, "A", CUTOFF, 2) is None
    gappy = history(wave(16))
    for slot in range(-9, 0, 1):
        del gappy[("A", CUTOFF + timedelta(minutes=15 * slot))]
    assert w4.wave_forecast(gappy, "A", CUTOFF, 2) is None


def test_missing_data_or_out_of_range_horizon_returns_none():
    assert w4.wave_forecast(history(wave(16)), "A", CUTOFF, 0) is None
    assert w4.wave_forecast(history(wave(16)), "A", CUTOFF, 5) is None
    assert w4.wave_forecast(history(wave(16)), "UNKNOWN", CUTOFF, 1) is None


def test_must_beat_persistence_by_the_margin():
    # Serie casi plana con una pequena oscilacion de 16 slots: lag-16 es muy
    # preciso pero la persistencia tambien, asi que no hay ventaja que justifique el cambio.
    data = history(wave(16, amplitude=2.0))
    assert not w4.wave_active(data, "A", CUTOFF, 1)


def test_history_timestamps_is_a_small_fixed_window():
    stamps = w4.history_timestamps(CUTOFF)
    assert len(stamps) == w4.HISTORY_SLOTS == 119
    assert stamps[0] == CUTOFF and stamps[-1] == CUTOFF - timedelta(minutes=15 * 118)


def test_fast_exit_when_only_the_last_two_slots_break_the_wave():
    # onda limpia de 4h, pero los ultimos 2 slots se desvian de golpe (fin abrupto)
    broken = lambda slot: wave(16)(slot) + (625.0 if slot >= -1 else 0.0)
    data = history(broken)
    assert w4._accuracy(data, "A", CUTOFF, 16, w4.LONG_WINDOW) >= w4.ENTER
    assert w4._accuracy(data, "A", CUTOFF, 16, w4.SHORT_WINDOW) >= w4.ENTER_RECENT
    assert w4._accuracy(data, "A", CUTOFF, 16, w4.FAST_WINDOW) >= w4.EXIT_FAST  # la salida de 4 slots aun no dispara
    assert w4._accuracy(data, "A", CUTOFF, 16, w4.FASTEST_WINDOW) < w4.EXIT_FASTEST
    assert not w4.wave_active(data, "A", CUTOFF, 1)
    assert w4.wave_forecast(data, "A", CUTOFF, 1) is None


def test_active_on_an_8h_wave_and_averages_32_slot_periods():
    # 2026-10-04: la onda paso de 16 a 32 slots; la capa debe seguirla.
    data = history(wave(32))
    for horizon in (1, 2, 3, 4):
        assert w4.active_period(data, "A", CUTOFF, horizon) == 32
        assert abs(w4.wave_forecast(data, "A", CUTOFF, horizon) - wave(32)(horizon - 32)) < 1e-9


def test_16_slot_wave_still_selects_16_not_32():
    data = history(wave(16))
    assert w4.active_period(data, "A", CUTOFF, 1) == 16


def test_inactive_when_neither_period_repeats():
    import random

    rng = random.Random(3)
    data = history(lambda slot: 1000.0 + rng.uniform(-400, 400))
    assert w4.active_period(data, "A", CUTOFF, 1) is None
    assert w4.wave_forecast(data, "A", CUTOFF, 1) is None
