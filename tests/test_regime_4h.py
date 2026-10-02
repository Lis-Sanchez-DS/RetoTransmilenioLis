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


def test_active_on_a_clean_4h_wave_and_repeats_the_value_16_slots_back():
    data = history(wave(16))
    for horizon in (1, 2, 3, 4):
        assert w4.wave_active(data, "A", CUTOFF, horizon)
        assert abs(w4.wave_forecast(data, "A", CUTOFF, horizon) - wave(16)(horizon - 16)) < 1e-9


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


def test_missing_data_or_out_of_range_horizon_returns_none():
    data = history(wave(16))
    del data[("A", CUTOFF - timedelta(minutes=15 * 5))]
    assert w4.wave_forecast(data, "A", CUTOFF, 2) is None
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
    assert len(stamps) == w4.HISTORY_SLOTS == 32
    assert stamps[0] == CUTOFF and stamps[-1] == CUTOFF - timedelta(minutes=15 * 31)
