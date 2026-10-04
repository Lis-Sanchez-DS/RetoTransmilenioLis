from datetime import datetime, timedelta, timezone

from app import trend_gate as tg

CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def at(slot):
    return CUTOFF + timedelta(minutes=15 * slot)


def pairs(demands, predictions):
    n = len(demands)
    return [(at(slot - (n - 1)), demands[slot], predictions[slot]) for slot in range(n)]


def history(values_by_slot, station="A"):
    return {(station, at(slot)): value for slot, value in values_by_slot.items()}


FLAT = {slot: 500.0 for slot in range(-30, 1)}


def test_margin_is_persistence_minus_chain_accuracy():
    data = history(FLAT)
    # demanda plana 500: persistencia perfecta; cadena con error de 100 en cada slot -> precision 0.8
    margin = tg.gate_margin(data, "A", pairs([500.0] * 8, [400.0] * 8))
    assert abs(margin - 0.2) < 1e-9


def test_margin_negative_when_chain_beats_persistence():
    data = history({slot: 100.0 * slot for slot in range(-30, 1)})  # sube 100 por slot
    ps = [(at(s), 100.0 * s, 100.0 * s) for s in range(-7, 1)]  # cadena perfecta
    assert tg.gate_margin(data, "A", ps) < 0


def test_margin_none_without_enough_pairs_or_missing_previous_value():
    data = history(FLAT)
    assert tg.gate_margin(data, "A", pairs([500.0] * 7, [400.0] * 7)) is None
    assert tg.gate_margin(data, "B", pairs([500.0] * 8, [400.0] * 8)) is None  # otra estacion: sin historia
    holes = dict(data)
    del holes[("A", at(-3))]
    assert tg.gate_margin(holes, "A", pairs([500.0] * 8, [400.0] * 8)) is None


def test_margin_ignores_rows_without_a_prediction_and_uses_the_latest_window():
    data = history(FLAT)
    rows = [(at(-12), 500.0, None)] + pairs([500.0] * 8, [400.0] * 8)
    assert abs(tg.gate_margin(data, "A", rows) - 0.2) < 1e-9


def test_trend_forecast_extrapolates_damped_and_never_goes_negative():
    data = history({-2: 400.0, -1: 300.0, 0: 200.0})
    # y0 + 0.6 * h * (y0 - y[-2]) / 2 = 200 + 0.6*h*(-200)/2 = 200 - 60h
    assert abs(tg.trend_forecast(data, "A", CUTOFF, 1) - 140.0) < 1e-9
    assert abs(tg.trend_forecast(data, "A", CUTOFF, 4) - 0.0) < 1e-9
    assert tg.trend_forecast(data, "A", CUTOFF, 4) >= 0
    assert tg.trend_forecast(history({0: 5.0}), "A", CUTOFF, 1) is None


def test_blended_value_only_when_margin_reaches_the_threshold():
    data = history({-2: 400.0, -1: 300.0, 0: 200.0})
    assert tg.blended_value(data, "A", CUTOFF, 1, 300.0, None) == 300.0
    assert tg.blended_value(data, "A", CUTOFF, 1, 300.0, tg.GATE_MARGIN - 0.001) == 300.0
    assert abs(tg.blended_value(data, "A", CUTOFF, 1, 300.0, tg.GATE_MARGIN) - 220.0) < 1e-9  # 0.5*140 + 0.5*300
    assert tg.blended_value(history({0: 5.0}), "A", CUTOFF, 1, 300.0, 0.5) == 300.0  # sin tendencia -> igual
