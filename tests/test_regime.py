import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from app import regime

CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class PersistenceModel:
    """XGBoost de mentira: predice lag_1 (columna 0), como una cadena que no ve ninguna estructura."""

    def predict(self, features):
        return np.asarray(features)[:, 0]


def series(fn, noise=0.0, seed=0):
    rng = random.Random(seed)
    times = [CUTOFF + timedelta(minutes=15 * (k - (regime.HISTORY_SLOTS - 1))) for k in range(regime.HISTORY_SLOTS)]
    values = np.array([max(0.0, fn(k - (regime.HISTORY_SLOTS - 1)) * (1 + rng.gauss(0, noise))) for k in range(regime.HISTORY_SLOTS)])
    return values, times


def wave(period, amplitude=0.8, phase=0.0):
    return lambda slot: 1000.0 * (1 + amplitude * math.sin(2 * math.pi * slot / period + phase))


def chain_output(values, times):
    _, now = regime.chain_matrix(PersistenceModel(), values, times)
    return now


def test_picks_up_a_6h_wave_and_beats_the_chain():
    values, times = series(wave(24), noise=0.03)
    chain = chain_output(values, times)
    out = regime.blend_station(values, times, PersistenceModel(), chain)
    truth = {h: wave(24)(h) for h in (1, 2, 3, 4)}
    assert out is not None
    for h in (1, 2, 3, 4):
        assert out[h]["weights"]["M0"] < 0.2
        assert abs(out[h]["value"] - truth[h]) < abs(chain[h] - truth[h])  # mejor que la cadena


def test_handles_a_wave_that_inverted_a_while_ago():
    # onda de 4h cuya fase se invirtio hace 120 slots: tras re-aprender, sigue la onda actual
    shape = lambda slot: wave(16)(slot) if slot < -120 else wave(16, phase=math.pi)(slot)
    values, times = series(shape, noise=0.03)
    out = regime.blend_station(values, times, PersistenceModel(), chain_output(values, times))
    assert out is not None
    truth = {h: wave(16, phase=math.pi)(h) for h in (1, 2, 3, 4)}
    assert abs(out[4]["value"] - truth[4]) < 0.1 * 1800


def test_no_structure_means_exactly_the_chain():
    rng = random.Random(9)
    noise = [rng.uniform(900, 1100) for _ in range(regime.HISTORY_SLOTS)]
    times = [CUTOFF + timedelta(minutes=15 * (k - (regime.HISTORY_SLOTS - 1))) for k in range(regime.HISTORY_SLOTS)]
    values = np.array(noise)
    chain = chain_output(values, times)
    out = regime.blend_station(values, times, PersistenceModel(), chain)
    assert out is not None
    for h in (1, 2, 3, 4):
        assert out[h]["weights"] == {"M0": 1.0}
        assert out[h]["value"] == chain[h]


def test_constant_series_is_left_alone():
    values, times = series(lambda slot: 500.0)
    chain = chain_output(values, times)
    out = regime.blend_station(values, times, PersistenceModel(), chain)
    assert all(out[h]["weights"] == {"M0": 1.0} and out[h]["value"] == chain[h] for h in (1, 2, 3, 4))


def test_apply_generic_layer_modifies_in_place_returns_chain_weights_and_respects_skip():
    values, times = series(wave(24), noise=0.03)
    chain = chain_output(values, times)
    history = {("A", t): v for t, v in zip(times, values)}
    predictions = [
        {"station_id": "A", "target_at": (CUTOFF + timedelta(minutes=15 * h)).isoformat(), "value": chain[h]} for h in (1, 2, 3, 4)
    ]
    claimed = (predictions[0]["station_id"], predictions[0]["target_at"])
    original_first = predictions[0]["value"]
    weights = regime.apply_generic_layer(predictions, history, {"A": PersistenceModel()}, CUTOFF, skip={claimed})
    assert predictions[0]["value"] == original_first and claimed not in weights  # reclamada por la onda de 4h: intacta
    assert all(weights[("A", p["target_at"])] < 0.5 for p in predictions[1:])
    assert predictions[3]["value"] != chain[4]


def test_missing_history_or_model_leaves_predictions_untouched():
    values, times = series(wave(24))
    history = {("A", t): v for t, v in zip(times, values)}
    del history[("A", times[10])]  # un hueco en la historia
    prediction = {"station_id": "A", "target_at": (CUTOFF + timedelta(minutes=15)).isoformat(), "value": 123.0}
    assert regime.apply_generic_layer([dict(prediction)], history, {"A": PersistenceModel()}, CUTOFF) == {
        ("A", prediction["target_at"]): 1.0
    }
    untouched = dict(prediction)
    regime.apply_generic_layer([untouched], history, {"A": PersistenceModel()}, CUTOFF)
    assert untouched["value"] == 123.0
    no_model = dict(prediction)
    assert regime.apply_generic_layer([no_model], {("A", t): v for t, v in zip(times, values)}, {}, CUTOFF)[("A", prediction["target_at"])] == 1.0
    assert no_model["value"] == 123.0


def test_history_range_is_a_fixed_window():
    start, end = regime.history_range(CUTOFF)
    assert end == CUTOFF and (end - start) == timedelta(minutes=15 * (regime.HISTORY_SLOTS - 1))


def test_chain_matrix_matches_the_recursive_chain_used_in_production():
    """chain_matrix debe reproducir exactamente predict_cycle_targets (sin recorte) para el corte actual."""
    from app.drift import LAGS
    from app.submit_xgboost import predict_cycle_targets

    values, times = series(wave(16), noise=0.05, seed=2)

    class LevelModel:
        def predict(self, features):
            f = np.asarray(features)
            return 0.6 * f[:, 0] + 0.3 * f[:, 1] + 0.1 * f[:, 3] + 20 * f[:, 4]

    history = {("A", t): v for t, v in zip(times, values)}
    targets = [{"station_id": "A", "target_at": (CUTOFF + timedelta(minutes=15 * h)).isoformat()} for h in (1, 2, 3, 4)]
    production = {round((datetime.fromisoformat(p["target_at"]) - CUTOFF).total_seconds() / 900): p["value"]
                  for p in predict_cycle_targets(targets, history, {"A": LevelModel()}, CUTOFF.isoformat())}
    _, now = regime.chain_matrix(LevelModel(), values, times)
    for h in (1, 2, 3, 4):
        assert abs(now[h] - production[h]) < 1e-2, (h, now[h], production[h])
    assert LAGS == (1, 2, 4, 96)
