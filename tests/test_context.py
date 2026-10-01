import math
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from app import context as context_module
from app.drift import LAGS, RatioTargetModel, _train_station_horizon_model
from app.features import temporal_features
from app.submit_xgboost import predict_cycle_targets

CUTOFF = datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc)


def test_context_flag_is_off_by_default():
    assert context_module.USE_CONTEXT_FEATURES is False


def test_context_features_returns_values_or_nan_per_missing_instant():
    table = {CUTOFF: {"event_intensity": 0.5, "rain_forecast": 1.25, "temperature_forecast": None}}

    present = context_module.context_features(CUTOFF, table)
    missing = context_module.context_features(CUTOFF + timedelta(minutes=15), table)

    assert present[:2] == [0.5, 1.25]
    assert math.isnan(present[2])  # a None field is missing data, not zero
    assert all(math.isnan(v) for v in missing)
    assert all(math.isnan(v) for v in context_module.context_features(CUTOFF, None))


def test_context_features_accept_local_offset_timestamps():
    table = {CUTOFF: {"event_intensity": 0.7, "rain_forecast": 0.0, "temperature_forecast": 9.0}}

    # 05:00 at -05:00 is 10:00 UTC
    assert context_module.context_features("2026-09-18T05:00:00-05:00", table)[0] == 0.7


def test_fetch_context_follows_pagination_and_parses_offsets(monkeypatch):
    pages = [
        {"data": [{"observed_at": "2026-09-08T00:00:00-05:00", "event_intensity": 0.1,
                   "rain_forecast": 0.2, "temperature_forecast": 8.0, "rain_mm": 9.0}],
         "next_cursor": "abc"},
        {"data": [{"observed_at": "2026-09-08T00:15:00-05:00", "event_intensity": 0.3,
                   "rain_forecast": 0.4, "temperature_forecast": 9.0}],
         "next_cursor": None},
    ]
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params.get("cursor"))
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr("app.context.requests.get", fake_get)

    table = context_module.fetch_context("2026-09-08T00:00:00-05:00", "2026-09-08T01:00:00-05:00")

    assert calls == [None, "abc"]
    first = datetime(2026, 9, 8, 5, 0, tzinfo=timezone.utc)
    assert table[first] == {"event_intensity": 0.1, "rain_forecast": 0.2, "temperature_forecast": 8.0}
    assert "rain_mm" not in table[first]  # observed rain would be leakage
    assert len(table) == 2


def test_fetch_context_empty_when_api_has_no_rows(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [], "next_cursor": None, "count": 0}

    monkeypatch.setattr("app.context.requests.get", lambda *a, **k: FakeResponse())

    assert context_module.fetch_context(CUTOFF, CUTOFF) == {}


class _RecordingModel:
    def __init__(self):
        self.rows = []

    def predict(self, rows):
        self.rows.extend(list(r) for r in rows)
        return np.array([100.0] * len(rows))


def _history_for(station):
    return {(station, CUTOFF - timedelta(minutes=15 * (lag - 1))): 100.0 for lag in LAGS}


def _targets(station):
    return [{"station_id": station, "target_at": (CUTOFF + timedelta(minutes=15 * h)).isoformat()} for h in (1, 2, 3, 4)]


def test_chain_features_unchanged_when_flag_is_off():
    model = _RecordingModel()

    predict_cycle_targets(_targets("A"), _history_for("A"), {"A": model}, CUTOFF.isoformat())

    assert {len(row) for row in model.rows} == {len(LAGS) + 4}  # lags + 4 calendar features


def test_chain_appends_context_at_the_target_timestamp_when_flag_is_on(monkeypatch):
    monkeypatch.setattr("app.context.USE_CONTEXT_FEATURES", True)
    target = CUTOFF + timedelta(minutes=15)
    table = {target: {"event_intensity": 0.9, "rain_forecast": 2.0, "temperature_forecast": 11.0}}
    model = _RecordingModel()

    predict_cycle_targets(_targets("A"), _history_for("A"), {"A": model}, CUTOFF.isoformat(), None, table)

    h1, h2 = model.rows[0], model.rows[1]
    assert len(h1) == len(LAGS) + 4 + 3
    assert h1[-3:] == [0.9, 2.0, 11.0]
    assert all(math.isnan(v) for v in h2[-3:])  # no context published for +30min


def test_training_with_context_flag_produces_an_eleven_feature_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.context.USE_CONTEXT_FEATURES", True)
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    slots = 700
    observed_at = pd.date_range("2026-08-01", periods=slots, freq="15min", tz="UTC")
    demand = (100 + 50 * np.sin(2 * np.pi * np.arange(slots) / 96)).round().astype(int)
    frame = pd.DataFrame({"observed_at": observed_at, "demand": demand})
    # context published for only the first half: the rest must stay trainable (NaN, not dropped)
    table = {
        ts.to_pydatetime(): {"event_intensity": 0.0, "rain_forecast": 0.0, "temperature_forecast": 10.0}
        for ts in observed_at[: slots // 2]
    }

    path = _train_station_horizon_model("TEST", 1, frame, table)
    model = joblib.load(path)

    assert isinstance(model, RatioTargetModel)
    assert model.n_features_in_ == len(LAGS) + 4 + 3
    row = [100.0, 100.0, 100.0, 100.0] + temporal_features(observed_at[-1]) + [float("nan")] * 3
    assert np.isfinite(model.predict([row])[0])
