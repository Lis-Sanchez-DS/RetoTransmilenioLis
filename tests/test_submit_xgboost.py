from datetime import datetime, timedelta, timezone

import requests

from app.submit_xgboost import LAGS, predict_cycle_targets, submit_current_cycle


class EchoPlusOneModel:
    """predict() returns lag_1 (the first feature) plus 1, to trace recursion."""

    def predict(self, features):
        return [features[0][0] + 1]


def test_predict_cycle_targets_recurses_lags_forward_per_horizon():
    """Each of a station's 4 horizons must use lags relative to its OWN
    target time, not all relative to data_cutoff. This is the bug fix: the
    +30/+45/+60min horizons need lag_1/2/4 values that aren't observed yet,
    so each prediction must feed back in as the stand-in for the next
    horizon's missing lag. With EchoPlusOneModel, correct recursion produces
    strictly increasing predictions (101, 102, 103, 104); the old bug (lag_1
    always read from data_cutoff regardless of horizon) would produce the
    same value (101) four times.
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Populate history generously so every lag/horizon combination resolves;
    # only the value exactly at cutoff matters for this test's assertion.
    history = {(station, cutoff - timedelta(minutes=m)): 0.0 for m in range(0, 20000, 15)}
    history[(station, cutoff)] = 100.0

    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30, 45, 60)
    ]
    models = {station: EchoPlusOneModel()}

    predictions = predict_cycle_targets(targets, history, models)

    assert [p["value"] for p in predictions] == [101.0, 102.0, 103.0, 104.0]


def test_predict_cycle_targets_keeps_stations_independent():
    """Station A's predictions must not leak into station B's local history."""
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    history = {}
    for station, base in (("A", 100.0), ("B", 500.0)):
        for m in range(0, 20000, 15):
            history[(station, cutoff - timedelta(minutes=m))] = base

    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for station in ("A", "B")
        for m in (15, 30, 45, 60)
    ]
    models = {"A": EchoPlusOneModel(), "B": EchoPlusOneModel()}

    predictions = predict_cycle_targets(targets, history, models)

    a_values = [p["value"] for p in predictions if p["station_id"] == "A"]
    b_values = [p["value"] for p in predictions if p["station_id"] == "B"]
    assert a_values == [101.0, 102.0, 103.0, 104.0]
    assert b_values == [501.0, 502.0, 503.0, 504.0]


def test_predict_cycle_targets_skips_station_with_missing_history():
    """A station missing lag history is skipped, not fatal to the whole cycle.

    One station's incomplete history (e.g. a gap in the week lag_672 needs)
    shouldn't zero out every other station's otherwise-valid submission.
    """
    incomplete_station = "A"
    healthy_station = "B"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    target_ts = cutoff + timedelta(minutes=15)
    targets = [
        {"station_id": incomplete_station, "target_at": target_ts.isoformat()},
        {"station_id": healthy_station, "target_at": target_ts.isoformat()},
    ]
    history = {
        (healthy_station, target_ts - timedelta(minutes=15 * lag)): 500.0 + lag
        for lag in LAGS
    }
    models = {incomplete_station: EchoPlusOneModel(), healthy_station: EchoPlusOneModel()}

    predictions = predict_cycle_targets(targets, history, models)

    assert [p["station_id"] for p in predictions] == [healthy_station]


def test_submit_current_cycle_treats_409_as_already_submitted(monkeypatch):
    """A forecast cycle can still be "current" on the loop's next 5-min tick
    after we already submitted for it. The server rejects the repeat with
    409 (not an idempotent replay), which must be treated as a benign no-op
    - not raised as a real failure that trips the workflow's fail counter.
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    target_ts = cutoff + timedelta(minutes=15)
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [
            {"station_id": station, "target_at": target_ts.isoformat()},
        ],
    }

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            self._rows = [
                (station, (target_ts - timedelta(minutes=15 * lag)).isoformat(), 500.0 + lag)
                for lag in LAGS
            ] if 'FROM "Original Data"' in query else []
            return self

        def fetchall(self):
            return self._rows

    class FakeModel:
        def predict(self, features):
            return [42.0]

    class FakeResponse:
        status_code = 409

        def raise_for_status(self):
            raise requests.HTTPError("409 Client Error: Conflict", response=self)

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)

    result = submit_current_cycle()

    assert result is None
