from datetime import datetime, timedelta, timezone

import requests

from app.submit_xgboost import LAGS, predict_cycle_targets, submit_current_cycle


class RecordingModel:
    """Records the exact feature row of every call and predicts lag_1 + 1,
    so a chain of calls is traceable: each output feeds the next call's
    lag_1 (or lag_2) slot, making the recursive substitution visible in
    `self.calls` rather than just in the final numbers.
    """

    def __init__(self):
        self.calls = []

    def predict(self, features):
        row = list(features[0])
        self.calls.append(row)
        return [row[0] + 1]


def test_predict_cycle_targets_chains_h1_prediction_recursively():
    """h1 uses only real lags. h2's lag_1 slot must be h1's own prediction
    (never a real value). h3's lag_1 slot must be h2's prediction and its
    lag_2 slot must be h1's prediction. h4 continues the pattern. lag_4 and
    lag_96 stay real at every horizon (LAGS = (1, 2, 4, 96), and no horizon
    here exceeds 4, so k=4 and k=96 always satisfy k >= h).
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    real_lags = {1: 100.0, 2: 200.0, 4: 400.0, 96: 9600.0}
    history = {
        (station, cutoff - timedelta(minutes=15 * (lag - 1))): value
        for lag, value in real_lags.items()
    }
    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30, 45, 60)
    ]
    model = RecordingModel()
    models = {station: model}

    predictions = predict_cycle_targets(targets, history, models, cutoff)

    assert [p["value"] for p in predictions] == [101.0, 102.0, 103.0, 104.0]
    # Feature order matches LAGS = (1, 2, 4, 96): [lag_1_slot, lag_2_slot, lag_4_slot, lag_96_slot].
    assert model.calls[0][:4] == [100.0, 200.0, 400.0, 9600.0]  # h1: fully real
    assert model.calls[1][:4] == [101.0, 200.0, 400.0, 9600.0]  # h2: lag_1 <- h1's prediction
    assert model.calls[2][:4] == [102.0, 101.0, 400.0, 9600.0]  # h3: lag_1<-h2, lag_2<-h1
    assert model.calls[3][:4] == [103.0, 102.0, 400.0, 9600.0]  # h4: lag_1<-h3, lag_2<-h2


def test_predict_cycle_targets_keeps_stations_independent():
    """Station A's recursive chain must not leak into station B's."""
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    history = {}
    for station, base in (("A", 100.0), ("B", 500.0)):
        for lag in LAGS:
            history[(station, cutoff - timedelta(minutes=15 * (lag - 1)))] = base

    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for station in ("A", "B")
        for m in (15, 30, 45, 60)
    ]
    models = {"A": RecordingModel(), "B": RecordingModel()}

    predictions = predict_cycle_targets(targets, history, models, cutoff)

    a_values = [p["value"] for p in predictions if p["station_id"] == "A"]
    b_values = [p["value"] for p in predictions if p["station_id"] == "B"]
    assert a_values == [101.0, 102.0, 103.0, 104.0]
    assert b_values == [501.0, 502.0, 503.0, 504.0]


def test_predict_cycle_targets_skips_station_with_missing_history():
    """A station missing any real lag is skipped ENTIRELY (all its targets),
    not just one horizon - unlike the old direct per-horizon design, every
    horizon here ultimately depends on the same real lags, so a gap blocks
    the whole recursive chain, not one link of it. One station's incomplete
    history still must not zero out another station's otherwise-valid targets.
    """
    incomplete_station = "A"
    healthy_station = "B"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    targets = [
        {"station_id": incomplete_station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30)
    ] + [
        {"station_id": healthy_station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30)
    ]
    history = {
        (healthy_station, cutoff - timedelta(minutes=15 * (lag - 1))): 500.0 + lag
        for lag in LAGS
    }
    models = {incomplete_station: RecordingModel(), healthy_station: RecordingModel()}

    predictions = predict_cycle_targets(targets, history, models, cutoff)

    assert {p["station_id"] for p in predictions} == {healthy_station}
    assert len([p for p in predictions if p["station_id"] == healthy_station]) == 2


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

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", lambda station_id: (0.0, 0.0))

    result = submit_current_cycle()

    assert result is None


def test_submit_current_cycle_applies_ewma_bias_correction(monkeypatch):
    """The EWMA correction (station_bias_components) must land on the actual
    submitted values, on top of the raw model output - not just be computed
    and discarded."""
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

    captured_payload = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"submission_id": "sub-1", "status": "accepted", "predictions_received": 1, "expected_predictions": 1}

    def fake_post(url, headers=None, json=None, timeout=None):
        if "/submissions" in url:
            captured_payload.update(json)
        return FakeResponse()

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr(
        "app.submit_xgboost.api_get",
        lambda path: cycle if "current" in path else {"display_name": "x"},
    )
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", fake_post)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", lambda station_id: (-10.0, -10.0))

    submit_current_cycle()

    assert captured_payload["predictions"][0]["value"] == 32.0


def test_submit_current_cycle_applies_horizon_decayed_bias(monkeypatch):
    """When regular_bias != boosted_bias (i.e. Page-Hinkley is actively
    boosted for this station), horizon 1 (+15min) must get the FULL boosted
    correction and the longest horizon (+60min) must get the FULL regular
    correction (EWMA_DECAY_POWER's weight is exactly 1.0 / 0.0 at the two
    ends) - and station_bias_components must be called only ONCE for the
    station even though this cycle has all 4 of its horizons, preserving the
    "one DB read per station per submission" invariant regardless of horizon
    count.
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [
            {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
            for m in (15, 30, 45, 60)
        ],
    }

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            self._rows = [
                (station, (cutoff - timedelta(minutes=15 * (lag - 1))).isoformat(), 500.0 + lag)
                for lag in LAGS
            ] if 'FROM "Original Data"' in query else []
            return self

        def fetchall(self):
            return self._rows

    class FakeModel:
        def predict(self, features):
            return [42.0]

    captured_payload = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"submission_id": "sub-1", "status": "accepted", "predictions_received": 4, "expected_predictions": 4}

    def fake_post(url, headers=None, json=None, timeout=None):
        if "/submissions" in url:
            captured_payload.update(json)
        return FakeResponse()

    bias_calls = []

    def fake_bias_components(station_id):
        bias_calls.append(station_id)
        return (0.0, -100.0)  # regular=0.0, boosted=-100.0 - maximally distinguishable

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr(
        "app.submit_xgboost.api_get",
        lambda path: cycle if "current" in path else {"display_name": "x"},
    )
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", fake_post)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", fake_bias_components)

    submit_current_cycle()

    values = [p["value"] for p in captured_payload["predictions"]]
    assert values[0] == 0.0  # h1 (+15min): raw 42.0 + full boosted bias (-100.0) -> clamped to 0.0
    assert values[3] == 42.0  # h4 (+60min): raw 42.0 + full regular bias (0.0) -> unchanged
    assert bias_calls == ["A"]  # exactly one DB read for the whole station, all 4 horizons


def test_submit_current_cycle_queries_only_needed_lag_timestamps(monkeypatch):
    """submit_current_cycle() must ask the DB for exactly the lag timestamps
    this cycle's targets can need, not the entire "Original Data"/"Temp"
    tables - that unbounded read (tens of thousands of rows, every ~5min)
    was the single largest source of Supabase DB egress. Locks in the query
    params actually sent.
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    target_ts = cutoff + timedelta(minutes=15)
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [{"station_id": station, "target_at": target_ts.isoformat()}],
    }
    captured = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            captured.append((query, params))
            self._rows = []
            return self

        def fetchall(self):
            return self._rows

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: object())

    try:
        submit_current_cycle()
    except RuntimeError:
        pass  # expected: FakeConnection returns no rows, so no station has enough history

    queries = [(q, p) for q, p in captured if "station_id = ANY" in q]
    assert len(queries) == 2  # "Original Data" + "Temp"
    expected_timestamps = sorted(cutoff - timedelta(minutes=15 * (lag - 1)) for lag in LAGS)
    for _, params in queries:
        station_ids, timestamps = params
        assert station_ids == [station]
        assert sorted(timestamps) == expected_timestamps
