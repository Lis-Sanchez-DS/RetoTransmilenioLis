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


class ConstantModel:
    """Always predicts the same fixed value regardless of input features -
    useful for pinning down exactly where the alarm-gated band clamp kicks
    in, independent of any recursive chaining behavior."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def predict(self, features):
        self.calls.append(list(features[0]))
        return [self.value]


def _flat_history(station, cutoff, anchor):
    return {
        (station, cutoff - timedelta(minutes=15 * (lag - 1))): anchor
        for lag in LAGS
    }


def test_predict_cycle_targets_clamps_overprediction_during_alarm():
    """A prediction far above the anchor (last real observed value) must be
    clamped to the band's upper edge when this station's alarm is active -
    see PH_ALARM_BAND_PCT (0.5 -> anchor * 1.5)."""
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    anchor = 100.0
    history = _flat_history(station, cutoff, anchor)
    targets = [{"station_id": station, "target_at": (cutoff + timedelta(minutes=15)).isoformat()}]
    model = ConstantModel(500.0)  # 5x anchor: far outside a 50% band

    predictions = predict_cycle_targets(targets, history, {station: model}, cutoff, {station: True})

    assert predictions[0]["value"] == 150.0


def test_predict_cycle_targets_clamps_underprediction_during_alarm():
    """A severe UNDER-prediction must be clamped symmetrically - the band
    is not just a ceiling on over-prediction."""
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    anchor = 100.0
    history = _flat_history(station, cutoff, anchor)
    targets = [{"station_id": station, "target_at": (cutoff + timedelta(minutes=15)).isoformat()}]
    model = ConstantModel(5.0)  # far below anchor: far outside a 50% band

    predictions = predict_cycle_targets(targets, history, {station: model}, cutoff, {station: True})

    assert predictions[0]["value"] == 50.0


def test_predict_cycle_targets_no_clamp_without_alarm():
    """Without a fresh alarm for this station - explicitly False, or the
    station simply missing from alarm_flags, or alarm_flags omitted
    entirely - the raw prediction must pass through untouched, however far
    it is from the anchor."""
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    anchor = 100.0
    history = _flat_history(station, cutoff, anchor)
    targets = [{"station_id": station, "target_at": (cutoff + timedelta(minutes=15)).isoformat()}]
    model = ConstantModel(500.0)

    assert predict_cycle_targets(targets, history, {station: model}, cutoff, {station: False})[0]["value"] == 500.0
    assert predict_cycle_targets(targets, history, {station: model}, cutoff, {})[0]["value"] == 500.0
    assert predict_cycle_targets(targets, history, {station: model}, cutoff)[0]["value"] == 500.0


def test_predict_cycle_targets_clamped_value_propagates_through_chain():
    """The clamped value, not the raw model output, must be what feeds a
    later horizon's lag_1/lag_2 slot - otherwise the chain would keep
    compounding the extreme raw prediction internally even while emitting
    the clamped number for the earlier horizon."""
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    anchor = 100.0
    history = _flat_history(station, cutoff, anchor)
    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30)
    ]
    model = ConstantModel(500.0)

    predictions = predict_cycle_targets(targets, history, {station: model}, cutoff, {station: True})

    assert predictions[0]["value"] == 150.0  # h1 clamped to the band's upper edge
    assert model.calls[1][0] == 150.0  # h2's lag_1 slot fed the CLAMPED h1 value, not raw 500.0


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
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)

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
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)

    submit_current_cycle()

    assert captured_payload["predictions"][0]["value"] == 32.0


def test_submit_current_cycle_clamps_prediction_when_station_alarm_is_fresh(monkeypatch):
    """End-to-end: submit_current_cycle must actually fetch this station's
    fresh-alarm state (_station_has_fresh_page_hinkley_alarm) and pass it
    through to predict_cycle_targets, so a raw prediction far from the
    anchor really does get clamped in the submitted payload - not just in
    the unit-level predict_cycle_targets tests above."""
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    target_ts = cutoff + timedelta(minutes=15)
    anchor = 100.0
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [{"station_id": station, "target_at": target_ts.isoformat()}],
    }

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            self._rows = [
                (station, (target_ts - timedelta(minutes=15 * lag)).isoformat(), anchor)
                for lag in LAGS
            ] if 'FROM "Original Data"' in query else []
            return self

        def fetchall(self):
            return self._rows

    class FakeModel:
        def predict(self, features):
            return [500.0]  # 5x anchor: far outside a 50% band

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

    alarm_calls = []

    def fake_alarm(station_id):
        alarm_calls.append(station_id)
        return True

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", fake_post)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", lambda station_id: (0.0, 0.0))
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", fake_alarm)

    submit_current_cycle()

    assert alarm_calls == [station]
    assert captured_payload["predictions"][0]["value"] == 150.0  # anchor * 1.5, clamped


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
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)

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
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)

    try:
        submit_current_cycle()
    except RuntimeError:
        pass  # expected: FakeConnection returns no rows, so no station has enough history

    queries = [(q, p) for q, p in captured if "station_id = ANY" in q]
    assert len(queries) == 2  # "Original Data" + "Temp"
    expected_timestamps = set(cutoff - timedelta(minutes=15 * (lag - 1)) for lag in LAGS)
    # La onda de 4h (USE_4H_WAVE) suma una ventana fija de 32 slots: sigue siendo
    # una lista acotada de timestamps exactos, nunca la tabla completa.
    expected_timestamps |= {cutoff - timedelta(minutes=15 * k) for k in range(32)}
    for _, params in queries:
        station_ids, timestamps = params
        assert station_ids == [station]
        assert sorted(timestamps) == sorted(expected_timestamps)
        assert len(timestamps) == 33  # ventana de 32 slots + lag_96


def test_submit_current_cycle_queries_only_lag_timestamps_when_wave_is_off(monkeypatch):
    """Con USE_4H_WAVE apagada la consulta vuelve a ser exactamente la de antes."""
    monkeypatch.setattr("app.submit_xgboost.USE_4H_WAVE", False)
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [{"station_id": station, "target_at": (cutoff + timedelta(minutes=15)).isoformat()}],
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
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)
    try:
        submit_current_cycle()
    except RuntimeError:
        pass
    queries = [(q, p) for q, p in captured if "station_id = ANY" in q]
    for _, params in queries:
        assert sorted(params[1]) == sorted(cutoff - timedelta(minutes=15 * (lag - 1)) for lag in LAGS)


def _run_submit_with_history(monkeypatch, station, cutoff, series, horizons=(1, 2, 3, 4), bias=(-10.0, -10.0)):
    """Corre submit_current_cycle con una historia sintetica `series` (funcion
    de numero de slot -> demanda; el cutoff es el slot 0) y devuelve
    (payload, veces que se leyo el sesgo EWMA)."""
    cycle = {
        "cycle_id": "cycle-1",
        "data_cutoff": cutoff.isoformat(),
        "targets": [
            {"station_id": station, "target_at": (cutoff + timedelta(minutes=15 * h)).isoformat()} for h in horizons
        ],
    }
    bias_calls = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            if 'FROM "Original Data"' in query and params is not None:
                self._rows = [(station, ts.isoformat(), series(round((ts - cutoff).total_seconds() / 900))) for ts in params[1]]
            else:
                self._rows = []
            return self

        def fetchall(self):
            return self._rows

    class FakeModel:
        def predict(self, features):
            return [100.0]

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"submission_id": "s", "status": "accepted", "predictions_received": len(horizons), "expected_predictions": len(horizons)}

    def fake_post(url, headers=None, json=None, timeout=None):
        if "/submissions" in url:
            captured.update(json)
        return FakeResponse()

    def fake_bias(station_id):
        bias_calls.append(station_id)
        return bias

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", fake_post)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", fake_bias)
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)
    submit_current_cycle()
    return captured, bias_calls


def test_submit_uses_4h_wave_and_skips_ewma_when_wave_is_intact(monkeypatch):
    import math

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    wave = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 16)  # periodo exacto de 16 slots
    payload, bias_calls = _run_submit_with_history(monkeypatch, "A", cutoff, wave)

    by_target = {p["target_at"]: p["value"] for p in payload["predictions"]}
    for h in (1, 2, 3, 4):
        expected = wave(h - 16)  # y[objetivo - 16]
        assert abs(by_target[(cutoff + timedelta(minutes=15 * h)).isoformat()] - expected) < 0.01
    assert bias_calls == []  # EWMA no se aplica a valores que ya no vienen de la cadena


def test_submit_falls_back_to_standard_xgboost_when_wave_is_not_4h(monkeypatch):
    import math

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    six_hour_wave = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 24)  # 6h, no 4h
    payload, bias_calls = _run_submit_with_history(monkeypatch, "A", cutoff, six_hour_wave)

    # Chain (model=100) + EWMA bias (-10) = 90 en los cuatro horizontes: nada cambio.
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]
    assert bias_calls == ["A"]


def test_submit_wave_layer_error_never_blocks_the_submission(monkeypatch):
    import math

    def boom(*args, **kwargs):
        raise RuntimeError("fallo inesperado")

    monkeypatch.setattr("app.submit_xgboost.wave_module.wave_forecast", boom)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    payload, _ = _run_submit_with_history(
        monkeypatch, "A", cutoff, lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 16)
    )
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]
