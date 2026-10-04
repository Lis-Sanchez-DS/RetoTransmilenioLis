from datetime import datetime, timedelta, timezone

import pytest
import requests

from app.submit_xgboost import LAGS, lag_timestamps, predict_cycle_targets, submit_current_cycle


@pytest.fixture(autouse=True)
def _isolated_marker(tmp_path, monkeypatch):
    """El marcador del ultimo ciclo enviado vive en disco: cada prueba usa el suyo."""
    monkeypatch.setattr("app.submit_xgboost.LAST_SUBMITTED_MARKER", str(tmp_path / ".last_submitted_cycle"))
    # La capa general viene apagada por defecto; las pruebas la ejercitan encendida
    # (las que prueban otra cosa la apagan explicitamente despues).
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", True)
    # Nada de red ni de ledger real en las pruebas: la red de respaldo (stream) no
    # responde, el ledger no existe y registrar una entrega no hace nada.
    monkeypatch.setattr("app.submit_xgboost._latest_values_stream", lambda station_ids, cutoff: {})
    monkeypatch.setattr("app.submit_xgboost.last_submission_cycle", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.record_submission", lambda *args, **kwargs: None)
    # La compuerta de tendencia lee la DB: por defecto sin pares guardados (inactiva).
    monkeypatch.setattr("app.submit_xgboost.station_recent_pairs", lambda station_id, cutoff, limit: [])


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


def _full_history(station, cutoff, value_at):
    """Todos los timestamps reales que la cadena puede leer, con value_at(offset en slots, <= 0)."""
    return {(station, ts): value_at(round((ts - cutoff).total_seconds() / 900)) for ts in lag_timestamps(cutoff)}


def test_predict_cycle_targets_chains_h1_prediction_recursively():
    """lag_k para el objetivo del horizonte h es la demanda k*15 min ANTES DEL
    OBJETIVO (como entreno el modelo h1, donde lag_k = y.shift(k)). Con k >= h es
    el valor real en cutoff + 15*(h-k); con k < h es la prediccion propia de la
    cadena para el horizonte (h-k). Con y(offset) = 1000 + offset cada lag real es
    distinguible, asi que esta prueba tambien fija el anclaje al OBJETIVO
    (corregido el 2026-10-02: antes se leia anclado al corte, h-1 pasos mas viejo).
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    history = _full_history(station, cutoff, lambda offset: 1000.0 + offset)  # y[c]=1000, y[c-1]=999, ...
    targets = [
        {"station_id": station, "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30, 45, 60)
    ]
    model = RecordingModel()
    models = {station: model}

    predictions = predict_cycle_targets(targets, history, models, cutoff)

    assert [p["value"] for p in predictions] == [1001.0, 1002.0, 1003.0, 1004.0]
    # Feature order matches LAGS = (1, 2, 4, 96): [lag_1_slot, lag_2_slot, lag_4_slot, lag_96_slot].
    assert model.calls[0][:4] == [1000.0, 999.0, 997.0, 905.0]  # h1: y[c], y[c-1], y[c-3], y[c-95]
    assert model.calls[1][:4] == [1001.0, 1000.0, 998.0, 906.0]  # h2: lag_1<-h1 pred; lag_2=y[c], lag_4=y[c-2], lag_96=y[c-94]
    assert model.calls[2][:4] == [1002.0, 1001.0, 999.0, 907.0]  # h3: lag_1<-h2, lag_2<-h1; lag_4=y[c-1], lag_96=y[c-93]
    assert model.calls[3][:4] == [1003.0, 1002.0, 1000.0, 908.0]  # h4: lag_1<-h3, lag_2<-h2; lag_4=y[c], lag_96=y[c-92]


def test_predict_cycle_targets_keeps_stations_independent():
    """Station A's recursive chain must not leak into station B's."""
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    history = {}
    for station, base in (("A", 100.0), ("B", 500.0)):
        history.update(_full_history(station, cutoff, lambda offset, base=base: base))

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


def test_predict_cycle_targets_skips_station_without_any_history():
    """A station with no history at all is skipped, and that must not zero out
    another station's otherwise-valid targets."""
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
    history = _full_history(healthy_station, cutoff, lambda offset: 500.0 + offset)
    models = {incomplete_station: RecordingModel(), healthy_station: RecordingModel()}

    predictions = predict_cycle_targets(targets, history, models, cutoff)

    assert {p["station_id"] for p in predictions} == {healthy_station}
    assert len([p for p in predictions if p["station_id"] == healthy_station]) == 2


def test_predict_cycle_targets_fills_gap_so_all_targets_are_sent():
    """A quality=missing hole in one lag must not drop the station: the server
    500s on fewer than the 48 expected targets."""
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    targets = [
        {"station_id": "A", "target_at": (cutoff + timedelta(minutes=m)).isoformat()}
        for m in (15, 30, 45, 60)
    ]
    history = _full_history("A", cutoff, lambda offset: 100.0 + offset)
    del history[("A", cutoff - timedelta(minutes=15))]

    predictions = predict_cycle_targets(targets, history, {"A": RecordingModel()}, cutoff)

    assert len(predictions) == 4


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
    return _full_history(station, cutoff, lambda offset: anchor)


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
                (station, ts.isoformat(), 500.0)
                for ts in lag_timestamps(cutoff)
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

    queries = [(q, p) for q, p in captured if "observed_at = ANY" in q]
    assert len(queries) == 2  # "Original Data" + "Temp"
    expected_timestamps = set(lag_timestamps(cutoff))
    # La onda de 4h (USE_4H_WAVE) suma una ventana fija de 119 slots (periodos de 16 o 32
    # slots, promediando varios en el de 16): sigue siendo una lista acotada de timestamps exactos,
    # nunca la tabla completa.
    expected_timestamps |= {cutoff - timedelta(minutes=15 * k) for k in range(119)}
    for _, params in queries:
        station_ids, timestamps = params
        assert station_ids == [station]
        assert sorted(timestamps) == sorted(expected_timestamps)
        assert len(timestamps) == len(expected_timestamps)


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
    queries = [(q, p) for q, p in captured if "observed_at = ANY" in q]
    for _, params in queries:
        assert sorted(params[1]) == lag_timestamps(cutoff)


def _run_submit_with_history(
    monkeypatch, station, cutoff, series, horizons=(1, 2, 3, 4), bias=(-10.0, -10.0), post_handler=None, model=None
):
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
                if len(params) == 6:  # consulta por rango de la capa general: (ids, inicio, fin, ids, inicio, fin)
                    start, end = params[1], params[2]
                    stamps = []
                    ts = start
                    while ts <= end:
                        stamps.append(ts)
                        ts += timedelta(minutes=15)
                else:  # consulta por timestamps exactos
                    stamps = params[1]
                self._rows = [(station, ts.isoformat(), series(round((ts - cutoff).total_seconds() / 900))) for ts in stamps]
            else:
                self._rows = []
            self.queries.append(query)
            return self

        queries = []

        def fetchall(self):
            return self._rows

    class FakeModel:
        def predict(self, features):
            return [100.0] * len(features)

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"submission_id": "s", "status": "accepted", "predictions_received": len(horizons), "expected_predictions": len(horizons)}

    def fake_post(url, headers=None, json=None, timeout=None):
        if post_handler is not None:
            return post_handler(url, headers, json)
        if "/submissions" in url:
            captured.update(json)
        return FakeResponse()

    def fake_bias(station_id):
        bias_calls.append(station_id)
        return bias

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: {"collected": 0, "inserted": 0, "pages": 1, "cursor": None})
    monkeypatch.setattr("app.submit_xgboost.api_get", lambda path: cycle if "current" in path else {"display_name": "x"})
    monkeypatch.setattr("app.submit_xgboost.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.submit_xgboost.joblib.load", lambda path: model if model is not None else FakeModel())
    monkeypatch.setattr("app.submit_xgboost.os.path.getmtime", lambda path: 0)
    monkeypatch.setattr("app.submit_xgboost.check_collector_heartbeat", lambda: None)
    monkeypatch.setattr("app.submit_xgboost.requests.post", fake_post)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "test-key", raising=False)
    monkeypatch.setattr("app.submit_xgboost.station_bias_components", fake_bias)
    monkeypatch.setattr("app.submit_xgboost._station_has_fresh_page_hinkley_alarm", lambda station_id: False)
    FakeConnection.queries = []
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

    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)  # aisla el respaldo de 4h

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    six_hour_wave = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 24)  # 6h, no 4h
    payload, bias_calls = _run_submit_with_history(monkeypatch, "A", cutoff, six_hour_wave)

    # Chain (model=100) + EWMA bias (-10) = 90 en los cuatro horizontes: nada cambio.
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]
    assert bias_calls == ["A"]


def test_submit_wave_layer_error_never_blocks_the_submission(monkeypatch):
    import math

    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo inesperado")

    monkeypatch.setattr("app.submit_xgboost.wave_module.wave_forecast", boom)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    payload, _ = _run_submit_with_history(
        monkeypatch, "A", cutoff, lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 16)
    )
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]


def test_generic_layer_follows_a_6h_wave_the_4h_backup_ignores(monkeypatch):
    """Un patron parecido pero no identico (periodo de 6h): el respaldo de 4h no
    se activa, la capa general si, y el sesgo EWMA solo cuenta por el peso de la
    cadena (casi cero aqui): el valor enviado sigue a la onda, no al 100 + (-10)
    del XGBoost de mentira."""
    import math

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    six_hour_wave = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 24)
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, six_hour_wave)

    by_target = {p["target_at"]: p["value"] for p in payload["predictions"]}
    for h in (1, 2, 3, 4):
        expected = six_hour_wave(h)  # la onda continua: lo que realmente pasara
        submitted = by_target[(cutoff + timedelta(minutes=15 * h)).isoformat()]
        assert abs(submitted - expected) < 0.05 * 1800, (h, submitted, expected)  # dentro de 5% de la amplitud pico a pico
        assert submitted > 300  # muy lejos del 90 que enviaria el XGBoost de mentira


def test_generic_layer_leaves_standard_pipeline_alone_without_periodic_structure(monkeypatch):
    import random

    rng = random.Random(3)
    noise = {slot: rng.uniform(900, 1100) for slot in range(-400, 10)}
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    payload, bias_calls = _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: noise[slot])
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]  # cadena 100 + sesgo -10, intacto
    assert bias_calls == ["A"]


def test_generic_layer_error_restores_standard_values_and_still_submits(monkeypatch):
    import math

    def corrupt_then_fail(predictions, *args, **kwargs):
        for prediction in predictions:
            prediction["value"] = -12345.0  # modifica en su lugar y luego falla
        raise RuntimeError("fallo a mitad de camino")

    monkeypatch.setattr("app.submit_xgboost.regime_module.apply_generic_layer", corrupt_then_fail)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    payload, _ = _run_submit_with_history(
        monkeypatch, "A", cutoff, lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 24)
    )
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]


def test_generic_layer_is_skipped_for_predictions_the_4h_backup_claimed(monkeypatch):
    """Cuando la onda de 4h reclama todo, la capa general ni siquiera consulta la historia larga."""
    import math

    seen_queries = []
    real_fetch = __import__("app.submit_xgboost", fromlist=["x"])._fetch_generic_history

    def spy(*args, **kwargs):
        seen_queries.append(args)
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr("app.submit_xgboost._fetch_generic_history", spy)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 16))
    assert seen_queries == []


def test_second_loop_iteration_for_the_same_cycle_does_no_heavy_work(monkeypatch):
    import math

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    series = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 24)
    _run_submit_with_history(monkeypatch, "A", cutoff, series)  # primera vuelta: envia y deja el marcador
    calls = {"history": 0}
    monkeypatch.setattr("app.submit_xgboost._fetch_generic_history", lambda *a, **k: calls.__setitem__("history", calls["history"] + 1) or {})
    payload, bias_calls = _run_submit_with_history(monkeypatch, "A", cutoff, series)  # segunda vuelta, mismo ciclo
    assert payload == {} and bias_calls == [] and calls["history"] == 0


def _pairs_off_by(cutoff, demand, prediction, n=8):
    return [(cutoff - timedelta(minutes=15 * k), demand, prediction) for k in range(n - 1, -1, -1)]


def test_trend_gate_blends_with_trend_when_persistence_beats_the_chain(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    seen = []

    def pairs(station_id, at, limit):
        seen.append((station_id, at, limit))
        return _pairs_off_by(cutoff, 500.0, 100.0)  # la cadena viene fallando por mucho

    monkeypatch.setattr("app.submit_xgboost.station_recent_pairs", pairs)
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: 500.0)
    # cadena 100 + EWMA -10 = 90; tendencia con serie plana = 500 -> 0.5*500 + 0.5*90
    assert [p["value"] for p in payload["predictions"]] == [295.0, 295.0, 295.0, 295.0]
    assert seen == [("A", cutoff, 8)]  # una sola lectura por estacion


def test_trend_gate_leaves_values_alone_when_the_chain_is_fine(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.submit_xgboost.station_recent_pairs", lambda station_id, at, limit: _pairs_off_by(cutoff, 500.0, 500.0)
    )
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: 500.0)
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]


def test_trend_gate_does_not_touch_predictions_the_4h_wave_claimed(monkeypatch):
    import math

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    wave = lambda slot: 1000.0 + 800.0 * math.sin(2 * math.pi * slot / 16)
    calls = []
    monkeypatch.setattr(
        "app.submit_xgboost.station_recent_pairs",
        lambda station_id, at, limit: calls.append(station_id) or _pairs_off_by(cutoff, 500.0, 100.0),
    )
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, wave)
    by_target = {p["target_at"]: p["value"] for p in payload["predictions"]}
    for h in (1, 2, 3, 4):
        assert abs(by_target[(cutoff + timedelta(minutes=15 * h)).isoformat()] - wave(h - 16)) < 0.01
    assert calls == []


def test_trend_gate_error_never_blocks_the_submission(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def boom(*args, **kwargs):
        raise RuntimeError("db caida")

    monkeypatch.setattr("app.submit_xgboost.station_recent_pairs", boom)
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: 500.0)
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]


def test_trend_gate_flag_off_restores_previous_behavior(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.USE_TREND_GATE", False)
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "app.submit_xgboost.station_recent_pairs", lambda station_id, at, limit: _pairs_off_by(cutoff, 500.0, 100.0)
    )
    payload, _ = _run_submit_with_history(monkeypatch, "A", cutoff, lambda slot: 500.0)
    assert [p["value"] for p in payload["predictions"]] == [90.0, 90.0, 90.0, 90.0]


# --- Reliability (2026-10-04 audit): never lose a cycle ---------------------------------


class Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = text or str(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code}")
            error.response = self
            raise error


OK_BODY = {"submission_id": "sub_1", "status": "accepted", "predictions_received": 4, "expected_predictions": 4}
CUT = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
FLAT = lambda slot: 500.0


def _posts(responses):
    """post_handler that replays `responses` in order and records (headers, json)."""
    seen = []

    def handler(url, headers, payload):
        seen.append((headers, payload))
        return responses[min(len(seen) - 1, len(responses) - 1)]

    handler.seen = seen
    return handler


def _persistence_check(payload, expected=500.0):
    assert [p["value"] for p in payload["predictions"]] == [expected] * 4


def test_current_cycle_is_fetched_first_and_an_outage_exits_75_without_other_calls(monkeypatch):
    from app.net import UpstreamUnavailable

    calls = []

    def api(path):
        calls.append(path)
        raise UpstreamUnavailable("servidor caido")

    monkeypatch.setattr("app.submit_xgboost.api_get", api)
    monkeypatch.setattr("app.submit_xgboost.API_KEY", "k", raising=False)
    monkeypatch.setattr("app.submit_xgboost.collect_new_data", lambda: calls.append("collect"))
    with pytest.raises(SystemExit) as exit_info:
        from app.submit_xgboost import main

        main()
    assert exit_info.value.code == 75
    assert calls == ["/v1/forecast-cycles/current"]  # ni stream ni /v1/me antes de saber si hay servidor


def test_stack_failure_degrades_to_persistence_instead_of_losing_the_cycle(monkeypatch):
    class Boom:
        def predict(self, features):
            raise RuntimeError("modelo corrupto")

    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr(
        "app.submit_xgboost._latest_values_db", lambda ids, cutoff: {station: 321.0 for station in ids}
    )
    handler = _posts([Resp(201, OK_BODY)])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler, model=Boom())
    _persistence_check(handler.seen[0][1], 321.0)  # persistencia (ultimo dato de la BD), no un fallo


def test_nan_model_output_is_never_submitted_as_zero(monkeypatch):
    class NaNModel:
        def predict(self, features):
            return [float("nan")] * len(features)

    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost._latest_values_db", lambda ids, cutoff: {s: 321.0 for s in ids})
    handler = _posts([Resp(201, OK_BODY)])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler, model=NaNModel())
    _persistence_check(handler.seen[0][1], 321.0)


def test_one_broken_station_does_not_take_down_the_others():
    class Good:
        def predict(self, features):
            return [100.0]

    class Boom:
        def predict(self, features):
            raise ValueError("shape mismatch")

    cutoff = CUT
    targets = [
        {"station_id": s, "target_at": (cutoff + timedelta(minutes=15 * h)).isoformat()}
        for s in ("A", "B")
        for h in (1, 2, 3, 4)
    ]
    history = {**_full_history("A", cutoff, lambda o: 500.0), **_full_history("B", cutoff, lambda o: 500.0)}
    result = predict_cycle_targets(targets, history, {"A": Good(), "B": Boom()}, cutoff.isoformat())
    assert {p["station_id"] for p in result} == {"A"} and len(result) == 4


def test_complete_predictions_sanitizes_fills_and_drops_extras(monkeypatch):
    from app.submit_xgboost import _complete_predictions

    monkeypatch.setattr("app.submit_xgboost._latest_values_db", lambda ids, cutoff: {s: 77.0 for s in ids})
    cycle = {
        "data_cutoff": CUT.isoformat(),
        "targets": [{"station_id": "A", "target_at": f"t{i}"} for i in range(1, 6)],
    }
    raw = [
        {"station_id": "A", "target_at": "t1", "value": 10.0},
        {"station_id": "A", "target_at": "t2", "value": float("nan")},
        {"station_id": "A", "target_at": "t3", "value": float("inf")},
        {"station_id": "A", "target_at": "t4", "value": -5.0},
        {"station_id": "A", "target_at": "extra", "value": 1.0},
    ]
    out = _complete_predictions(cycle, raw)
    assert [p["target_at"] for p in out] == ["t1", "t2", "t3", "t4", "t5"]  # solo y todos los targets
    assert [p["value"] for p in out] == [10.0, 77.0, 77.0, 0.0, 77.0]


def test_complete_predictions_raises_only_when_there_is_no_data_anywhere(monkeypatch):
    from app.submit_xgboost import _complete_predictions

    monkeypatch.setattr("app.submit_xgboost._latest_values_db", lambda ids, cutoff: {})
    cycle = {"data_cutoff": CUT.isoformat(), "targets": [{"station_id": "A", "target_at": "t1"}]}
    with pytest.raises(RuntimeError, match="Sin ningun dato"):
        _complete_predictions(cycle, [])


def test_persistence_falls_back_to_the_stream_when_the_database_is_down(monkeypatch):
    from app.submit_xgboost import _latest_values

    def db_down(ids, cutoff):
        raise RuntimeError("supabase 402")

    monkeypatch.setattr("app.submit_xgboost._latest_values_db", db_down)
    monkeypatch.setattr("app.submit_xgboost._latest_values_stream", lambda ids, cutoff: {s: 9.0 for s in ids})
    assert _latest_values(["A", "B"], CUT) == {"A": 9.0, "B": 9.0}


def test_stream_fallback_builds_a_recent_cursor_and_keeps_the_latest_value_at_or_before_cutoff(monkeypatch):
    import base64
    import json as jsonlib

    from app import submit_xgboost as module

    sent = {}

    class StreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            rec = lambda st, at, v: {"station_id": st, "observed_at": at, "measurement": {"value": v, "unit": "passengers", "quality": "observed"}}
            return {
                "data": [
                    rec("A", "2026-01-01T11:30:00Z", "10.00"),
                    rec("A", "2026-01-01T12:00:00Z", "20.00"),
                    rec("A", "2026-01-01T12:15:00Z", "99.00"),  # despues del corte: no cuenta
                    rec("B", "2026-01-01T11:45:00Z", "30.00"),
                    {"station_id": "A", "observed_at": "2026-01-01T11:45:00Z", "measurement": {"value": None, "quality": "missing"}},
                ]
            }

    def fake_get(url, params=None, timeout=None, headers=None):
        sent.update(params)
        return StreamResponse()

    monkeypatch.undo()  # esta prueba usa la funcion real (el fixture autouse la reemplaza)
    monkeypatch.setattr("app.submit_xgboost.requests.get", fake_get)
    values = module._latest_values_stream(["A", "B"], CUT)
    assert values == {"A": 20.0, "B": 30.0}
    cursor = jsonlib.loads(base64.b64decode(sent["cursor"]))
    assert cursor[2] == "00000" and cursor[1].startswith("2026-01-01T09:00:00")  # corte - 3h


def test_409_cycle_closed_is_reported_as_missed_not_as_delivered(monkeypatch, capsys):
    ledger = []
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.record_submission", lambda *a, **k: ledger.append(a))
    handler = _posts([Resp(409, {"detail": {"code": "cycle_closed"}})])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert "PERDIDO" in capsys.readouterr().out
    assert ledger == []  # no se marca como entregado


def test_409_idempotency_conflict_is_delivered_and_recorded_without_overwriting(monkeypatch, capsys):
    ledger = []
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.record_submission", lambda *a, **k: ledger.append((a, k)))
    handler = _posts([Resp(409, {"detail": {"code": "idempotency_conflict"}})])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert "ya tenía una submission" in capsys.readouterr().out
    assert len(ledger) == 1 and ledger[0][0][0] == "cycle-1" and ledger[0][0][2] is False


def test_guardrail_rejection_retries_once_with_persistence_and_a_new_key(monkeypatch, capsys):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost._latest_values_db", lambda ids, cutoff: {s: 500.0 for s in ids})
    handler = _posts([Resp(422, {"detail": {"code": "invalid_targets"}}, "malo"), Resp(201, OK_BODY)])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert len(handler.seen) == 2
    assert handler.seen[0][0]["Idempotency-Key"] == "run-cycle-1"
    assert handler.seen[1][0]["Idempotency-Key"] == "run-cycle-1-fb"
    _persistence_check(handler.seen[1][1], 500.0)
    assert "submission rechazada (422)" in capsys.readouterr().out


def test_auth_errors_are_not_masked_by_the_fallback(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    handler = _posts([Resp(401, {"detail": {"code": "invalid_api_key"}})])
    with pytest.raises(RuntimeError, match="rechazada"):
        _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert len(handler.seen) == 1


def test_gateway_status_on_post_is_transient_then_upstream_unavailable(monkeypatch):
    from app.net import UpstreamUnavailable

    monkeypatch.setattr("app.net.time.sleep", lambda s: None)
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    handler = _posts([Resp(503, {}, "bad gateway")])
    with pytest.raises(UpstreamUnavailable):
        _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert len(handler.seen) == 3  # 3 intentos, no un fallo "real"


def test_successful_submission_is_recorded_in_the_ledger_with_receipt_and_commit(monkeypatch):
    ledger = []
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.record_submission", lambda cycle_id, details, overwrite=True: ledger.append((cycle_id, details, overwrite)))
    handler = _posts([Resp(201, {**OK_BODY, "attempt": 1})])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    cycle_id, details, overwrite = ledger[0]
    assert cycle_id == "cycle-1" and overwrite is True
    assert details["submission_id"] == "sub_1" and details["commit"] == "abcdef123456" and details["attempt"] == 1


def test_ledger_failure_never_breaks_a_delivered_submission(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("db caida")

    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.record_submission", boom)
    handler = _posts([Resp(201, OK_BODY)])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert len(handler.seen) == 1


def test_cycle_delivered_by_the_watchdog_is_skipped_via_the_ledger(monkeypatch):
    monkeypatch.setattr("app.submit_xgboost.USE_GENERIC_REGIME", False)
    monkeypatch.setattr("app.submit_xgboost.last_submission_cycle", lambda: "cycle-1")
    handler = _posts([Resp(201, OK_BODY)])
    _run_submit_with_history(monkeypatch, "A", CUT, FLAT, post_handler=handler)
    assert handler.seen == []


def test_collection_failure_of_any_kind_is_swallowed_so_the_submission_still_goes(monkeypatch, capsys):
    from app import submit_xgboost as module

    def boom():
        raise RuntimeError("psycopg.OperationalError")

    monkeypatch.setattr("app.submit_xgboost.collect_new_data", boom)
    module._collect_best_effort()  # no lanza
    assert "no se pudo recolectar" in capsys.readouterr().out


def test_collector_stale_exits_76_as_a_warning_not_a_failure(monkeypatch, capsys):
    from app.health import CollectorStale

    def stale():
        raise CollectorStale("collector sin correr hace 120 min")

    monkeypatch.setattr("app.submit_xgboost.submit_current_cycle", stale)
    from app.submit_xgboost import main

    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 76
    assert "::warning::" in capsys.readouterr().out
