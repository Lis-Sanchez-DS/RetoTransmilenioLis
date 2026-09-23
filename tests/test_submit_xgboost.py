from datetime import datetime, timedelta, timezone

from app.submit_xgboost import predict_cycle_targets


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


def test_predict_cycle_targets_raises_on_missing_history():
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    targets = [{"station_id": station, "target_at": (cutoff + timedelta(minutes=15)).isoformat()}]
    models = {station: EchoPlusOneModel()}

    try:
        predict_cycle_targets(targets, {}, models)
    except RuntimeError as exc:
        assert "No hay suficiente historia" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing history")
