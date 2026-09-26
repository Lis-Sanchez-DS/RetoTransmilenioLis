from datetime import datetime, timedelta, timezone

import pandas as pd

from app.drift import (
    check_and_retrain,
    station_accuracy_stats,
    station_ewma_bias,
    stations_needing_retrain,
)


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=None):
        return self

    def fetchall(self):
        return self.rows


def test_station_accuracy_stats_computes_wape_accuracy_and_count(monkeypatch):
    rows = [
        ("A", 100, 100),
        ("A", 100, 90),
        ("A", 100, 95),
    ]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    stats = station_accuracy_stats()

    row = stats.loc[stats["station_id"] == "A"].iloc[0]
    assert row["count"] == 3
    # abs_error = 0 + 10 + 5 = 15; abs_demand = 300; accuracy = 1 - 15/300 = 0.95
    assert row["accuracy"] == 0.95


def test_station_accuracy_stats_empty_when_no_rows(monkeypatch):
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection([]))

    stats = station_accuracy_stats()

    assert stats.empty


def test_station_accuracy_stats_reads_both_tables(monkeypatch):
    """A station just promoted by a retrain has a short "Temp" history (it was
    just emptied); the accuracy window has to keep counting those rows once
    they move into "Original Data", or a freshly-retrained station would look
    like it has too little data to ever be checked again for hours.
    """
    captured = {}

    class CapturingConnection(FakeConnection):
        def execute(self, query, params=None):
            captured["query"] = query
            return self

    monkeypatch.setattr("app.drift.connection", lambda: CapturingConnection([]))

    station_accuracy_stats()

    assert '"Temp"' in captured["query"]
    assert '"Original Data"' in captured["query"]


def test_stations_needing_retrain_requires_enough_points_and_low_accuracy():
    stats = pd.DataFrame(
        [
            # Bad accuracy with a full window of checks: should retrain.
            {"station_id": "drifted", "accuracy": 0.80, "count": 4},
            # Bad accuracy but the window isn't full yet.
            {"station_id": "too_few", "accuracy": 0.80, "count": 2},
            # Full window, but accuracy is fine.
            {"station_id": "healthy", "accuracy": 0.95, "count": 4},
        ]
    )

    result = stations_needing_retrain(stats)

    assert result == ["drifted"]


def test_stations_needing_retrain_empty_stats():
    assert stations_needing_retrain(pd.DataFrame(columns=["station_id", "accuracy", "count"])) == []


class RoutedFakeConnection:
    """Dispatches each query to canned rows based on the table/params it touches."""

    def __init__(self, temp_accuracy_rows, historical_rows, temp_recent_rows):
        self.temp_accuracy_rows = temp_accuracy_rows
        self.historical_rows = historical_rows
        self.temp_recent_rows = temp_recent_rows
        self.executed = []
        self._last_rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def transaction(self):
        return self

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if 'FROM ranked' in query:
            self._result = self.temp_accuracy_rows
        elif 'INSERT INTO "Original Data"' in query:
            self._result = None
            self._last_rowcount = 0
        elif 'DELETE FROM "Temp"' in query:
            self._result = None
            self._last_rowcount = len(self.temp_recent_rows)
        elif 'FROM "Original Data" WHERE station_id' in query:
            self._result = self.historical_rows
        elif 'FROM "Temp" WHERE station_id' in query:
            self._result = self.temp_recent_rows
        else:
            raise AssertionError(f"Unexpected query in test: {query}")
        return self

    def fetchall(self):
        return self._result

    @property
    def rowcount(self):
        return self._last_rowcount


def _fake_upload(monkeypatch, uploads):
    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, headers=None, data=None, timeout=None):
        uploads.append({"url": url, "headers": headers, "bytes": len(data)})
        return FakeResponse()

    monkeypatch.setattr("app.drift.requests.post", fake_post)


def test_check_and_retrain_never_drops_history(monkeypatch, tmp_path):
    """Retraining only ever grows "Original Data": the new "Temp" points are
    folded in and nothing is ever deleted from history, regardless of how
    much new data arrived or what it looks like.
    """
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 700 historical points: enough to clear every lag (max lag is 672).
    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 100 + (i % 20)) for i in range(60)
    ]
    temp_accuracy_rows = [(station, demand, demand * 0.7) for _, demand in temp_recent_rows]

    fake_conn = RoutedFakeConnection(temp_accuracy_rows, historical_rows, temp_recent_rows)
    monkeypatch.setattr("app.drift.connection", lambda: fake_conn)
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("app.drift.SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setattr("app.drift.SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
    uploads = []
    _fake_upload(monkeypatch, uploads)

    retrained = check_and_retrain()

    assert retrained == ["A"]
    # All 4 horizon models were actually written to disk (not just claimed).
    model_paths = [tmp_path / f"xgboost_A_h{h}.joblib" for h in (1, 2, 3, 4)]
    for model_path in model_paths:
        assert model_path.exists()
        assert model_path.stat().st_size > 0
    # Each was uploaded to the right bucket/path with upsert, and with real file bytes.
    assert len(uploads) == 4
    uploaded_urls = sorted(u["url"] for u in uploads)
    expected_urls = sorted(
        f"https://fake.supabase.co/storage/v1/object/models/xgboost/xgboost_A_h{h}.joblib"
        for h in (1, 2, 3, 4)
    )
    assert uploaded_urls == expected_urls
    assert all(u["headers"]["x-upsert"] == "true" for u in uploads)
    queries_with_params = [(q, p) for q, p in fake_conn.executed]
    # History only grows: no DELETE FROM "Original Data" should ever run.
    assert not any('DELETE FROM "Original Data"' in q for q, _ in queries_with_params)
    assert any('INSERT INTO "Original Data"' in q for q, _ in queries_with_params)
    assert any('DELETE FROM "Temp"' in q for q, _ in queries_with_params)
    # The promotion must carry `prediction` along with `demand` - once these
    # rows move into "Original Data", station_accuracy_stats() needs the
    # prediction column there to keep scoring them within the 6h window.
    promote_query = next(q for q, _ in queries_with_params if 'INSERT INTO "Original Data"' in q)
    assert "prediction" in promote_query


def test_station_ewma_bias_tracks_a_sustained_miss(monkeypatch):
    """A station whose model has been consistently overpredicting should get
    a negative correction, growing as the miss persists across more points -
    mirroring 05100's real multi-hour demand collapse (predictions pinned
    high while actual demand kept falling)."""
    rows = [
        ("2026-01-01T00:00:00Z", 100.0, 200.0),
        ("2026-01-01T00:15:00Z", 100.0, 200.0),
        ("2026-01-01T00:30:00Z", 100.0, 200.0),
    ]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    bias = station_ewma_bias("A")

    # Every residual is -100 (actual - predicted); the EWMA should converge
    # toward -100 * damping, and always stay on the "correct downward" side.
    assert bias < 0
    params = {"alpha": 0.2, "damping": 0.5}
    expected_raw_bias = 0.0
    for _, demand, prediction in rows:
        expected_raw_bias = params["alpha"] * (demand - prediction) + (1 - params["alpha"]) * expected_raw_bias
    assert bias == params["damping"] * expected_raw_bias


def test_station_ewma_bias_zero_with_no_history(monkeypatch):
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection([]))

    assert station_ewma_bias("A") == 0.0
