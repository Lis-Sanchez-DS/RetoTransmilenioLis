from datetime import datetime, timedelta, timezone

import pandas as pd

from app.drift import (
    check_and_retrain,
    station_accuracy_stats,
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
            # Bad accuracy with enough data: should retrain.
            {"station_id": "drifted", "accuracy": 0.80, "count": 60},
            # Bad accuracy but not enough datapoints yet.
            {"station_id": "too_few", "accuracy": 0.80, "count": 10},
            # Plenty of data, but accuracy is fine.
            {"station_id": "healthy", "accuracy": 0.95, "count": 60},
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
        if 'FROM combined, cutoff' in query:
            self._result = self.temp_accuracy_rows
        elif 'INSERT INTO "Original Data"' in query:
            self._result = None
            self._last_rowcount = 0
        elif 'DELETE FROM "Original Data"' in query:
            # drop_oldest = params[1]; not asserted on directly here.
            self._result = None
            self._last_rowcount = params[1] if params else 0
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


def test_check_and_retrain_always_replaces_history_1_to_1(monkeypatch, tmp_path):
    """No statistical test gates the decision anymore: every retrain drops
    exactly as many of the oldest "Original Data" rows as new points it
    incorporates from "Temp" - a fixed-size sliding window, regardless of
    whether the new data "looks like" a distribution shift or not.
    """
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 700 historical points: enough to clear every lag (max lag is 672).
    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # 60 fresh points continuing the exact same pattern - previously this
    # would have been judged "same distribution" and kept all history.
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
    # The retrained model was actually written to disk (not just claimed).
    model_path = tmp_path / "xgboost_A.joblib"
    assert model_path.exists()
    assert model_path.stat().st_size > 0
    # It was uploaded to the right bucket/path with upsert, and with real file bytes.
    assert len(uploads) == 1
    assert uploads[0]["url"] == "https://fake.supabase.co/storage/v1/object/models/xgboost/xgboost_A.joblib"
    assert uploads[0]["headers"]["x-upsert"] == "true"
    assert uploads[0]["bytes"] == model_path.stat().st_size
    queries_with_params = [(q, p) for q, p in fake_conn.executed]
    # 1:1 replacement: exactly len(temp_recent_rows) oldest rows dropped, always.
    drop_queries = [(q, p) for q, p in queries_with_params if 'DELETE FROM "Original Data"' in q]
    assert len(drop_queries) == 1
    assert drop_queries[0][1] == (station, len(temp_recent_rows))
    assert any('INSERT INTO "Original Data"' in q for q, _ in queries_with_params)
    assert any('DELETE FROM "Temp"' in q for q, _ in queries_with_params)
    # The promotion must carry `prediction` along with `demand` - once these
    # rows move into "Original Data", station_accuracy_stats() needs the
    # prediction column there to keep scoring them within the 6h window.
    promote_query = next(q for q, _ in queries_with_params if 'INSERT INTO "Original Data"' in q)
    assert "prediction" in promote_query
