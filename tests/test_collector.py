from datetime import datetime, timedelta, timezone

from app.collector import LAGS, collect_new_data, predict_records


def test_collector_collects_all_pages_and_returns_cursor(monkeypatch):
    records = [
        {"station_id": "station-1", "observed_at": f"2026-01-01T00:{i:02d}:00Z", "demand": i}
        for i in range(16)
    ]
    captured = {}

    class FakeCursor:
        def execute(self, query, params=None):
            captured.setdefault("queries", []).append((query, params))

        def fetchone(self):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def transaction(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            captured.setdefault("queries", []).append((query, params))
            return self

        def fetchone(self):
            return None

    monkeypatch.setattr("app.collector.connection", lambda: FakeConnection())
    monkeypatch.setattr(
        "app.collector.predict_records",
        lambda batch: [(item, float(item["demand"])) for item in batch],
    )
    calls = []
    def fetcher(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"data": records[:15], "next_cursor": "cursor-15"}
        return {"data": records[15:], "next_cursor": None}

    result = collect_new_data(fetcher)

    assert result == {"collected": 16, "pages": 2, "cursor": "cursor-15"}
    insert_queries = [q for q, _ in captured["queries"] if 'INSERT INTO "Temp"' in q]
    assert len(insert_queries) == 16


def test_collector_stops_when_stream_is_drained(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def transaction(self):
            return self

        def execute(self, query, params=None):
            return self

        def fetchone(self):
            return None

    monkeypatch.setattr("app.collector.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.collector.predict_records", lambda batch: [])

    def fetcher(**kwargs):
        return {"data": [], "next_cursor": None}

    result = collect_new_data(fetcher)

    assert result == {"collected": 0, "pages": 1, "cursor": None}


def test_predict_records_reads_history_from_both_tables(monkeypatch):
    """A non-drifted station's recent history lives only in "Temp"

    ("Original Data" only updates via a drift retrain). predict_records()
    must still resolve every lag by reading both tables, not just
    "Original Data" - otherwise a fresh record's lag_1 lookup (the most
    recent point) fails the moment it isn't in the stale seed table.
    """
    station = "A"
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    new_ts = cutoff + timedelta(minutes=15)

    # Every lag point except lag_1 lives in "Original Data"; lag_1 (the most
    # recent one, exactly at cutoff) lives only in "Temp", simulating a
    # non-drifted station whose freshest data never reached "Original Data".
    original_rows = [
        (station, new_ts - timedelta(minutes=15 * lag), 100.0 + lag)
        for lag in LAGS
        if lag != 1
    ]
    temp_rows = [(station, cutoff, 101.0)]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=None):
            if 'FROM "Original Data"' in query:
                self._result = original_rows
            elif 'FROM "Temp"' in query:
                self._result = temp_rows
            else:
                raise AssertionError(f"Unexpected query: {query}")
            return self

        def fetchall(self):
            return self._result

    class FakeModel:
        def predict(self, features):
            return [42.0]

    monkeypatch.setattr("app.collector.connection", lambda: FakeConnection())
    monkeypatch.setattr("app.collector.joblib.load", lambda path: FakeModel())

    new_record = {
        "station_id": station,
        "observed_at": (cutoff + timedelta(minutes=15)).isoformat(),
        "demand": 999,
    }

    result = predict_records([new_record])

    assert len(result) == 1
    _, prediction = result[0]
    assert prediction == 42.0
