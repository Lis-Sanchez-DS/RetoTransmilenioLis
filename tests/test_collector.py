from app.collector import collect_last_15


def test_collector_limits_batch_to_last_15_and_returns_cursor(monkeypatch):
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
    result = collect_last_15(lambda **kwargs: {"observations": records, "next_cursor": "cursor-15"})

    assert result == {"collected": 15, "cursor": "cursor-15"}
    insert_queries = [q for q, _ in captured["queries"] if 'INSERT INTO "Temp"' in q]
    assert len(insert_queries) == 15
