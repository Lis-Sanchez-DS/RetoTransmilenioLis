from app.collector import collect_new_data


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
        if len(calls) == 2:
            return {"data": records[15:], "next_cursor": "cursor-16"}
        return {"data": [], "next_cursor": "cursor-16"}

    result = collect_new_data(fetcher)

    assert result == {"collected": 16, "pages": 2, "cursor": "cursor-16"}
    insert_queries = [q for q, _ in captured["queries"] if 'INSERT INTO "Temp"' in q]
    assert len(insert_queries) == 16
