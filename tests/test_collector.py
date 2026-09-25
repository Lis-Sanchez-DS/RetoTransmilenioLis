import urllib.error
from datetime import datetime, timedelta, timezone

from app.collector import LAGS, _synthetic_cursor, collect_new_data, predict_records


def test_collector_collects_all_pages_and_returns_cursor(monkeypatch):
    records = [
        {
            "station_id": "station-1",
            "observed_at": f"2026-01-01T00:{i:02d}:00Z",
            "demand": i,
            "released_at": f"2026-01-02T00:{i:02d}:00Z",
        }
        for i in range(16)
    ]
    captured = {}

    class FakeCursor:
        def execute(self, query, params=None):
            captured.setdefault("queries", []).append((query, params))

        def fetchone(self):
            return None

    class FakeConnection:
        rowcount = 1

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

    # The first page's real server cursor gets saved as-is; the second (and
    # final) page hits next_cursor: null - the normal "caught up" case - so
    # the resume point falls back to a cursor built from its own last record.
    expected_final_cursor = _synthetic_cursor(records[-1])
    assert result == {"collected": 16, "inserted": 16, "pages": 2, "cursor": expected_final_cursor}
    insert_queries = [q for q, _ in captured["queries"] if 'INSERT INTO "Temp"' in q]
    assert len(insert_queries) == 16
    cursor_saves = [params[0] for q, params in captured["queries"] if "INSERT INTO collector_state" in q]
    assert cursor_saves == ["cursor-15", expected_final_cursor]


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

    assert result == {"collected": 0, "inserted": 0, "pages": 1, "cursor": None}


def test_synthetic_cursor_matches_real_api_cursor():
    """Regression fixture: this is a real cursor the live Pulso TransMi API
    returned for this exact record during development. If the server ever
    changes its cursor encoding, this is the test that should catch it.
    """
    record = {
        "station_id": "05000",
        "observed_at": "2026-09-09T05:00:00Z",
        "demand": 109,
        "released_at": "2026-09-21T15:30:04.958049Z",
    }
    real_cursor_from_api = (
        "WyIyMDI2LTA5LTIxVDE1OjMwOjA0Ljk1ODA0OSswMDowMCIsIjIwMjYtMDktMDlUMDU6MDA6MDArMDA6MDAiLCIwNTAwMCJd"
    )

    assert _synthetic_cursor(record) == real_cursor_from_api


def test_fetch_page_falls_back_to_full_refetch_when_cursor_rejected(monkeypatch):
    from app.collector import _fetch_page

    requests_made = []

    def fake_urlopen(request, timeout=None):
        requests_made.append(request.full_url)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"data": [], "next_cursor": null}'

        if "cursor=" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 422, "cursor invalido", None, None)
        return FakeResponse()

    monkeypatch.setattr("app.collector.urlopen", fake_urlopen)

    payload = _fetch_page(None, "un-cursor-que-el-servidor-ya-no-acepta")

    assert payload == {"data": [], "next_cursor": None}
    assert len(requests_made) == 2
    assert "cursor=" in requests_made[0]
    assert "cursor=" not in requests_made[1]


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
