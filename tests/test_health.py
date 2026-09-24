from datetime import datetime, timedelta, timezone

import pytest

from app.health import HEARTBEAT_STALE_AFTER_MINUTES, CollectorStale, check_collector_heartbeat


class FakeConnection:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=None):
        return self

    def fetchone(self):
        return self.row


def test_check_collector_heartbeat_passes_when_recent(monkeypatch):
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    monkeypatch.setattr("app.health.connection", lambda: FakeConnection((recent,)))

    check_collector_heartbeat()  # should not raise


def test_check_collector_heartbeat_passes_when_no_rows_yet(monkeypatch):
    monkeypatch.setattr("app.health.connection", lambda: FakeConnection(None))

    check_collector_heartbeat()  # nothing to compare against - not an error


def test_check_collector_heartbeat_raises_when_stale(monkeypatch):
    stale = datetime.now(timezone.utc) - timedelta(minutes=HEARTBEAT_STALE_AFTER_MINUTES + 1)
    monkeypatch.setattr("app.health.connection", lambda: FakeConnection((stale,)))

    with pytest.raises(CollectorStale):
        check_collector_heartbeat()


def test_check_collector_heartbeat_handles_naive_timestamps(monkeypatch):
    stale_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=HEARTBEAT_STALE_AFTER_MINUTES + 1
    )
    monkeypatch.setattr("app.health.connection", lambda: FakeConnection((stale_naive,)))

    with pytest.raises(CollectorStale):
        check_collector_heartbeat()
