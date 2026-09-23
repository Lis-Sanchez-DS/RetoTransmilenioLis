import pandas as pd

from app.drift import station_accuracy_stats, stations_needing_retrain


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
