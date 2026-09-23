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


def test_station_accuracy_stats_computes_mean_std_count(monkeypatch):
    rows = [
        ("A", 100, 100),
        ("A", 100, 90),
        ("A", 100, 95),
    ]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    stats = station_accuracy_stats()

    row = stats.loc[stats["station_id"] == "A"].iloc[0]
    assert row["count"] == 3
    assert 0.9 < row["mean_accuracy"] < 1.0


def test_station_accuracy_stats_empty_when_no_rows(monkeypatch):
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection([]))

    stats = station_accuracy_stats()

    assert stats.empty


def test_stations_needing_retrain_requires_all_three_conditions():
    stats = pd.DataFrame(
        [
            # Consistently bad and stable with enough data: should retrain.
            {"station_id": "drifted", "mean_accuracy": 0.80, "std_accuracy": 0.01, "count": 60},
            # Bad mean but too noisy (high std): not a stable drift signal.
            {"station_id": "noisy", "mean_accuracy": 0.80, "std_accuracy": 0.10, "count": 60},
            # Bad and stable but not enough datapoints yet.
            {"station_id": "too_few", "mean_accuracy": 0.80, "std_accuracy": 0.01, "count": 10},
            # Stable and plenty of data, but accuracy is fine.
            {"station_id": "healthy", "mean_accuracy": 0.95, "std_accuracy": 0.01, "count": 60},
        ]
    )

    result = stations_needing_retrain(stats)

    assert result == ["drifted"]


def test_stations_needing_retrain_empty_stats():
    assert stations_needing_retrain(pd.DataFrame(columns=["station_id", "mean_accuracy", "std_accuracy", "count"])) == []
