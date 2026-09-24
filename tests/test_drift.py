from datetime import datetime, timedelta, timezone

import pandas as pd

from app.drift import (
    DEFAULT_PARAMS,
    HYPERPARAM_GRID,
    REGIME_SAMPLE_SIZE,
    check_and_retrain,
    _new_data_matches_history,
    _select_best_params,
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


def test_new_data_matches_history_true_for_same_distribution(monkeypatch):
    # Reference and new samples both drawn from the same tight, repeating pattern.
    reference_rows = [(100 + (i % 20),) for i in range(REGIME_SAMPLE_SIZE)]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(reference_rows))
    recent = [(None, 100 + (i % 20)) for i in range(REGIME_SAMPLE_SIZE)]

    assert _new_data_matches_history("A", recent) is True


def test_new_data_matches_history_false_for_shifted_distribution(monkeypatch):
    # Reference is a tight low range; new data is a much larger, non-overlapping range.
    reference_rows = [(100 + (i % 20),) for i in range(REGIME_SAMPLE_SIZE)]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(reference_rows))
    recent = [(None, 5000 + (i % 7) * 500) for i in range(REGIME_SAMPLE_SIZE)]

    assert _new_data_matches_history("A", recent) is False


def test_new_data_matches_history_defaults_true_when_not_enough_data(monkeypatch):
    reference_rows = [(100,)] * 10  # fewer than REGIME_SAMPLE_SIZE
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(reference_rows))
    recent = [(None, 100)] * 10

    assert _new_data_matches_history("A", recent) is True


def test_select_best_params_falls_back_when_too_little_data_for_cv():
    """With too few rows to split into CV_SPLITS meaningful folds, the search
    is skipped entirely rather than cross-validating on scraps - DEFAULT_PARAMS
    (the config the model always used before this search existed) is returned
    unchanged, and importantly no XGBoost fit is attempted.
    """
    features = pd.DataFrame({"lag_1": range(15), "lag_2": range(15)})
    y = pd.Series(range(15), dtype=float)

    assert _select_best_params("A", features, y) == DEFAULT_PARAMS


def test_select_best_params_picks_a_grid_candidate_with_enough_data():
    """With enough rows for walk-forward CV, the search runs the full grid
    and returns one of its candidates (not necessarily DEFAULT_PARAMS).
    """
    n = 60
    y = pd.Series([float(i % 10) for i in range(n)])
    features = pd.DataFrame(
        {f"lag_{lag}": y.shift(lag).fillna(0.0) for lag in (1, 2)}
    )

    best_params = _select_best_params("A", features, y)

    assert best_params in HYPERPARAM_GRID


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
        elif 'ORDER BY observed_at DESC' in query:
            # Regime-check reference sample: most recent `limit` demand values.
            limit = params[1]
            self._result = [(demand,) for _, demand in self.historical_rows[-limit:]]
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


def test_check_and_retrain_keeps_full_history_when_new_data_matches_distribution(monkeypatch, tmp_path):
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 700 historical points: enough to clear every lag (max lag is 672).
    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # 60 fresh points continuing the exact same pattern: same distribution.
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 100 + (i % 20)) for i in range(60)
    ]
    # Same 60 points scored against the (drifted) live model: consistently ~30% off.
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
    # Temp rows for the drifted station were folded into Original Data and cleared.
    queries = [q for q, _ in fake_conn.executed]
    assert any('INSERT INTO "Original Data"' in q for q in queries)
    assert any('DELETE FROM "Temp"' in q for q in queries)
    # Same distribution: no history should have been discarded.
    assert not any('DELETE FROM "Original Data"' in q for q in queries)
    # The promotion must carry `prediction` along with `demand` - once these
    # rows move into "Original Data", station_accuracy_stats() needs the
    # prediction column there to keep scoring them within the 6h window.
    promote_query = next(q for q in queries if 'INSERT INTO "Original Data"' in q)
    assert "prediction" in promote_query


def test_check_and_retrain_drops_oldest_rows_when_distribution_shifted(monkeypatch, tmp_path):
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 700 historical points following a small, tight sawtooth (100-119).
    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # 60 new points from a completely different, much larger regime.
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 5000 + (i % 7) * 500) for i in range(60)
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
    queries_with_params = [(q, p) for q, p in fake_conn.executed]
    drop_queries = [(q, p) for q, p in queries_with_params if 'DELETE FROM "Original Data"' in q]
    # The regime shift was detected: the oldest len(temp_recent_rows) rows are dropped.
    assert len(drop_queries) == 1
    assert drop_queries[0][1] == (station, len(temp_recent_rows))
    assert any('INSERT INTO "Original Data"' in q for q, _ in queries_with_params)
    assert any('DELETE FROM "Temp"' in q for q, _ in queries_with_params)
