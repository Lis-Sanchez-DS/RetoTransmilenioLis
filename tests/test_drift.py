from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from app.drift import (
    RatioTargetModel,
    _train_station_horizon_model,
    ACCURACY_THRESHOLD,
    ADAPTIVE_MIN_THRESHOLD,
    check_and_retrain,
    station_accuracy_stats,
    station_adaptive_thresholds,
    station_baseline_accuracy,
    station_horizon_bias,
    stations_needing_retrain,
)


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self._query = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=None):
        self._query = query
        return self

    def fetchall(self):
        return self.rows

    def fetchone(self):
        # _is_page_hinkley_enabled's collector_state lookup: no stored
        # decision in this fixture, so it falls back to the
        # PH_ENABLED_STATIONS bootstrap default, same as a station that's
        # never retrained yet.
        if "collector_state" in self._query:
            return None
        return self.rows[0] if self.rows else None


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


def test_accuracy_threshold_is_back_to_085():
    assert ACCURACY_THRESHOLD == 0.85


def test_station_baseline_accuracy_skips_stations_with_too_little_history(monkeypatch):
    # (station_id, sum_abs_error, sum_abs_demand, count)
    rows = [("steady", 100.0, 1000.0, 192), ("new", 10.0, 100.0, 10)]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    baselines = station_baseline_accuracy()

    assert baselines == {"steady": 0.9}


def test_station_adaptive_thresholds_are_baseline_minus_drop_clamped():
    thresholds = station_adaptive_thresholds(
        {"accurate": 0.95, "ordinary": 0.86, "noisy": 0.72, "collapsed": 0.30}
    )

    assert thresholds["accurate"] == ACCURACY_THRESHOLD  # capped at 0.85
    assert abs(thresholds["ordinary"] - 0.81) < 1e-9
    assert thresholds["noisy"] == ADAPTIVE_MIN_THRESHOLD  # floored at 0.75
    assert thresholds["collapsed"] == ADAPTIVE_MIN_THRESHOLD  # never drags the bar to zero


def test_stations_needing_retrain_uses_each_stations_own_threshold():
    stats = pd.DataFrame(
        [
            # Noisy station that always runs ~0.78: 0.78 is normal for it (bar 0.75).
            {"station_id": "noisy", "accuracy": 0.78, "count": 4},
            # Steady 0.89 station sliding to 0.80 is a real drop (bar 0.84).
            {"station_id": "steady", "accuracy": 0.80, "count": 4},
            # No baseline yet: falls back to the fixed 0.85 bar.
            {"station_id": "new", "accuracy": 0.80, "count": 4},
        ]
    )

    result = stations_needing_retrain(stats, {"noisy": 0.75, "steady": 0.84})

    assert result == ["new", "steady"]


class RoutedFakeConnection:
    """Dispatches each query to canned rows based on the table/params it touches."""

    def __init__(self, temp_accuracy_rows, historical_rows, temp_recent_rows, ph_history_rows=None):
        self.temp_accuracy_rows = temp_accuracy_rows
        self.historical_rows = historical_rows
        self.temp_recent_rows = temp_recent_rows
        # _page_hinkley_would_help's full-history query, keyed by state_key
        # suffix (default: not enough rows to form an opinion, i.e.
        # _update_page_hinkley_enabled_state should no-op). Override per
        # test via this dict.
        self.ph_history_rows = ph_history_rows if ph_history_rows is not None else []
        self.collector_state: dict[str, str] = {}
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
        if 'FROM baseline_ranked' in query:
            # station_baseline_accuracy: no baseline in this fixture, so
            # every station falls back to the fixed ACCURACY_THRESHOLD.
            self._result = []
        elif 'FROM ranked' in query:
            self._result = self.temp_accuracy_rows
        elif 'INSERT INTO "Original Data"' in query:
            self._result = None
            self._last_rowcount = 0
        elif 'DELETE FROM "Temp"' in query:
            self._result = None
            self._last_rowcount = len(self.temp_recent_rows)
        elif 'INSERT INTO collector_state' in query:
            # _update_page_hinkley_enabled_state's upsert: params = (state_key, value).
            self.collector_state[params[0]] = params[1]
            self._result = None
        elif 'FROM collector_state' in query:
            # _is_page_hinkley_enabled's lookup: params = (state_key,).
            value = self.collector_state.get(params[0])
            self._result = (value,) if value is not None else None
        elif ') recent' in query:
            # station_bias_components / _station_has_fresh_page_hinkley_alarm's
            # bounded combined-history query. None of this fixture's rows
            # carry a `prediction` column, so there's nothing to evaluate -
            # no alarm, and the boosted path (if enabled) sees no rows either.
            self._result = []
        elif 'UNION ALL' in query and 'ORDER BY observed_at' in query:
            # _page_hinkley_would_help's unbounded full-history query (no
            # DESC, no ") recent" wrapper - distinct from the two branches
            # above).
            self._result = self.ph_history_rows
        elif 'FROM "Original Data" WHERE station_id' in query:
            self._result = self.historical_rows
        elif 'FROM "Temp" WHERE station_id' in query:
            self._result = self.temp_recent_rows
        else:
            raise AssertionError(f"Unexpected query in test: {query}")
        return self

    def fetchall(self):
        return self._result

    def fetchone(self):
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

    # 700 historical points: enough to clear every lag (max lag is 96).
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
    # Only the h1 model gets retrained now - h2/h3/h4 are produced by
    # chaining h1 forward at inference time (see submit_xgboost.py).
    model_path = tmp_path / "xgboost_A_h1.joblib"
    assert model_path.exists()
    assert model_path.stat().st_size > 0
    for stale_horizon in (2, 3, 4):
        assert not (tmp_path / f"xgboost_A_h{stale_horizon}.joblib").exists()
    # Uploaded to the right bucket/path with upsert, with real file bytes.
    assert len(uploads) == 1
    assert uploads[0]["url"] == "https://fake.supabase.co/storage/v1/object/models/xgboost/xgboost_A_h1.joblib"
    assert uploads[0]["headers"]["x-upsert"] == "true"
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


def test_check_and_retrain_skips_station_below_min_new_points(monkeypatch, tmp_path):
    """A station that's below ACCURACY_THRESHOLD but has only a small trickle
    of new "Temp" rows (fewer than MIN_NEW_FOR_RETRAIN) should NOT retrain.

    This is the fix for the 2026-09-27 incident: has_pending_data() gates
    the drift check on "Temp has anything at all, for any station" so a
    stalled feed never blocks retraining forever, but without this
    per-station minimum, that meant a handful of new rows (as few as 1-2)
    was enough to trigger a full 4-horizon retrain + 4 Supabase uploads for
    a chronically-drifted station on almost every ~30min collector cycle.
    """
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # Below MIN_NEW_FOR_RETRAIN (20): a real trickle, but not enough to
    # justify spending a retrain on it yet.
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 100 + (i % 20)) for i in range(5)
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

    assert retrained == []
    assert uploads == []
    assert list(tmp_path.iterdir()) == []
    queries_with_params = [(q, p) for q, p in fake_conn.executed]
    assert not any('INSERT INTO "Original Data"' in q for q, _ in queries_with_params)
    assert not any('DELETE FROM "Temp"' in q for q, _ in queries_with_params)


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

    bias = station_horizon_bias("A", 1)

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

    assert station_horizon_bias("A", 1) == 0.0


def test_station_ewma_bias_processes_rows_in_chronological_order(monkeypatch):
    """The query now asks for the most recent rows in DESCENDING order (so a
    LIMIT can bound how much history gets pulled), but the EWMA recursion is
    order-sensitive and must run oldest-to-newest. Uses distinct (non-
    constant) residuals, where getting the reversal wrong would produce a
    different number - a fixture where all residuals are equal (as in the
    other tests here) can't actually catch a broken reversal.
    """
    # Rows exactly as the DB would return them under the new query: newest first.
    rows_desc = [
        ("2026-01-01T00:30:00Z", 100.0, 150.0),  # residual -50, newest
        ("2026-01-01T00:15:00Z", 100.0, 120.0),  # residual -20
        ("2026-01-01T00:00:00Z", 100.0, 100.0),  # residual 0, oldest
    ]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(list(rows_desc)))

    bias = station_horizon_bias("A", 1)

    alpha, damping = 0.2, 0.5
    expected_raw = 0.0
    for _, demand, prediction in reversed(rows_desc):  # oldest to newest
        expected_raw = alpha * (demand - prediction) + (1 - alpha) * expected_raw
    assert bias == damping * expected_raw

    # Pin down that processing DB order (newest-first) without reversing
    # really would give a different number for this fixture, so this test
    # would actually fail if the reversal were ever removed by mistake.
    wrong_order_raw = 0.0
    for _, demand, prediction in rows_desc:
        wrong_order_raw = alpha * (demand - prediction) + (1 - alpha) * wrong_order_raw
    assert damping * wrong_order_raw != bias


def test_ewma_lookback_rows_bounds_query_and_stays_accurate(monkeypatch):
    """station_ewma_bias() must ask the DB for a bounded number of rows (via
    LIMIT), not a station's entire history - fetching everything just to
    feed a fast-decaying EWMA (alpha=0.2 makes anything past ~100 rows back
    numerically irrelevant) was one of the two largest sources of Supabase
    DB egress. Also checks the derived lookback is sane for a slower-decaying
    alpha, in case a future per-station override ever picks one.
    """
    from app.drift import _ewma_lookback_rows, EWMA_TRUNCATION_EPSILON

    captured = {}

    class CapturingConnection(FakeConnection):
        def execute(self, query, params=None):
            captured["query"] = query
            captured["params"] = params
            return super().execute(query, params)

    monkeypatch.setattr("app.drift.connection", lambda: CapturingConnection([]))

    station_horizon_bias("A", 1)

    assert "LIMIT" in captured["query"]
    default_alpha = 0.2
    assert captured["params"][-1] == _ewma_lookback_rows(default_alpha)
    # Sanity on the math itself: truncating at the derived lookback really
    # does leave less than EWMA_TRUNCATION_EPSILON of the total weight
    # unaccounted for, regardless of which alpha is plugged in.
    for alpha in (0.2, 0.05, 0.5):
        lookback = _ewma_lookback_rows(alpha)
        assert (1 - alpha) ** lookback < EWMA_TRUNCATION_EPSILON


def test_page_hinkley_enabled_station_matches_plain_ewma_when_no_alarm_fires(monkeypatch):
    """For a Page-Hinkley-enabled station whose residuals never trip the
    alarm threshold, station_horizon_bias() must produce the exact same
    number as the plain fixed EWMA (at every horizon) - the boosted regime
    should be provably a no-op absent a real, sustained shift, not just
    "close"."""
    from app.drift import PH_ENABLED_STATIONS, DEFAULT_EWMA_PARAMS

    assert "05100" in PH_ENABLED_STATIONS
    rows_desc = [
        ("2026-01-01T00:30:00Z", 100.0, 102.0),
        ("2026-01-01T00:15:00Z", 100.0, 98.0),
        ("2026-01-01T00:00:00Z", 100.0, 101.0),
    ]
    alpha, damping = DEFAULT_EWMA_PARAMS["alpha"], DEFAULT_EWMA_PARAMS["damping"]
    expected_raw = 0.0
    for _, demand, prediction in reversed(rows_desc):
        expected_raw = alpha * (demand - prediction) + (1 - alpha) * expected_raw
    expected = damping * expected_raw

    for horizon in (1, 2, 3, 4):
        monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(list(rows_desc)))
        assert station_horizon_bias("05100", horizon) == expected


def test_page_hinkley_alarm_fires_on_sustained_shift_not_on_routine_noise():
    """The Page-Hinkley detector (used both for the boosted-EWMA regime and
    for the fast retrain path) must stay quiet through ordinary alternating
    noise, but fire once residuals shift hard and stay shifted - mirroring
    05100's real collapse (predictions pinned high while actual demand kept
    falling)."""
    from app.drift import _page_hinkley_recently_alarmed

    quiet_rows = [
        ("t", 100.0, 100.0 + (5 if i % 2 == 0 else -5)) for i in range(150)
    ]
    assert _page_hinkley_recently_alarmed(quiet_rows) is False

    shifted_rows = list(quiet_rows) + [
        ("t", 100.0, 100.0 + 60 + (5 if i % 2 == 0 else -5)) for i in range(20)
    ]
    assert _page_hinkley_recently_alarmed(shifted_rows) is True


def _alarm_routed_connection(temp_accuracy_rows, historical_rows, temp_recent_rows):
    """A RoutedFakeConnection where the Page-Hinkley alarm query always
    reports a real, sustained shift (predictions pinned high while demand
    suddenly drops and stays down, exactly like 05100's real collapse)."""
    quiet = [(f"t{i}", 100.0, 100.0 + (5 if i % 2 == 0 else -5)) for i in range(150)]
    shifted = [(f"t{150+i}", 100.0, 160.0 + (5 if i % 2 == 0 else -5)) for i in range(20)]
    ph_history_rows = shifted[::-1] + quiet[::-1]  # DESC order, as the real query returns

    class AlarmRoutedConnection(RoutedFakeConnection):
        def execute(self, query, params=None):
            if ') recent' in query:
                self.executed.append((query, params))
                self._result = ph_history_rows
                return self
            return super().execute(query, params)

    return AlarmRoutedConnection(temp_accuracy_rows, historical_rows, temp_recent_rows)


def test_check_and_retrain_page_hinkley_alarm_bypasses_accuracy_threshold(monkeypatch, tmp_path):
    """A confirmed Page-Hinkley alarm pulls a station INTO consideration even
    though its own crude 4-point rolling accuracy still looks fine (it
    bypasses ACCURACY_THRESHOLD, reacting faster than that short average
    could). With enough new rows piled up too, it retrains - see the
    row-count bypass test below for the case where it doesn't have enough."""
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # >= MIN_NEW_FOR_RETRAIN (20): the row-count floor is satisfied.
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 100 + (i % 20)) for i in range(25)
    ]
    # Good accuracy (demand*0.99): stations_needing_retrain would NOT flag
    # this station on its own - only the Page-Hinkley alarm pulls it in.
    temp_accuracy_rows = [(station, demand, demand * 0.99) for _, demand in temp_recent_rows]

    fake_conn = _alarm_routed_connection(temp_accuracy_rows, historical_rows, temp_recent_rows)
    monkeypatch.setattr("app.drift.connection", lambda: fake_conn)
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("app.drift.SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setattr("app.drift.SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
    uploads = []
    _fake_upload(monkeypatch, uploads)

    retrained = check_and_retrain()

    assert retrained == ["A"]
    assert len(uploads) == 1


def test_check_and_retrain_page_hinkley_alarm_bypasses_min_new_points_too(monkeypatch, tmp_path):
    """A confirmed Page-Hinkley alarm now bypasses MIN_NEW_FOR_RETRAIN as
    well as ACCURACY_THRESHOLD (changed 2026-09-29): even a station with
    ZERO new "Temp" rows must still retrain if its alarm is active, because
    the only way its ph_enabled flag gets re-evaluated against a real,
    currently-active shift is at retrain time - waiting for 20 fresh rows
    would leave it on a stale (often disabled) boost decision through the
    exact window it would help most. This is a deliberate reversal of the
    2026-09-28 design (the previous version of this test asserted the
    opposite): the accepted cost is that a station whose alarm stays active
    across many cycles can retrain repeatedly on close to the same data."""
    station = "A"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    # Zero new rows in "Temp" for retraining - the alarm alone must still be
    # enough. station_accuracy_stats' own recent-checks query still needs at
    # least one row so "A" appears in its result at all (real accuracy is
    # irrelevant here - only the alarm, not ACCURACY_THRESHOLD, is doing the
    # bypassing in this test).
    temp_recent_rows: list[tuple] = []
    temp_accuracy_rows = [(station, 100.0, 99.0)]

    fake_conn = _alarm_routed_connection(temp_accuracy_rows, historical_rows, temp_recent_rows)
    monkeypatch.setattr("app.drift.connection", lambda: fake_conn)
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("app.drift.SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setattr("app.drift.SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
    uploads = []
    _fake_upload(monkeypatch, uploads)

    retrained = check_and_retrain()

    assert retrained == ["A"]
    assert len(uploads) == 1


def _quiet_then_shifted_rows(quiet_n: int, shift_n: int, shift_mag: float = 60.0) -> list[tuple]:
    """(observed_at, demand, prediction) triples, chronological: quiet_n
    rows of ordinary alternating noise, then shift_n rows where the
    prediction stays pinned `shift_mag` above demand - a sustained regime
    shift like 05100's real collapse (predictions pinned high while actual
    demand fell and stayed down)."""
    quiet = [(f"t{i}", 100.0, 100.0 + (5 if i % 2 == 0 else -5)) for i in range(quiet_n)]
    shifted = [
        (f"t{quiet_n + i}", 100.0, 100.0 + shift_mag + (5 if i % 2 == 0 else -5))
        for i in range(shift_n)
    ]
    return quiet + shifted


def test_page_hinkley_would_help_none_when_too_little_history(monkeypatch):
    """_update_page_hinkley_enabled_state must not flip a station on or off
    off the back of too little data - a station with fewer than
    PH_LOOKBACK_ROWS recorded prediction/actual pairs should get None (i.e.
    "leave the existing decision alone"), not a guess."""
    from app.drift import _page_hinkley_would_help, PH_LOOKBACK_ROWS

    rows = _quiet_then_shifted_rows(50, 10)
    assert len(rows) < PH_LOOKBACK_ROWS
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    assert _page_hinkley_would_help("A") is None


def test_page_hinkley_would_help_true_for_sustained_shift_false_for_pure_noise(monkeypatch):
    """The yes/no question _update_page_hinkley_enabled_state actually asks:
    would the single SHARED PH_ADAPTIVE_PARAMS config (never re-tuned per
    station) beat the plain fixed EWMA by a real margin on this station's
    own history? A station with a real, sustained collapse should clear the
    bar; a station that's just ordinary noise the whole time should not -
    this is what keeps 09000/09122 (which never showed this pattern) from
    getting the boost even after future retrains."""
    from app.drift import _page_hinkley_would_help

    shifted_rows = _quiet_then_shifted_rows(200, 100)
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(shifted_rows))
    assert _page_hinkley_would_help("A") is True

    quiet_rows = _quiet_then_shifted_rows(300, 0)
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(quiet_rows))
    assert _page_hinkley_would_help("A") is False


def test_update_and_read_back_page_hinkley_enabled_state(monkeypatch):
    """_update_page_hinkley_enabled_state's decision must round-trip through
    _is_page_hinkley_enabled via collector_state - the persistence a fresh
    GitHub Actions process needs to remember a decision made by a previous
    retrain."""
    from app.drift import _is_page_hinkley_enabled, _update_page_hinkley_enabled_state, PH_ENABLED_STATIONS

    conn = RoutedFakeConnection([], [], [], ph_history_rows=_quiet_then_shifted_rows(200, 100))
    monkeypatch.setattr("app.drift.connection", lambda: conn)

    # A station outside the bootstrap default starts falling back to it.
    station = "never-seen-before"
    assert station not in PH_ENABLED_STATIONS
    assert _is_page_hinkley_enabled(station) is False

    _update_page_hinkley_enabled_state(station)

    assert _is_page_hinkley_enabled(station) is True
    assert conn.collector_state[f"page_hinkley_enabled:{station}"] == "true"


def test_check_and_retrain_updates_page_hinkley_state_for_newly_justified_station(monkeypatch, tmp_path):
    """The actual behavior the user asked for: when a station retrains, it
    should also get checked for whether the shared Page-Hinkley config is
    now justified for it - not just the fixed 05100/06000/07111 set decided
    once on 2026-09-27. A station outside that set whose full history now
    shows a real, sustained shift should come out of this retrain with the
    boost turned on."""
    station = "Z"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    historical_rows = [
        (start + timedelta(minutes=15 * i), 100 + (i % 20)) for i in range(700)
    ]
    temp_recent_rows = [
        (start + timedelta(minutes=15 * (700 + i)), 100 + (i % 20)) for i in range(60)
    ]
    temp_accuracy_rows = [(station, demand, demand * 0.7) for _, demand in temp_recent_rows]
    ph_history_rows = _quiet_then_shifted_rows(200, 100)

    fake_conn = RoutedFakeConnection(
        temp_accuracy_rows, historical_rows, temp_recent_rows, ph_history_rows=ph_history_rows
    )
    monkeypatch.setattr("app.drift.connection", lambda: fake_conn)
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("app.drift.SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setattr("app.drift.SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
    _fake_upload(monkeypatch, [])
    # Isolate Page-Hinkley's own decision from the EWMA re-tune, which would
    # otherwise adapt to this synthetic shift and change PH's baseline.
    monkeypatch.setattr("app.drift._update_ewma_params_state", lambda station_id: None)

    retrained = check_and_retrain()

    assert retrained == [station]
    assert fake_conn.collector_state[f"page_hinkley_enabled:{station}"] == "true"


def _biased_rows(n, bias):
    """Prediction persistently `bias` below demand: a big correction helps."""
    return [(f"t{i}", 100.0, 100.0 - bias) for i in range(n)]


def test_tune_ewma_params_needs_history_and_margin(monkeypatch):
    from app.drift import _tune_ewma_params, DEFAULT_EWMA_PARAMS, EWMA_TUNE_MIN_ROWS

    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(_biased_rows(EWMA_TUNE_MIN_ROWS - 1, 30)))
    assert _tune_ewma_params("A") is None  # too little history to trust a search

    # Unbiased residuals: nothing beats the default by the margin.
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(_biased_rows(300, 0)))
    assert _tune_ewma_params("A") == DEFAULT_EWMA_PARAMS

    # Persistent bias: a stronger correction wins clearly.
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(_biased_rows(300, 30)))
    tuned = _tune_ewma_params("A")
    assert tuned != DEFAULT_EWMA_PARAMS and tuned["damping"] > DEFAULT_EWMA_PARAMS["damping"]


def test_ewma_params_round_trip_through_collector_state(monkeypatch):
    from app.drift import _ewma_params, _update_ewma_params_state, DEFAULT_EWMA_PARAMS

    conn = RoutedFakeConnection([], [], [], ph_history_rows=_biased_rows(300, 30))
    monkeypatch.setattr("app.drift.connection", lambda: conn)
    assert _ewma_params("A") == DEFAULT_EWMA_PARAMS  # nothing stored yet

    tuned = _update_ewma_params_state("A")

    assert conn.collector_state["ewma_params:A"] == f"{tuned['alpha']},{tuned['damping']}"
    assert _ewma_params("A") == tuned


def test_blend_horizon_bias_full_boost_at_shortest_full_regular_at_longest():
    """blend_horizon_bias must return exactly boosted_bias at horizon 1 and
    exactly regular_bias at horizon 4 (HORIZONS' own min/max) - the two
    anchors EWMA_DECAY_POWER's front-loaded curve is built around, so a
    boosted station's +15min submission keeps the full win and its +60min
    submission can never do worse than plain fixed EWMA."""
    from app.drift import blend_horizon_bias, HORIZONS

    regular, boosted = 1.0, -9.0
    assert blend_horizon_bias(regular, boosted, min(HORIZONS)) == boosted
    assert blend_horizon_bias(regular, boosted, max(HORIZONS)) == regular


def test_station_bias_components_equal_when_page_hinkley_disabled(monkeypatch):
    """A station without the boost enabled must get identical regular and
    boosted components (both equal to the plain fixed EWMA bias) - so
    blend_horizon_bias produces the same, unboosted correction at every
    horizon for it, regardless of EWMA_DECAY_POWER."""
    from app.drift import station_bias_components, DEFAULT_EWMA_PARAMS

    rows = [
        ("2026-01-01T00:00:00Z", 100.0, 200.0),
        ("2026-01-01T00:15:00Z", 100.0, 200.0),
    ]
    monkeypatch.setattr("app.drift.connection", lambda: FakeConnection(rows))

    regular, boosted = station_bias_components("A")  # "A" is outside PH_ENABLED_STATIONS

    assert regular == boosted
    alpha, damping = DEFAULT_EWMA_PARAMS["alpha"], DEFAULT_EWMA_PARAMS["damping"]
    expected = 0.0
    for _, demand, prediction in rows:
        expected = alpha * (demand - prediction) + (1 - alpha) * expected
    assert regular == damping * expected


class _FixedLogRatio:
    """Stand-in for the inner XGBRegressor: always predicts a fixed log-ratio."""

    n_features_in_ = 8

    def __init__(self, log_ratio):
        self.log_ratio = log_ratio

    def predict(self, rows):
        return np.full(len(rows), self.log_ratio)


def test_ratio_target_model_scales_last_value_by_predicted_ratio():
    model = RatioTargetModel(_FixedLogRatio(np.log(1.5)))
    rows = [[99.0] + [0.0] * 7, [999.0] + [0.0] * 7]

    levels = model.predict(rows)

    # level = ratio * (lag_1 + 1) - 1 -> 1.5 * 100 - 1 and 1.5 * 1000 - 1
    assert np.allclose(levels, [149.0, 1499.0])
    assert model.n_features_in_ == 8


def test_ratio_target_model_never_predicts_below_zero():
    model = RatioTargetModel(_FixedLogRatio(np.log(0.001)))

    assert model.predict([[0.0] + [0.0] * 7])[0] == 0.0


def test_ratio_target_model_accepts_plain_list_and_numpy_inputs():
    model = RatioTargetModel(_FixedLogRatio(0.0))

    assert model.predict([[10.0] + [0.0] * 7])[0] == 10.0
    assert model.predict(np.array([[10.0] + [0.0] * 7]))[0] == 10.0


def test_trained_ratio_model_extrapolates_beyond_its_training_range(monkeypatch, tmp_path):
    """The reason this model exists: a level-target tree is capped at the
    largest demand it ever saw; the ratio model must keep scaling with the
    last value instead."""
    monkeypatch.setattr("app.drift.MODEL_DIR", str(tmp_path))
    slots = 800
    observed_at = pd.date_range("2026-08-01", periods=slots, freq="15min", tz="UTC")
    # Smooth daily wave between ~50 and ~150: nothing here ever exceeds 200.
    demand = (100 + 50 * np.sin(2 * np.pi * np.arange(slots) / 96)).round().astype(int)
    frame = pd.DataFrame({"observed_at": observed_at, "demand": demand})

    path = _train_station_horizon_model("TEST", 1, frame)
    model = joblib.load(path)

    assert isinstance(model, RatioTargetModel)
    assert model.n_features_in_ == 8
    last = 1000.0  # 5x the training maximum
    from app.features import temporal_features
    surge_row = [last, last, last, last] + temporal_features(observed_at[-1] + pd.Timedelta(minutes=15))
    prediction = float(model.predict([surge_row])[0])
    assert prediction > 400  # a level model trained on <=200 could never say this
