"""Detección de drift por estación y reentrenamiento automático.

El collector guarda en "Temp" cada observación nueva junto con la
predicción que el modelo vigente le habría dado. La señal de drift es la
accuracy (la misma métrica del proyecto, 1 - WAPE) de esa estación, medida
solo sobre sus últimos RECENT_CHECKS puntos con predicción registrada (no
sobre una ventana de tiempo fija ni sobre todo lo acumulado en "Temp" desde
el último reentrenamiento) - así el disparador reacciona a cómo está
funcionando el modelo *ahora mismo* en vez de diluirse con un histórico
largo que puede mezclar tramos buenos y malos, y no se queda ciego si el
stream se ralentiza o se detiene (una ventana de tiempo fija podría no
llegar nunca a acumular puntos suficientes; un conteo fijo de chequeos sí).

Cuando una estación dispara el drift, el reentrenamiento SIEMPRE conserva
todo el histórico: los puntos nuevos de "Temp" se incorporan a
"Original Data" y nunca se descarta nada. (Antes hubo dos variantes que sí
recortaban historia - una prueba de Kolmogorov-Smirnov que decidía
conservar todo o recortar, y luego un reemplazo 1:1 fijo sin esa prueba -
ambas se quitaron el 2026-09-24 después de que una comparación directa
mostrara que un modelo entrenado con todo el histórico predice mejor que
uno entrenado solo con una ventana reciente, en 12/12 estaciones. No
reintroducir un recorte de historia sin repetir esa comparación.)
"""

import math
import os
import statistics

import joblib
import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor

from app import context as context_module
from app.db import connection
from app.features import temporal_features

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "models/xgboost")
MODEL_BUCKET = "models"
LAGS = (1, 2, 4, 96)
# Direct multi-horizon forecasting: one model per +15/+30/+45/+60min target
# instead of one model recursively fed forward. Each horizon's lag features
# are anchored to the same reference point (data_cutoff / the observation
# itself) and are always real observed values - never another horizon's own
# prediction - so error can't compound across horizons the way it did with
# the old recursive approach. See submit_xgboost.py's predict_cycle_targets.
HORIZONS = (1, 2, 3, 4)

ACCURACY_THRESHOLD = 0.85
# Back to 0.85 on 2026-10-01 (it was 0.9 from 2026-09-29). It is now the CAP
# of each station's own adaptive threshold (see station_adaptive_thresholds
# below) and the fallback for a station without enough history to have a
# baseline of its own - not one bar applied identically to every station.
# Why not a single fixed bar: the 4-point accuracy window is very noisy
# (std ~7pp per station on live data), so at 0.90 it flagged ~74% of
# windows for a retrain and at 0.85 still ~45%, mostly noise from inherently
# jumpy stations, while a steady 89% station sliding to 86% was never
# flagged at all.

# Per-station adaptive threshold: a station is flagged when its last
# RECENT_CHECKS accuracy falls ADAPTIVE_DROP below its OWN baseline accuracy
# (the ADAPTIVE_BASELINE_CHECKS real pairs before that window, ~2 days),
# clamped to [ADAPTIVE_MIN_THRESHOLD, ACCURACY_THRESHOLD]. The floor stops a
# long collapse from dragging the bar down with it forever; sustained shifts
# are also caught independently by the Page-Hinkley alarm, which bypasses
# this gate entirely. On live data (Sep 9-18, 12 stations) this flagged ~23%
# of windows vs ~45% for a fixed 0.85.
ADAPTIVE_BASELINE_CHECKS = 192
ADAPTIVE_MIN_BASELINE_CHECKS = 48
ADAPTIVE_DROP = 0.05
ADAPTIVE_MIN_THRESHOLD = 0.75
# Count-based window, not a time window: a station's drift signal is its
# accuracy over its own last RECENT_CHECKS real prediction/actual pairs,
# whichever tick they landed on. MIN_DATAPOINTS requires the window to be
# full before trusting the signal - a station with only 1-2 checks so far
# shouldn't be judged (or retrained) off a handful of points.
RECENT_CHECKS = 4
MIN_DATAPOINTS = RECENT_CHECKS

# A drifted station only actually retrains once at least this many new rows
# have piled up in "Temp" - not merely "more than zero". has_pending_data()
# (below) deliberately keeps re-running the drift check as long as Temp has
# anything at all in it, for any station, so a stalled feed never blocks
# retraining forever (see its docstring) - but without this second gate,
# that meant a station already below ACCURACY_THRESHOLD got fully retrained
# (4 horizons, 4 Supabase uploads) on almost every ~30min collector cycle
# off as few as 1-2 new rows against a ~5000-row history: real cost, no
# real change to the model, and the actual cause of the Supabase egress
# spike from 2026-09-26. This does not reintroduce the old "blocked
# forever" bug: as long as new data keeps trickling in at all, the count
# keeps growing and eventually clears this bar - it just stops spending a
# full retrain on every single trickle.
MIN_NEW_FOR_RETRAIN = 20

# Chosen per station via walk-forward CV grid search restricted to the train
# split (2026-09-25): every station's own CV picked a smaller/slower model
# than the previous shared 400-estimator/40-leaf/reg_lambda=1.0 default, but
# reg_lambda itself split station by station (1.0 vs 3.0), and a single
# shared config underperformed each station's own best config on held-out
# test data - so each station keeps its own winning combination rather than
# being forced onto one global default. Mean held-out test accuracy across
# the 12 stations: 85.07% -> 85.37%, with 8/12 stations improving.
DEFAULT_MODEL_PARAMS = {"learning_rate": 0.05, "n_estimators": 400, "max_leaves": 40, "reg_lambda": 1.0}
STATION_MODEL_PARAMS = {
    "02300": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "03000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "05000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "05100": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "06000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "06111": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "07105": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "07107": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "07111": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "09000": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
    "09122": {"learning_rate": 0.05, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 1.0},
    "10009": {"learning_rate": 0.03, "n_estimators": 200, "max_leaves": 20, "reg_lambda": 3.0},
}


def _model_params(station_id: str) -> dict:
    """A station outside the tuned set (e.g. a newly added one) falls back to
    the old shared default rather than crashing or silently getting some
    other station's config."""
    return STATION_MODEL_PARAMS.get(station_id, DEFAULT_MODEL_PARAMS)


# lag_672 (a full week back) was dropped from LAGS entirely on 2026-09-27,
# for every station and horizon. It had already been removed or down-
# weighted for a handful of pairs (a per-pair NO_LAG672_MODELS/
# SOFT_DEEMPHASIZE_LAG672_MODELS set, tested 2026-09-26 via two independent
# chronological folds), but re-running that same two-fold test with the data
# accumulated since then showed full removal now clears the improvement bar
# (>=0.3pp on both folds) for far more pairs than the original 7 - roughly
# 15 more, including most of 05100 (whose known multi-hour demand collapses
# make "what happened this same time last week" actively misleading) - and
# a down-weighted lag_672 never once beat full removal head-to-head anywhere
# it was tried. Rather than maintain an ever-growing per-pair exception
# list, lag_672 is now simply out of the shared LAGS tuple for every model.
# A few pairs (e.g. 09000, 07105 h4) individually preferred keeping it, but
# the loss there is small and a single consistent feature set for every
# model removes an entire class of bug: the feature-shape mismatches from
# some models expecting 8 features and others 9 caused a real ~48min
# submissions outage on 2026-09-26 (see submit_xgboost.py/collector.py -
# they used to need lags_for() specifically to avoid this).
#
# Inference-time bias correction, applied on top of a station's raw model
# output at submission time - not a training change, and never touches what
# gets stored as "prediction" in Temp/Original Data (that stays the raw
# model's own output, so drift.py's accuracy signal and this correction's
# own residual history never feed back into each other).
#
# Chosen via a per-station grid search over alpha (EWMA decay) x damping
# (how much of the tracked bias to actually apply), picking whichever
# config maximizes OVERALL held-out-style accuracy on that station's full
# history (not just its most recent/noisiest stretch) - a config biased
# toward only recent data reacts fast but chases noise on an ordinary day;
# one biased toward all of history barely reacts at all. A candidate must
# beat DEFAULT_EWMA_PARAMS by at least MIN_EWMA_IMPROVEMENT_PP before a
# station gets its own override, for the same reason DEFAULT_MODEL_PARAMS
# has one: on this search (2026-09-26, all 12 stations, including 05100's
# real multi-hour demand collapse), every station's best candidate beat the
# default by less than that margin - even 05100, whose collapse looked like
# it needed a much more aggressive config when evaluated in isolation on
# just that ~24-row window, but the gain nearly vanishes once judged against
# the whole dataset instead. So every station uses the shared default for
# now; STATION_EWMA_PARAMS exists so a future station-specific override is a
# one-line addition once a real margin actually turns up.
DEFAULT_EWMA_PARAMS = {"alpha": 0.2, "damping": 0.5}
MIN_EWMA_IMPROVEMENT_PP = 0.5
STATION_EWMA_PARAMS: dict[str, dict] = {}

# How much of the boosted-regime correction (see the Page-Hinkley-adaptive
# EWMA section below) carries over to each forecast horizon. HORIZON 1
# (+15min, the same nowcast the boost is tuned on) always gets the FULL
# boosted bias; horizons 2-4 blend it back toward the plain/regular bias,
# with the blend front-loaded by EWMA_DECAY_POWER so it's already mostly
# regular by +30min instead of fading in a straight line. Backtested
# 2026-09-28 on all 12 stations over a held-out tail that includes 05100's
# real collapse: applying the SAME fully-boosted bias to every horizon (what
# this pipeline did before) wins big at +15min but actively overcorrects
# from +30min on - a flat linear decay (power=1) closed most but not all of
# that gap; power=8 (already ~96% regular by +30min, ~99.98% by +45min)
# matched or beat plain fixed EWMA at every horizon while keeping virtually
# the entire +15min win, the best result of every variant tried (raw
# uncorrected, flat CUSUM, flat Page-Hinkley, linear-decay CUSUM/Page-
# Hinkley, and a more sensitive alarm threshold - which raised the +15min
# win further but caused false-positive boosts on stations that had never
# alarmed before, netting worse overall).
EWMA_DECAY_POWER = 8


def _horizon_decay_weight(horizon: int) -> float:
    """1.0 at the shortest horizon (full boost), 0.0 at the longest (fully
    regressed to the plain/regular bias), front-loaded per EWMA_DECAY_POWER
    in between. Independent of which station or correction is being blended -
    pure function of horizon, so it's cheap to call per prediction target."""
    h_min, h_max = min(HORIZONS), max(HORIZONS)
    if h_max == h_min:
        return 1.0
    return ((h_max - horizon) / (h_max - h_min)) ** EWMA_DECAY_POWER


# Re-tuned automatically at every retrain (see _update_ewma_params_state):
# the grid below is searched on the station's FULL recorded history, and a
# candidate only replaces DEFAULT_EWMA_PARAMS when it wins by at least
# MIN_EWMA_IMPROVEMENT_PP - the same margin-checked, full-history discipline
# as the offline search above, so a noisy window can't re-roll the config.
# Walk-forward backtest (tune on history before each fold, score on the
# unseen part, 12 stations): fold 70-85% no station cleared the margin
# (0.00pp), last-15% fold +0.35pp station mean (02300 +2.06, 05000 +1.48,
# 05100 +0.69), no station worse.
EWMA_TUNE_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6)
EWMA_TUNE_DAMPINGS = (0.0, 0.25, 0.5, 0.75, 1.0)
EWMA_TUNE_MIN_ROWS = 200


def _ewma_params(station_id: str) -> dict:
    """The (alpha, damping) in force for this station: the value stored by its
    last retrain (collector_state key "ewma_params:{station}", "alpha,damping")
    if any, else a STATION_EWMA_PARAMS override, else DEFAULT_EWMA_PARAMS. One
    indexed single-row lookup - cheap, called once per station per cycle."""
    with connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM collector_state WHERE state_key = %s",
            (f"ewma_params:{station_id}",),
        ).fetchone()
    if row is not None:
        try:
            alpha, damping = (float(x) for x in str(row[0]).split(","))
            return {"alpha": alpha, "damping": damping}
        except ValueError:
            pass
    return STATION_EWMA_PARAMS.get(station_id, DEFAULT_EWMA_PARAMS)


# An EWMA's weight on a residual n steps back decays geometrically as
# (1-alpha)^n, so a station's FULL history (tens of thousands of rows and
# growing) contributes essentially nothing beyond its most recent ~100-150
# points at alpha=0.2 - fetching all of it anyway (as station_bias_components
# used to, once per station per submission cycle, every ~5min) was one of the
# largest sources of Supabase DB egress. _ewma_lookback_rows() picks a row
# count small enough to matter for egress but large enough that the result
# is numerically indistinguishable (within EWMA_TRUNCATION_EPSILON) from
# using the complete, unbounded history - derived from that station's own
# alpha rather than hardcoded, so it stays correct even if a future
# STATION_EWMA_PARAMS override ever picks a much slower-decaying alpha.
EWMA_TRUNCATION_EPSILON = 1e-9


def _ewma_lookback_rows(alpha: float) -> int:
    return max(50, math.ceil(math.log(EWMA_TRUNCATION_EPSILON) / math.log(1 - alpha)))


def station_bias_components(station_id: str) -> tuple[float, float]:
    """Fetches this station's recent residual history ONCE and returns
    (regular_bias, boosted_bias): the plain fixed-EWMA correction and the
    Page-Hinkley-boosted correction, both as of right now. station_horizon_bias
    below blends between these two per forecast horizon - callers handling
    multiple horizons for the same station in one cycle (submit_xgboost.py)
    should call this once per station and blend per horizon locally with
    blend_horizon_bias, so a submission cycle still costs exactly one DB read
    per station regardless of how many horizons it predicts.

    Tracks a causal (no-lookahead) exponentially-weighted average of that
    station's own (actual - predicted) residuals, using that station's most
    recent prediction/actual pairs across "Original Data" and "Temp" - the
    same source drift.py's own accuracy signal reads - in chronological
    order. Reacts within one collector cycle to a real, sustained miss
    (like 05100's collapse) instead of waiting for a full drift-triggered
    retrain, which can take hours or - while the upstream feed is stalled -
    may not be able to fire at all.

    Only the last _ewma_lookback_rows(alpha) rows are fetched (DESC, then
    reversed back to chronological order below) - see that function for why
    this doesn't change the result in any way that matters. A station with
    the Page-Hinkley boost currently on (see _is_page_hinkley_enabled) needs
    a longer lookback than that - PH_LOOKBACK_ROWS - since it also has to
    cover the rolling std window and the longest cooldown an alarm could
    still be "inside" of.
    """
    params = _ewma_params(station_id)
    alpha = params["alpha"]
    ph_enabled = _is_page_hinkley_enabled(station_id)
    lookback = max(_ewma_lookback_rows(alpha), PH_LOOKBACK_ROWS) if ph_enabled else _ewma_lookback_rows(alpha)
    with connection() as conn:
        rows = conn.execute(
            """SELECT observed_at, demand, prediction FROM (
                   SELECT observed_at, demand, prediction FROM "Original Data"
                   WHERE station_id = %s AND prediction IS NOT NULL
                   UNION ALL
                   SELECT observed_at, demand, prediction FROM "Temp"
                   WHERE station_id = %s AND prediction IS NOT NULL
               ) recent
               ORDER BY observed_at DESC
               LIMIT %s""",
            (station_id, station_id, lookback),
        ).fetchall()
    if not rows:
        return 0.0, 0.0
    rows.reverse()  # DESC from the query -> chronological, oldest first
    if not ph_enabled:
        bias = 0.0
        for _, demand, prediction in rows:
            residual = float(demand) - float(prediction)
            bias = alpha * residual + (1 - alpha) * bias
        regular = params["damping"] * bias
        return regular, regular  # no boost available for this station - both equal
    return _page_hinkley_dual_bias(rows, params, PH_ADAPTIVE_PARAMS)


def blend_horizon_bias(regular_bias: float, boosted_bias: float, horizon: int) -> float:
    """Pure, no-I/O blend of the two station_bias_components() outputs for a
    given forecast horizon - see _horizon_decay_weight and EWMA_DECAY_POWER's
    docstring/comment above for why horizon 1 gets the full boosted_bias and
    later horizons decay fast back toward regular_bias."""
    weight = _horizon_decay_weight(horizon)
    return weight * boosted_bias + (1 - weight) * regular_bias


def station_horizon_bias(station_id: str, horizon: int) -> float:
    """The correction to ADD to a station's raw model prediction for one
    specific forecast horizon, right now. Convenience single-call wrapper
    around station_bias_components + blend_horizon_bias for callers that
    only need one (station, horizon) pair; a caller needing multiple
    horizons for the same station (submit_xgboost.py) should call
    station_bias_components once and blend_horizon_bias per horizon instead,
    to avoid repeating the DB read."""
    regular_bias, boosted_bias = station_bias_components(station_id)
    return blend_horizon_bias(regular_bias, boosted_bias, horizon)


# --- Page-Hinkley-adaptive EWMA (regime-shift boost) ------------------------
#
# On top of the fixed EWMA above, a station with the boost currently on (per
# _is_page_hinkley_enabled) also runs a two-sided Page-Hinkley change-point
# test on its standardized residuals - a classical sequential change-
# detection test (the same family as CUSUM, which this replaced on
# 2026-09-28: both track a cumulative sum of standardized residuals and
# alarm once it drifts too far, but Page-Hinkley compares against its own
# running minimum/maximum rather than resetting to a hard zero floor).
# Backtested head-to-head against CUSUM on the same 12 stations and held-out
# window: Page-Hinkley matched or slightly exceeded CUSUM's win on 05100's
# real collapse, and did so WITHOUT the small regressions CUSUM caused on
# 09000/09122 - a strict improvement, not just a different tradeoff.
#
# While no change point has fired, its behavior is identical to the plain
# fixed EWMA above. Once Page-Hinkley fires (a real, sustained deviation -
# not routine noise), the station switches to a more reactive (alpha,
# damping) pair for a fixed cooldown window, then reverts. Built to react
# fast to a genuine regime shift (like 05100's multi-hour demand collapses)
# without the extra reactivity permanently hurting the other, well-behaved
# stations.
#
# Originally backtested as CUSUM on 2026-09-27, 3 independent chronological
# folds never touched during tuning: a per-station-tuned version overfit
# badly (~500 train points isn't enough to pin down 4 free params per
# station - different folds picked different "best" configs for the same
# station, and results didn't hold out-of-fold, including 05100's own).
# A single SHARED config, chosen by pooling training accuracy across all 12
# stations at once (so the calm majority regularizes the search away from
# chasing any one station's noise), was stable across folds and delivered a
# consistent, large gain on 05100 (+1.21 / +6.52 / +0.06pp vs. fixed EWMA
# across the 3 folds), with only two stations (09000, 09122) regressing.
# Re-pooling the search on just those two regressors, to try to fix them
# specifically, made both WORSE, not better - shrinking the pool to just the
# problem stations removes the regularizing effect that made pooling work in
# the first place. So the boost config itself stays a single SHARED constant,
# never re-tuned per station (that's what overfit in the first place) - the
# only thing that varies per station is whether it's turned on at all.
#
# PH_ENABLED_STATIONS below is only the bootstrap default (the 3 stations
# with a real, fold-consistent gain in that first backtest: 05100 large,
# 06000/07111 small but never negative). Which stations actually have it on
# is re-decided every time that station retrains, by
# _update_page_hinkley_enabled_state below - so a station that starts
# showing 05100-like behavior later earns the boost the next time it
# retrains, without needing a code change, and one that stops benefiting
# loses it just as automatically.
PH_DELTA = 0.5  # slack: ignore drift smaller than this many std, in either direction
PH_STD_WINDOW = 96  # ~1 day at 15min cadence, causal rolling std for standardizing residuals
PH_ADAPTIVE_PARAMS = {"boost_alpha": 0.4, "boost_damping": 0.7, "cooldown": 80, "lambda": 25.0}
PH_ENABLED_STATIONS: frozenset[str] = frozenset({"05100", "06000", "07111"})
# A candidate must beat the plain fixed EWMA by at least this much (on that
# station's own full recorded prediction history) before retraining flips it
# on - same margin and rationale as MIN_EWMA_IMPROVEMENT_PP above: a result
# under a real margin is noise, not a genuine, actionable difference.
MIN_PH_IMPROVEMENT_PP = 0.5
# Long enough for the rolling std window plus the longest cooldown an alarm
# could still be "inside" of, plus slack - independent of (and much larger
# than) the plain-EWMA lookback above, which decays away almost all of a
# station's history within ~50-150 rows and would be too short here.
PH_LOOKBACK_ROWS = PH_STD_WINDOW + PH_ADAPTIVE_PARAMS["cooldown"] + 50


def _page_hinkley_step(residual: float, state: dict, boost_params: dict) -> bool:
    """One causal Page-Hinkley update, mutating `state` (keys: m_pos, M_pos,
    m_neg, M_neg, cooldown_left, window) in place. Returns True the instant
    an alarm fires on this exact step. Shared by every function below that
    needs the Page-Hinkley state machine, so the fiddly two-sided-test
    bookkeeping exists in exactly one place instead of being retyped (and
    able to drift out of sync) in each caller.

    m_pos/m_neg are running cumulative sums of the standardized residual
    (minus a small slack, PH_DELTA) in each direction; M_pos/M_neg track
    their all-time running MINIMUM (never reset except on alarm) - the
    alarm condition is how far the cumulative sum has climbed since its own
    best point, which is what distinguishes Page-Hinkley from CUSUM's
    reset-to-zero-floor test above it replaced.
    """
    window = state["window"]
    std = statistics.pstdev(window) if len(window) >= 10 else None
    alarmed = False
    if std and std > 1e-6:
        z = residual / std
        state["m_pos"] += z - PH_DELTA
        state["M_pos"] = min(state["M_pos"], state["m_pos"])
        ph_pos = state["m_pos"] - state["M_pos"]
        state["m_neg"] += -z - PH_DELTA
        state["M_neg"] = min(state["M_neg"], state["m_neg"])
        ph_neg = state["m_neg"] - state["M_neg"]
        if ph_pos > boost_params["lambda"] or ph_neg > boost_params["lambda"]:
            state["cooldown_left"] = boost_params["cooldown"]
            state["m_pos"] = state["M_pos"] = 0.0
            state["m_neg"] = state["M_neg"] = 0.0
            alarmed = True
        elif state["cooldown_left"] > 0:
            state["cooldown_left"] -= 1
    elif state["cooldown_left"] > 0:
        state["cooldown_left"] -= 1
    window.append(residual)
    if len(window) > PH_STD_WINDOW:
        window.pop(0)
    return alarmed


def _new_page_hinkley_state() -> dict:
    return {"m_pos": 0.0, "M_pos": 0.0, "m_neg": 0.0, "M_neg": 0.0, "cooldown_left": 0, "window": []}


def _page_hinkley_dual_bias(rows: list[tuple], normal_params: dict, boost_params: dict) -> tuple[float, float]:
    """Causal (no-lookahead) walk over `rows` (chronological), returning
    (regular_bias, boosted_bias) as of the LAST row in one pass - the plain
    fixed-EWMA bias (as if Page-Hinkley never boosted at all) and the
    Page-Hinkley-boosted bias, computed together so station_bias_components
    needs only one walk over the fetched rows regardless of how many
    horizons the caller ends up blending."""
    normal_alpha, normal_damping = normal_params["alpha"], normal_params["damping"]
    boost_alpha, boost_damping = boost_params["boost_alpha"], boost_params["boost_damping"]

    regular_raw_bias = 0.0
    boosted_raw_bias = 0.0
    boosted_damping = normal_damping
    state = _new_page_hinkley_state()

    for _, demand, prediction in rows:
        residual = float(demand) - float(prediction)
        regular_raw_bias = normal_alpha * residual + (1 - normal_alpha) * regular_raw_bias

        boosted = state["cooldown_left"] > 0
        alpha = boost_alpha if boosted else normal_alpha
        boosted_damping = boost_damping if boosted else normal_damping
        boosted_raw_bias = alpha * residual + (1 - alpha) * boosted_raw_bias
        _page_hinkley_step(residual, state, boost_params)

    return normal_damping * regular_raw_bias, boosted_damping * boosted_raw_bias


def _page_hinkley_recently_alarmed(rows: list[tuple]) -> bool:
    """Detection only - never changes any prediction or correction. True if a
    real Page-Hinkley change point fired within the last `cooldown` points of
    this station's own residual history, i.e. a genuine, sustained regime
    shift is still "fresh". Safe to run for EVERY station, including ones
    outside PH_ENABLED_STATIONS or a brand-new station never seen before, as
    a faster-reacting companion to the RECENT_CHECKS/ACCURACY_THRESHOLD drift
    gate below: that gate needs a full 4-point accuracy window to average
    below the bar before it fires, while a real regime shift shows up in
    Page-Hinkley's statistical test point by point, well before that average
    would catch up. This only ever pulls a station INTO consideration
    earlier - the separate MIN_NEW_FOR_RETRAIN row-count floor in
    check_and_retrain still applies unconditionally afterward, so a real
    alarm with no new data yet never spends a wasted retrain."""
    state = _new_page_hinkley_state()
    for _, demand, prediction in rows:
        residual = float(demand) - float(prediction)
        _page_hinkley_step(residual, state, PH_ADAPTIVE_PARAMS)
    return state["cooldown_left"] > 0


def _station_has_fresh_page_hinkley_alarm(station_id: str) -> bool:
    """Whether _page_hinkley_recently_alarmed(...) is true for this station
    right now, fetching just enough recent history to evaluate it. Called
    every cycle for every station that isn't already flagged as accuracy-
    drifted (see check_and_retrain) - a bounded (PH_LOOKBACK_ROWS-row) query,
    not the kind of unbounded scan this file's other egress fixes were
    about."""
    with connection() as conn:
        rows = conn.execute(
            """SELECT observed_at, demand, prediction FROM (
                   SELECT observed_at, demand, prediction FROM "Original Data"
                   WHERE station_id = %s AND prediction IS NOT NULL
                   UNION ALL
                   SELECT observed_at, demand, prediction FROM "Temp"
                   WHERE station_id = %s AND prediction IS NOT NULL
               ) recent
               ORDER BY observed_at DESC
               LIMIT %s""",
            (station_id, station_id, PH_LOOKBACK_ROWS),
        ).fetchall()
    if len(rows) < 10:
        return False
    rows.reverse()
    return _page_hinkley_recently_alarmed(rows)


def _accuracy_score(abs_err: float, abs_dem: float) -> float:
    return max(0.0, 1 - abs_err / max(abs_dem, 1.0))


def _fixed_ewma_accuracy(rows: list[tuple], normal_params: dict) -> float:
    """WAPE-based accuracy (matches station_accuracy_stats's own formula) of
    the plain fixed EWMA over `rows` (chronological) - the baseline that
    _page_hinkley_would_help compares the boosted regime against."""
    alpha, damping = normal_params["alpha"], normal_params["damping"]
    raw_bias = 0.0
    abs_err = abs_dem = 0.0
    for _, demand, prediction in rows:
        demand, prediction = float(demand), float(prediction)
        corrected = max(0.0, prediction + damping * raw_bias)
        abs_err += abs(demand - corrected)
        abs_dem += abs(demand)
        raw_bias = alpha * (demand - prediction) + (1 - alpha) * raw_bias
    return _accuracy_score(abs_err, abs_dem)


def _page_hinkley_adaptive_accuracy(rows: list[tuple], normal_params: dict, boost_params: dict) -> float:
    """Same idea as _fixed_ewma_accuracy, but with the Page-Hinkley-boosted
    regime layered on top - the number _page_hinkley_would_help compares
    against it. Always uses the FULL boosted bias at every row (weight=1),
    matching horizon 1 - the horizon the boost is actually tuned and
    evaluated on; see EWMA_DECAY_POWER's comment for why later horizons
    deliberately get less of it."""
    normal_alpha, normal_damping = normal_params["alpha"], normal_params["damping"]
    boost_alpha, boost_damping = boost_params["boost_alpha"], boost_params["boost_damping"]
    raw_bias = 0.0
    abs_err = abs_dem = 0.0
    state = _new_page_hinkley_state()
    for _, demand, prediction in rows:
        demand, prediction = float(demand), float(prediction)
        boosted = state["cooldown_left"] > 0
        alpha = boost_alpha if boosted else normal_alpha
        damping = boost_damping if boosted else normal_damping
        corrected = max(0.0, prediction + damping * raw_bias)
        abs_err += abs(demand - corrected)
        abs_dem += abs(demand)
        residual = demand - prediction
        raw_bias = alpha * residual + (1 - alpha) * raw_bias
        _page_hinkley_step(residual, state, boost_params)
    return _accuracy_score(abs_err, abs_dem)


def _fetch_prediction_history(station_id: str) -> list[tuple]:
    """The station's COMPLETE (observed_at, demand, prediction) history,
    chronological. Only for once-per-retrain callers - never a hot path."""
    with connection() as conn:
        return conn.execute(
            """SELECT observed_at, demand, prediction FROM "Original Data"
               WHERE station_id = %s AND prediction IS NOT NULL
               UNION ALL
               SELECT observed_at, demand, prediction FROM "Temp"
               WHERE station_id = %s AND prediction IS NOT NULL
               ORDER BY observed_at""",
            (station_id, station_id),
        ).fetchall()


def _tune_ewma_params(station_id: str) -> dict | None:
    """Best (alpha, damping) on this station's full history, or None when
    there isn't enough history to trust a search. Falls back to
    DEFAULT_EWMA_PARAMS unless a candidate beats it by MIN_EWMA_IMPROVEMENT_PP."""
    rows = _fetch_prediction_history(station_id)
    if len(rows) < EWMA_TUNE_MIN_ROWS:
        return None
    best, best_acc = DEFAULT_EWMA_PARAMS, _fixed_ewma_accuracy(rows, DEFAULT_EWMA_PARAMS)
    default_acc = best_acc
    for alpha in EWMA_TUNE_ALPHAS:
        for damping in EWMA_TUNE_DAMPINGS:
            candidate = {"alpha": alpha, "damping": damping}
            acc = _fixed_ewma_accuracy(rows, candidate)
            if acc > best_acc:
                best, best_acc = candidate, acc
    if best_acc - default_acc < MIN_EWMA_IMPROVEMENT_PP / 100:
        return DEFAULT_EWMA_PARAMS
    return best


def _update_ewma_params_state(station_id: str) -> dict | None:
    """Re-tunes this station's EWMA (alpha, damping) once per retrain and
    persists it in collector_state ("ewma_params:{station}"), same pattern as
    the Page-Hinkley flag. Leaves the stored value alone when there is too
    little history. Returns the params now in force, or None if unchanged."""
    tuned = _tune_ewma_params(station_id)
    if tuned is None:
        return None
    with connection() as conn:
        conn.execute(
            """INSERT INTO collector_state (state_key, cursor_value, updated_at)
               VALUES (%s, %s, now())
               ON CONFLICT (state_key) DO UPDATE
               SET cursor_value = EXCLUDED.cursor_value, updated_at = now()""",
            (f"ewma_params:{station_id}", f"{tuned['alpha']},{tuned['damping']}"),
        )
    return tuned


def _page_hinkley_would_help(station_id: str) -> bool | None:
    """Whether the single SHARED PH_ADAPTIVE_PARAMS config would beat the
    plain fixed EWMA by at least MIN_PH_IMPROVEMENT_PP on this station's own
    full recorded (demand, prediction) history - never re-tuning the boost's
    own numbers per station (that's what overfit in the 2026-09-27
    per-station backtest), only ever asking a yes/no question about the
    fixed config. Returns None (meaning "don't change anything") when there
    isn't enough history yet to trust the comparison.

    Reads the station's COMPLETE history, deliberately unbounded: this only
    runs once per retrain (see _update_page_hinkley_enabled_state), and a
    retrain already reads this station's entire history anyway to build the
    training frame, so this adds no new class of cost - unlike
    station_bias_components above, which runs every ~5min submission cycle
    and is bounded for exactly that reason.
    """
    rows = _fetch_prediction_history(station_id)
    if len(rows) < PH_LOOKBACK_ROWS:
        return None
    normal_params = _ewma_params(station_id)
    fixed_acc = _fixed_ewma_accuracy(rows, normal_params)
    adaptive_acc = _page_hinkley_adaptive_accuracy(rows, normal_params, PH_ADAPTIVE_PARAMS)
    return (adaptive_acc - fixed_acc) >= (MIN_PH_IMPROVEMENT_PP / 100)


def _update_page_hinkley_enabled_state(station_id: str) -> None:
    """Re-decides whether station_id gets the Page-Hinkley-adaptive EWMA
    boost, called once per retrain (see check_and_retrain), and persists the
    decision in "collector_state" (the same generic key/value table
    collector.py already uses for its own cursor) under the key
    "page_hinkley_enabled:{station_id}" - each GitHub Actions run is a fresh
    process, so the decision has to live in the DB to survive between
    retrains, not in memory.

    Leaves the stored value untouched when _page_hinkley_would_help can't yet
    form an opinion (not enough history) - a station keeps whatever its last
    real decision was (or the PH_ENABLED_STATIONS bootstrap default, if it's
    never had one) rather than being silently switched off for lack of data.
    """
    would_help = _page_hinkley_would_help(station_id)
    if would_help is None:
        return
    with connection() as conn:
        conn.execute(
            """INSERT INTO collector_state (state_key, cursor_value, updated_at)
               VALUES (%s, %s, now())
               ON CONFLICT (state_key) DO UPDATE
               SET cursor_value = EXCLUDED.cursor_value, updated_at = now()""",
            (f"page_hinkley_enabled:{station_id}", "true" if would_help else "false"),
        )


def _is_page_hinkley_enabled(station_id: str) -> bool:
    """Whether station_id has the Page-Hinkley-adaptive EWMA boost on right
    now. Reads collector_state's stored per-station decision (a single
    indexed row lookup - cheap, unlike the unbounded scans this file's
    egress fixes were about) and falls back to the PH_ENABLED_STATIONS
    bootstrap default for a station that hasn't retrained since this
    mechanism was added and so has no stored decision yet."""
    with connection() as conn:
        row = conn.execute(
            "SELECT cursor_value FROM collector_state WHERE state_key = %s",
            (f"page_hinkley_enabled:{station_id}",),
        ).fetchone()
    if row is None:
        return station_id in PH_ENABLED_STATIONS
    return row[0] == "true"


def has_pending_data() -> bool:
    """Whether "Temp" currently holds any unpromoted observations at all.

    Used as the gate before running drift checks, in place of "did this
    exact collector run insert a new row". A stalled upstream feed can leave
    a drifted station's data sitting in "Temp" indefinitely without any run
    ever inserting a fresh row again - gating on "did this run insert
    something new" then blocks drift/retrain forever even though the
    station's already-known-bad data is sitting right there, unprocessed.
    Gating on "is there anything in Temp at all" still skips redundant
    checks once Temp is genuinely empty (e.g. every station was just
    retrained and promoted), but keeps checking as long as there's
    unprocessed data to evaluate, regardless of whether new rows arrived
    this exact cycle.

    This is a cheap, global (not per-station) short-circuit only - it does
    not by itself decide which stations actually get retrained. That
    decision still runs per station inside check_and_retrain(), which
    additionally requires MIN_NEW_FOR_RETRAIN new rows for that specific
    station before spending a real retrain on it.
    """
    with connection() as conn:
        return conn.execute('SELECT 1 FROM "Temp" LIMIT 1').fetchone() is not None


def station_accuracy_stats() -> pd.DataFrame:
    """Accuracy and count per station over each station's own last RECENT_CHECKS
    predicted/actual pairs.

    Accuracy = max(0, 1 - WAPE), i.e. 1 - sum(|demand - prediction|) / sum(|demand|),
    the project's accuracy metric.

    A count-based window (not a time window) per station: a station that has
    fallen behind or whose data is arriving irregularly still gets judged on
    its own most recent real checks, rather than a fixed clock window that
    could span very different amounts of real data station to station.

    A retrain empties "Temp" for the stations it just touched (their rows move
    into "Original Data" - see _promote_temp_to_original). Right after that, a
    just-retrained station's own "Temp" history can be shorter than the full
    window, even though the data itself still exists - it just moved. So this
    reads both tables and lets whichever one holds each point fill it in;
    it's the same rows, just possibly in their new location.
    """
    with connection() as conn:
        rows = conn.execute(
            """WITH combined AS (
                   SELECT station_id, demand, prediction, observed_at FROM "Temp"
                   UNION ALL
                   SELECT station_id, demand, prediction, observed_at FROM "Original Data"
               ),
               ranked AS (
                   SELECT station_id, demand, prediction,
                          row_number() OVER (PARTITION BY station_id ORDER BY observed_at DESC) AS rn
                   FROM combined
                   WHERE prediction IS NOT NULL
               )
               SELECT station_id, demand, prediction FROM ranked WHERE rn <= %s""",
            (RECENT_CHECKS,),
        ).fetchall()
    columns = ["station_id", "accuracy", "count"]
    if not rows:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame(rows, columns=["station_id", "demand", "prediction"])
    data["abs_error"] = (data["demand"] - data["prediction"]).abs()
    data["abs_demand"] = data["demand"].abs()
    stats = data.groupby("station_id").agg(
        abs_error=("abs_error", "sum"), abs_demand=("abs_demand", "sum"), count=("demand", "size")
    )
    stats["accuracy"] = (1 - stats["abs_error"] / stats["abs_demand"].clip(lower=1)).clip(lower=0)
    return stats.reset_index()[columns]


def station_baseline_accuracy() -> dict[str, float]:
    """Each station's accuracy over the ADAPTIVE_BASELINE_CHECKS real
    prediction/actual pairs that come BEFORE its most recent RECENT_CHECKS
    (so the baseline never includes the window being judged). Stations with
    fewer than ADAPTIVE_MIN_BASELINE_CHECKS such pairs are omitted and fall
    back to the fixed ACCURACY_THRESHOLD. Aggregated in SQL - 12 small rows
    come back, not the history itself."""
    with connection() as conn:
        rows = conn.execute(
            """WITH combined AS (
                   SELECT station_id, demand, prediction, observed_at FROM "Temp"
                   UNION ALL
                   SELECT station_id, demand, prediction, observed_at FROM "Original Data"
               ),
               baseline_ranked AS (
                   SELECT station_id, demand, prediction,
                          row_number() OVER (PARTITION BY station_id ORDER BY observed_at DESC) AS rn
                   FROM combined
                   WHERE prediction IS NOT NULL
               )
               SELECT station_id, SUM(ABS(demand - prediction)), SUM(ABS(demand)), COUNT(*)
               FROM baseline_ranked
               WHERE rn > %s AND rn <= %s
               GROUP BY station_id""",
            (RECENT_CHECKS, RECENT_CHECKS + ADAPTIVE_BASELINE_CHECKS),
        ).fetchall()
    baselines = {}
    for station_id, abs_error, abs_demand, count in rows:
        if count >= ADAPTIVE_MIN_BASELINE_CHECKS:
            baselines[station_id] = max(0.0, 1 - float(abs_error) / max(float(abs_demand), 1.0))
    return baselines


def station_adaptive_thresholds(baselines: dict[str, float]) -> dict[str, float]:
    """station_id -> its own drift threshold: baseline minus ADAPTIVE_DROP,
    clamped to [ADAPTIVE_MIN_THRESHOLD, ACCURACY_THRESHOLD]."""
    return {
        station_id: min(ACCURACY_THRESHOLD, max(ADAPTIVE_MIN_THRESHOLD, baseline - ADAPTIVE_DROP))
        for station_id, baseline in baselines.items()
    }


def stations_needing_retrain(stats: pd.DataFrame, thresholds: dict[str, float] | None = None) -> list[str]:
    """Stations with enough recent data whose accuracy has dropped below their
    bar: their own adaptive threshold if one is given, else ACCURACY_THRESHOLD."""
    if stats.empty:
        return []
    thresholds = thresholds or {}
    bar = stats["station_id"].map(lambda station_id: thresholds.get(station_id, ACCURACY_THRESHOLD))
    drifted = stats[(stats["count"] >= MIN_DATAPOINTS) & (stats["accuracy"] < bar)]
    return sorted(drifted["station_id"].tolist())


def _fetch_recent(station_id: str) -> list[tuple]:
    """New (Temp) observations for this station, oldest first."""
    with connection() as conn:
        return conn.execute(
            'SELECT observed_at, demand FROM "Temp" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()


def _training_frame(station_id: str, recent: list[tuple]) -> pd.DataFrame:
    """All historical data plus the new Temp observations - history only ever grows."""
    with connection() as conn:
        historical = conn.execute(
            'SELECT observed_at, demand FROM "Original Data" WHERE station_id = %s ORDER BY observed_at',
            (station_id,),
        ).fetchall()
    frame = pd.DataFrame(historical + recent, columns=["observed_at", "demand"])
    return frame.drop_duplicates(subset="observed_at").sort_values("observed_at").reset_index(drop=True)


class RatioTargetModel:
    """XGBoost trained on the RELATIVE move from the last observed value,
    exposed as a plain level predictor (2026-10-01).

    A level-target tree can't predict outside the range it trained on, so a
    station whose demand surges past anything it has seen (02300 and 05000
    on 2026-09-17/18: 19% of points above the all-time training maximum)
    gets stuck near that ceiling. This model instead learns
    log((demand + 1) / (lag_1 + 1)) - how much demand moves relative to the
    last value - and converts back with level = exp(pred) * (lag_1 + 1) - 1,
    so the output scales with the current level instead of being capped.

    `.predict(rows)` still takes the same feature rows and returns LEVELS, so
    collector.py / submit_xgboost.py (the only two `.predict()` call sites)
    need no change. Column 0 of every row must be lag_1 (LAGS[0] == 1, an
    invariant enforced at training time). In the recursive chain lag_1 is
    sometimes the chain's own earlier prediction rather than a real value;
    that is the same stand-in the level model already received there.

    Offline backtest (3 chronological folds, full production stack): station-
    mean accuracy +1.16 / +1.41 / +0.94pp vs the level model, pooled +1.32 /
    +1.84 / +2.80pp; on the 7 cycles after the 2026-09-18 shock +2.1pp. The
    only repeatable loss is on low-volume stations whose demand collapses
    (03000/05100 in the shock window, mostly at +45/+60min).
    """

    def __init__(self, model: XGBRegressor):
        self.model = model

    @property
    def n_features_in_(self) -> int:
        return self.model.n_features_in_

    def predict(self, rows) -> np.ndarray:
        features = np.asarray(rows, dtype=float)
        ratio = np.exp(self.model.predict(features))
        return np.maximum(0.0, ratio * (features[:, 0] + 1.0) - 1.0)


def _train_station_horizon_model(
    station_id: str, horizon: int, frame: pd.DataFrame, context: dict | None = None
) -> str:
    """Train a model that predicts `horizon` steps (15min each) directly ahead.

    Every horizon shares the same anchor: the lag_k feature always reads
    `horizon + k - 1` steps behind the row being predicted, which works out
    to the exact same real, already-observed timestamp regardless of
    horizon (data_cutoff - 15*(k-1) minutes) - never another horizon's own
    prediction. Only the target offset and calendar encoding (computed at
    the row's own real timestamp) change between horizons; h=1 reduces to
    the original single-step model exactly.
    """
    y = frame.demand.astype(float)
    lagged = pd.concat(
        {f"lag_{lag}": y.shift(horizon + lag - 1) for lag in LAGS}, axis=1
    )
    calendar = pd.DataFrame(
        [temporal_features(value) for value in frame["observed_at"]],
        columns=["hour_sin", "hour_cos", "week_sin", "week_cos"],
    )
    base_features = pd.concat([lagged, calendar], axis=1)
    # A row is trainable when its lags exist; context columns (below) are
    # allowed to be NaN - that is "no context published for that instant".
    valid = base_features.notna().all(axis=1)
    features = base_features
    if context_module.USE_CONTEXT_FEATURES:
        context_frame = pd.DataFrame(
            [context_module.context_features(value, context) for value in frame["observed_at"]],
            columns=list(context_module.CONTEXT_FIELDS),
        )
        features = pd.concat([base_features, context_frame], axis=1)
    assert LAGS[0] == 1, "RatioTargetModel reads lag_1 from column 0"
    # Target: relative move from the last observed value (lag_1 at this
    # horizon is the value at the cutoff). See RatioTargetModel.
    ratio_target = np.log((y + 1.0) / (lagged["lag_1"] + 1.0))
    model = XGBRegressor(
        grow_policy="lossguide",
        max_depth=0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **_model_params(station_id),
    )
    model.fit(features.loc[valid], ratio_target.loc[valid])
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"xgboost_{station_id}_h{horizon}.joblib")
    joblib.dump(RatioTargetModel(model), path)
    return path


def _upload_model(station_id: str, horizon: int, path: str) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY para subir el modelo.")
    with open(path, "rb") as fh:
        payload = fh.read()
    response = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{MODEL_BUCKET}/xgboost/xgboost_{station_id}_h{horizon}.joblib",
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "x-upsert": "true",
            "Content-Type": "application/octet-stream",
        },
        data=payload,
        timeout=60,
    )
    response.raise_for_status()


def _promote_temp_to_original(station_id: str) -> int:
    """Once the new model is live: fold Temp into history. History only ever grows -
    no rows are ever dropped from "Original Data" here.
    """
    with connection() as conn:
        with conn.transaction():
            # `prediction` is carried over too (not just demand): once these rows
            # move into "Original Data", station_accuracy_stats() needs to be able
            # to keep reading them there for the 6h accuracy window - see its
            # docstring for why.
            conn.execute(
                """INSERT INTO "Original Data" (station_id, observed_at, demand, prediction)
                   SELECT station_id, observed_at, demand, prediction FROM "Temp"
                   WHERE station_id = %s
                   ON CONFLICT (station_id, observed_at) DO NOTHING""",
                (station_id,),
            )
            deleted = conn.execute('DELETE FROM "Temp" WHERE station_id = %s', (station_id,)).rowcount
    return deleted


def _fetch_context_for(frame: pd.DataFrame) -> dict:
    """Published context covering a training frame; {} (all-NaN features) if the
    API can't be reached or has none - never blocks a retrain."""
    try:
        return context_module.fetch_context(frame["observed_at"].min(), frame["observed_at"].max())
    except Exception as exc:
        print(f"AVISO: no se pudo leer /v1/context ({exc}); se entrena sin contexto.", flush=True)
        return {}


def check_and_retrain() -> list[str]:
    """Detect drifted stations and retrain + republish their models. Returns the list retrained."""
    stats = station_accuracy_stats()
    thresholds = station_adaptive_thresholds(station_baseline_accuracy())
    drifted = set(stations_needing_retrain(stats, thresholds))
    # A confirmed Page-Hinkley alarm - a real, sustained regime shift, not
    # noise - can pull a station into consideration even before its own
    # crude RECENT_CHECKS(4)-point rolling accuracy has dropped below
    # ACCURACY_THRESHOLD, since that short average can lag behind what
    # Page-Hinkley's statistical test already sees.
    #
    # As of 2026-09-29 this bypasses MIN_NEW_FOR_RETRAIN too, not just
    # ACCURACY_THRESHOLD - a real, confirmed alarm means this station's
    # ph_enabled flag needs to be re-evaluated against the shift that's
    # actually happening right now, and that flag only gets refreshed at
    # retrain time. Waiting for 20 fresh rows to accumulate first leaves the
    # station on its stale (often disabled) boost decision through the exact
    # window it would help most. This deliberately reverses the 2026-09-28
    # design (see the removed test that used to assert the opposite,
    # test_check_and_retrain_page_hinkley_alarm_never_bypasses_min_new_points)
    # - the tradeoff is that a station whose alarm
    # stays active across many collector cycles with little new data
    # trickling in can now retrain repeatedly on close to the same data,
    # re-spending upload cost each time mostly to refresh the ph_enabled
    # decision. Accepted as the smaller cost against leaving a real,
    # detected regime shift uncorrected.
    alarm_triggered: set[str] = set()
    for station_id in stats["station_id"]:
        if _station_has_fresh_page_hinkley_alarm(station_id):
            drifted.add(station_id)
            alarm_triggered.add(station_id)

    retrained = []
    for station_id in sorted(drifted):
        recent = _fetch_recent(station_id)
        if len(recent) < MIN_NEW_FOR_RETRAIN and station_id not in alarm_triggered:
            continue

        frame = _training_frame(station_id, recent)
        if len(frame) <= max(LAGS) + max(HORIZONS):
            continue
        # Only h1 gets trained now - h2/h3/h4 are predicted by chaining this
        # same h1 model forward (see submit_xgboost.py's predict_cycle_targets
        # docstring for why the direct per-horizon models were dropped
        # 2026-09-29). The old xgboost_{station}_h{2,3,4}.joblib files are
        # left untouched in Supabase Storage - nothing reads them anymore,
        # but they stay there as a free rollback point.
        context = _fetch_context_for(frame) if context_module.USE_CONTEXT_FEATURES else None
        path = _train_station_horizon_model(station_id, 1, frame, context)
        _upload_model(station_id, 1, path)
        merged = _promote_temp_to_original(station_id)
        retrained.append(station_id)
        before_ewma = _ewma_params(station_id)
        after_ewma = _update_ewma_params_state(station_id) or before_ewma
        if after_ewma != before_ewma:
            print(
                f"EWMA de {station_id}: alpha/damping {before_ewma['alpha']}/{before_ewma['damping']}"
                f" -> {after_ewma['alpha']}/{after_ewma['damping']} tras este reentrenamiento.",
                flush=True,
            )
        was_enabled = _is_page_hinkley_enabled(station_id)
        _update_page_hinkley_enabled_state(station_id)
        now_enabled = _is_page_hinkley_enabled(station_id)
        bypass_note = (
            f" (alarma Page-Hinkley activa, se adelantó el piso de {MIN_NEW_FOR_RETRAIN} filas nuevas)"
            if station_id in alarm_triggered and len(recent) < MIN_NEW_FOR_RETRAIN
            else ""
        )
        print(
            f"Drift en {station_id}: reentrenado el modelo h1 (usado de forma recursiva "
            f"para los 4 horizontes) con {len(frame)} observaciones "
            f"({merged} nuevas incorporadas al histórico), modelo republicado en Supabase{bypass_note}.",
            flush=True,
        )
        if now_enabled != was_enabled:
            print(
                f"Page-Hinkley-adaptive EWMA para {station_id}: "
                f"{'activado' if now_enabled else 'desactivado'} tras este reentrenamiento.",
                flush=True,
            )
    return retrained
