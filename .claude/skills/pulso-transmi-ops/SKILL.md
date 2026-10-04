---
name: pulso-transmi-ops
description: Operate and modify the Pulso TransMi live MLOps competition pipeline (this repo, RetoTransmilenioLis) - a GitHub Actions + Supabase system that collects streaming transit demand data, trains/retrains per-station XGBoost models on drift, and submits 4-horizon demand forecasts every cycle to a professor-run competition API. Use this skill whenever working in this repo: touching app/collector.py, app/submit_xgboost.py, app/drift.py, app/health.py, app/net.py, the GitHub workflow files, or anything about submissions, forecasts, drift retraining, or the pipeline's reliability. It captures architectural decisions, past bugs and their fixes, and working conventions the user expects - read it before changing anything in this pipeline, not just when explicitly asked to.
---

# Pulso TransMi pipeline operations

## What this project is

A live competition: every forecast cycle, the API asks for demand
predictions at +15/+30/+45/+60 minutes for ~12 transit stations. Points are
scored by comparing submitted predictions to real observed demand once it
happens. The whole system **runs entirely on GitHub Actions + Supabase** -
there is no local/server component the user manages by hand, and no
step should ever require them to run something locally to keep the
pipeline alive. If you propose a fix, it must work unattended in Actions.

## Architecture at a glance

Two long-lived GitHub Actions workflows, each self-looping internally
(see "Why workflows self-loop" below):

- **`.github/workflows/collector.yml`** → runs `python -m app.collector`
  every ~30 min. Pulls new observations from the Pulso API into Supabase,
  predicts each new point with the current model (for drift monitoring),
  and runs drift detection/retraining (`app/drift.py`).
- **`.github/workflows/submissions.yml`** → runs `python -m app.submit_xgboost`
  every ~5 min. Discovers the current open forecast cycle and submits
  4-horizon predictions per station.

Both check for fresh models from Supabase Storage on every loop iteration
via `scripts/download_models.sh` (not just once at job start) - this matters
because a drift retrain mid-run needs to reach the *other* long-running job
quickly, not after it happens to restart hours later. As of 2026-09-26 this
check is incremental (see "Egress bandwidth incident" below) - it only
actually downloads a model file when it changed, not all 48 every time.

### Data model (Supabase Postgres, via `app/db.py`)

- `"Original Data"` - the one-time 45-day historical seed per station,
  loaded once by `app/load_original.py`. Only ever updated when a station's
  drift retrain promotes new data into it (see `app/drift.py`).
- `"Temp"` - staging table where **every** new streamed observation lands,
  along with the live model's own prediction for that point (used for
  drift monitoring). Not periodically cleared except by a drift retrain's
  promotion step.
- `collector_state` - single-row-per-key cursor state for the observations
  stream (`state_key='observations'`).
- `ops.job_runs` - job heartbeat log. `app/health.py` writes a row here
  each time the collector completes successfully; `submit_xgboost.py`
  checks its freshness (see "Collector heartbeat" below).

**Critical invariant**: any code that builds prediction history (lag
lookups) must read **both** `"Original Data"` AND `"Temp"`, never just one.
A non-drifted station's freshest data lives only in `Temp` - `Original Data`
only advances via a drift retrain. Reading only `Original Data` was a real,
previously-shipped bug in two places (see "Known past bugs" below) - it's a
good default suspicion any time a station-history query looks "off" or a
`RuntimeError: No hay suficiente historia` / `Missing lag` error appears.

### Accuracy metric

This project's own definition (not a generic MAPE):

```
WAPE = sum(|actual - predicted|) / max(sum(|actual|), 1)
Accuracy = max(0, 1 - WAPE)          # per station
```

Per-station accuracies are then averaged for an overall figure. When
reasoning about "why is accuracy X%", always check whether X is: (a) a
single small-sample cycle (4 targets per station - very sensitive to one
bad prediction), (b) the `Temp`-based rolling accuracy (much larger sample,
smooths out individual misses), or (c) the official leaderboard's pooled
`cumulative`/`rolling_24h` figure (see "Pulso TransMi API quirks" - these
can look catastrophically low for reasons that have nothing to do with
model quality).

### Drift detection and retraining (`app/drift.py`)

**Trigger is count-based, not time-based** (changed from an earlier 6h
`ACCURACY_WINDOW_HOURS`/`MIN_DATAPOINTS=20` design): a station's accuracy
over its own last `RECENT_CHECKS=4` predicted/actual pairs (`MIN_DATAPOINTS
= RECENT_CHECKS`, so a station never gets judged on fewer than 4 real
points) falls below **that station's own adaptive threshold** (2026-10-01):
its baseline accuracy over the `ADAPTIVE_BASELINE_CHECKS=192` pairs before the
window (~2 days) minus `ADAPTIVE_DROP=0.05`, clamped to
`[ADAPTIVE_MIN_THRESHOLD=0.75, ACCURACY_THRESHOLD=0.85]`, with the fixed 0.85
as fallback when a station has < 48 baseline pairs (`station_baseline_accuracy`
/ `station_adaptive_thresholds` in `app/drift.py`). `ACCURACY_THRESHOLD` was
0.85, then 0.9 (2026-09-29), then back to 0.85 as the cap. Reason for adaptive:
the 4-point accuracy window is noisy (std ~7pp/station), so any fixed bar mostly
sets retrain frequency - 0.90 flagged ~74% of live windows, 0.85 ~45%, adaptive
~23% (live data Sep 9-18) - while a steady 89% station sliding to 80% still
trips its own bar. The floor keeps a long collapse from dragging the bar down
forever; Page-Hinkley alarms bypass this gate regardless. Retrain *policy*
changes can't be cheaply backtested, so the accuracy effect is unmeasured.
A fixed count survives the
stream slowing down or stalling entirely (a fixed time window could just
never accumulate enough points), and reacts to how the model is doing
*right now* instead of being diluted by a longer history that mixes good
and bad stretches. Because a retrain empties `Temp` for the station it just
touched (its rows move into `Original Data`, including `prediction` -
carried over specifically for this), `station_accuracy_stats()` reads
**both** tables and ranks by `observed_at` across the union, so a
freshly-retrained station doesn't look artificially data-starved right
after.

**Gate before even computing drift stats**: `app/collector.py` calls
`check_and_retrain()` whenever `drift.has_pending_data()` says `"Temp"`
holds any row at all - not "did this exact run insert something new"
(`result["inserted"] > 0`), which is what this used to check (changed
2026-09-26). That earlier version avoided needlessly re-running
drift/retrain when the Pulso API returns pure duplicates during an upstream
feed stall (see "Data feed stalled" below) - but it has a real failure
mode: once the feed stalls for good, no run ever inserts a new row again,
so a station whose data already sits in `Temp` and already qualifies for
retrain (e.g. 05100's real multi-hour demand collapse, discovered
2026-09-26 - see below) can never actually retrain, forever, even though
its bad data has been sitting there the whole time. Checking "is Temp
non-empty" instead still skips redundant checks once Temp is genuinely
drained (right after every station retrains and promotes), but no longer
blocks retrain just because the upstream feed stopped delivering anything
new.

**Retraining always keeps 100% of history now** - no pruning at all. All
new (`Temp`) data for the station is promoted into `"Original Data"` after
every retrain (only once every horizon's model upload succeeds, to avoid
partial/destructive state on failure), and nothing is ever dropped from
`"Original Data"`. This is the end state of two earlier, both-removed
designs: first a two-sample Kolmogorov-Smirnov test (alpha=0.10) that chose
between "keep all history" and "drop the oldest N rows" on a detected
regime shift, then a simpler unconditional 1:1 replacement (drop one old
row per new row folded in) with no KS test. Both were removed on
2026-09-24 after a direct comparison showed a model trained on full
history beats one trained on any windowed/pruned variant, in 12/12
stations. Don't reintroduce a history-pruning branch here without repeating
that comparison.

**A hyperparameter grid search that ran live *inside* every retrain** (over
`_train_station_model`, one shared model pre-multi-horizon) **was tried and
reverted the same day (2026-09-24).** It cross-validated 8 XGBoost configs
per retrain with `TimeSeriesSplit(n_splits=3)`, but a live audit found the
CV signal was pure noise - the spread across all 8 candidates was ~0.4
percentage points, dwarfed by ~2pp fold-to-fold variance for the *same*
candidate. Combined with drift firing on nearly every cycle for nearly
every station under the old 0.90 threshold, the search was re-rolling the
deployed model's hyperparameters from noise roughly every 30 minutes
instead of converging to one stable, well-tuned model - the likely cause of
a real decline in live submission accuracy. **This is a different thing
from the per-station hyperparameters in production today** (`
STATION_MODEL_PARAMS`, keyed by station id, each with its own
`learning_rate`/`n_estimators`/`max_leaves`/`reg_lambda`): those were chosen
**once, offline**, via a walk-forward CV grid search restricted to the
training split only (never touching the held-out test set), and are now
static - no search runs during a live retrain, which is exactly what the
2026-09-24 revert established as the safe pattern. A station outside
`STATION_MODEL_PARAMS` (e.g. newly added) falls back to
`DEFAULT_MODEL_PARAMS`. If a search over these is ever revisited *inside*
a retrain again, it needs a held-out forward test (train on the front of
history, evaluate strictly on the unseen tail - not just averaged CV across
folds smaller than the final training set) and a minimum-improvement
margin before switching away from the known-good default, or it will
reproduce the 2026-09-24 failure mode.

**`MIN_NEW_FOR_RETRAIN = 20` (added 2026-09-27): a drifted station only
actually retrains once >= 20 new rows have piled up in `Temp` for it, not
merely "more than zero".** `has_pending_data()` (above) deliberately keeps
re-running the drift check as long as `Temp` has anything at all in it, for
*any* station - without this second, per-station gate, that meant a station
already below `ACCURACY_THRESHOLD` got fully retrained (4 horizons, 4
Supabase uploads) on almost every ~30min collector cycle off as few as 1-2
new rows against a ~5000-row history: real cost, no real change to the
model, and the actual root cause of the Supabase egress spike investigated
below. This doesn't reintroduce the "blocked forever" failure mode
`has_pending_data()` itself was built to fix - as long as new data keeps
trickling in at all, the count keeps growing and eventually clears the bar,
it just stops spending a full retrain on every single trickle.

**A fresh Page-Hinkley alarm (see below) bypasses BOTH `ACCURACY_THRESHOLD`
AND `MIN_NEW_FOR_RETRAIN` (2026-09-29).** Originally the alarm only skipped
the accuracy gate; it still waited for 20 new rows like every other drifted
station. Changed because a station with a real, fresh alarm and 0 new rows
still needs to retrain - not for new data (there isn't any yet), but so
`_page_hinkley_would_help` gets re-evaluated with the alarm now active,
instead of sitting on a stale on/off decision for however long it takes 20
rows to trickle in.

**One recursive model per station, not one model per (station, horizon)
(reverted to this 2026-09-29 - see "Recursive vs. direct multi-horizon
prediction" below).** A drifted station retrains and republishes only its
`h1` model (`_train_station_horizon_model(station_id, 1, frame)`), uploaded
to Supabase Storage as
`POST /storage/v1/object/models/xgboost/xgboost_{station}_h1.joblib` with
`x-upsert: true` - 12 uploads per retrain cycle, not 48. The 36
`h2`/`h3`/`h4` `.joblib` files from the direct-model era (2026-09 through
2026-09-29) are left untouched in Storage on purpose, as a cheap rollback
path - `download_models.py` still syncs all 48 files that exist, it's just
that only the 12 `h1` ones ever get a new `updated_at` going forward.

### Inference-time bias correction (EWMA) (`station_ewma_bias`)

Separate from retraining: a causal (no-lookahead) exponentially-weighted
moving average of a station's own `(actual - predicted)` residuals,
computed fresh on every submission from every real prediction/actual pair
on record (`"Original Data"` + `"Temp"`, same source `station_accuracy_stats`
reads), added on top of the raw model output in `submit_xgboost.py` right
before a prediction is sent. It exists because a real, sustained miss
(05100's multi-hour demand collapse) can take hours to fix via a
drift-triggered retrain - the EWMA reacts within one collector cycle
instead, without needing a retrain at all. Critically, it never touches
what gets stored as `prediction` in `"Temp"`/`"Original Data"` - that stays
the raw, uncorrected model output, so drift's own accuracy signal and this
correction's own residual history never feed back into each other.

`bias_t = alpha * residual_t + (1 - alpha) * bias_{t-1}`, then
`correction = damping * bias_t`, added to the raw prediction (clamped
`>= 0`). Config (`DEFAULT_EWMA_PARAMS = {alpha: 0.2, damping: 0.5}`,
`STATION_EWMA_PARAMS` for per-station overrides, currently empty) was
chosen the same way as `STATION_MODEL_PARAMS`: a per-station grid search
over `alpha x damping`, picking whichever config maximizes accuracy on
that station's **entire** history, not just its most recent/noisiest
stretch - a config biased toward only recent data reacts fast but chases
noise on an ordinary day. A candidate needs to beat the shared default by
`MIN_EWMA_IMPROVEMENT_PP = 0.5` pp before a station earns its own
override. As of the 2026-09-26 search (re-run twice, with different/larger
datasets, including 05100's real collapse), **no station cleared that
margin** - even 05100, whose collapse looked like it needed a much more
aggressive config when evaluated in isolation on just that ~24-row window,
but the gain nearly vanished once judged against the whole dataset. So
every station currently uses the shared default; `STATION_EWMA_PARAMS`
exists purely so a future override is a one-line addition once a real
margin actually turns up. Don't add a per-station override without
repeating this margin-checked, full-history search - reacting to an
isolated bad window is exactly the anti-pattern the 2026-09-24 revert (see
above) exists to prevent.

### EWMA (alpha, damping) re-tuned automatically at every retrain (2026-10-01)

Requested by the user (they chose it over "retrain when the bias stays large"
and "continuous alpha"). `check_and_retrain()` now calls
`_update_ewma_params_state(station)` after each model upload: a grid search
(`EWMA_TUNE_ALPHAS` x `EWMA_TUNE_DAMPINGS`) on the station's FULL recorded
history, kept only if it beats `DEFAULT_EWMA_PARAMS` by `MIN_EWMA_IMPROVEMENT_PP`
(0.5pp); needs `EWMA_TUNE_MIN_ROWS`=200 rows. The result is stored in
`collector_state` as `ewma_params:{station}` ("alpha,damping") and read by
`_ewma_params()`, which also becomes the baseline for the Page-Hinkley
"would it help" check. Walk-forward backtest (tune before each fold, score on
unseen data): fold 70-85% 0.00pp (margin never cleared), last 15% +0.35pp
station mean (02300 +2.06, 05000 +1.48, 05100 +0.69), none worse. Only ~925
rows per station have stored predictions, so the sample is small - this
supersedes the "STATION_EWMA_PARAMS stays empty" note above for live behavior.
Earlier warnings still apply: don't tune on a short recent window.

### `lag_672` was removed entirely, for every station and horizon (2026-09-27)

`lag_672` (a full week back) anchors every model to "what happened at this
same time last week" - a good default normally, but actively counterproductive
while a station's demand pattern is genuinely shifting (05100's collapses
are the clearest case). This went through two rounds of testing:

1. **2026-09-26**: two independent chronological folds (70-85% region, and
   the standard last-15% test split), per station per horizon, comparing
   the untouched 9-feature baseline against (a) dropping `lag_672` entirely
   and (b) softly down-weighting it (`feature_weights`, weight 0.2, with
   `colsample_bynode=0.8` - `feature_weights` is a silent no-op at the
   default `colsample=1.0`). A candidate needed to beat baseline by
   **>= 0.3pp on BOTH folds independently**. Result: only 7 (station,
   horizon) pairs qualified for full removal and 1 for the soft weight;
   everyone else kept the untouched baseline. This produced a per-pair
   override table and a `lags_for(station_id, horizon)` function that both
   `_train_station_horizon_model` and `predict_cycle_targets` had to call
   to know which lag set a given saved model expected.
2. **2026-09-27 re-test with more accumulated data**, same two-fold
   methodology: the soft-weighted variant **never once beat full removal**
   anywhere it was tried (tried weights 0.2/0.4/0.6), and with more history,
   full removal now cleared the bar for **~15 more pairs** beyond the
   original 7 - not a small correction, a clear trend that removal keeps
   winning as more data accumulates.

Given that, `lag_672` was dropped from `LAGS` **for all 12 stations and all
4 horizons** in `app/drift.py`, rather than maintaining an ever-growing
per-pair exception list. `lags_for()`, `NO_LAG672_MODELS`,
`SOFT_DEEMPHASIZE_LAG672_MODELS`, and `_feature_weight_kwargs` were all
**deleted** from `app/drift.py` - there is now exactly one `LAGS =
(1, 2, 4, 96)` tuple (4 lags + 4 calendar features = 8 features), shared by
every model, imported directly by `collector.py` and `submit_xgboost.py`
(`from app.drift import LAGS`) instead of each keeping its own local copy.
Held-out (last-15%) accuracy: 83.77% -> 84.37% (+0.60pp), 11/12 stations
improve (05100 the most, +2.62pp average); only `09000` regresses slightly
(-0.11pp) - accepted as the cost of one uniform feature set.

**This also structurally eliminates known past bug #7 below** (the
feature-shape-mismatch outage) - there is no more per-pair lag set for a
call site to forget to look up, so that whole class of bug can't recur here.
If a future feature-set change ever needs to vary by (station, horizon)
again, treat it as reintroducing exactly the fragility this removal
eliminated, and expect to rebuild something like `lags_for()` deliberately,
with the "grep every `.predict()` call site" discipline from bug #7's
lesson.

**`week_sin`/`week_cos` were tested the same way (2026-09-26) and found NOT
to have the same problem - don't retest this without new evidence.**
`app/features.py`'s `week_sin`/`week_cos` also encode a one-week period,
so it's a fair question whether they suffer the same staleness issue as
`lag_672`. They don't, structurally: `lag_672` is a raw demand *value* from
one specific week-old timestamp (a single anomalous point leaks straight
into the prediction), while `week_sin`/`week_cos` carry no demand value at
all - they just say "it's Tuesday 8:15am," so the model learns an
*aggregated* pattern across every historical Tuesday-8:15am, not one point.
Running the identical two-fold methodology (baseline vs. dropping both
`week_sin`+`week_cos` vs. softly down-weighting them, per station per
horizon) confirmed this: **zero station/horizon pairs were robust** - not
one cleared the 0.3pp margin on both folds with the same candidate; wins
were small and randomly flipped between "drop," "soft-weight," and
"baseline wins" depending on the fold, the same noise signature as every
other null result in this file (recency weighting, per-station EWMA). No
change was made. If revisited, use the same two-fold test - a single-fold
result here would be exactly the kind of noise this project has repeatedly
mistaken for a real signal.

### Deploying a feature-set/architecture change (any change to what a model's `.joblib` expects)

A change to `LAGS` (or any other change to the feature vector shape) only
changes what a *newly trained* model looks like - every already-deployed
`.joblib` file is untouched and still has the old feature count/schema.
Pushing the code change alone, without retraining, creates an immediate
mismatch between what the inference code sends and what the old file
expects - this fails loudly inside XGBoost's C++ layer
(`ValueError: Feature shape mismatch`), not as a friendly Python exception,
and it hits on every single prediction for that model, not intermittently.
A change like this is not complete until all of these happen together,
**in this order, verified before deploying** (this is exactly what the
2026-09-27 `lag_672` removal did):

1. Change the code (e.g. `LAGS` in `app/drift.py`).
2. Retrain and re-upload **every** affected model on current full history,
   using the exact same `_train_station_horizon_model` / `_upload_model`
   functions a real drift retrain would call (not a hand-rolled
   equivalent) - loading `.env` into `os.environ` first (note: the DB URL
   there is `SUPABASE_DATABASE_URL`, but `app/db.py` reads
   `os.environ["DATABASE_URL"]`, so it must be copied across or the
   connection call raises `KeyError`).
3. **Before uploading/deploying**, verify locally: load every retrained
   `.joblib` and check `model.n_features_in_` matches expectation, and run
   the actual `predict_records()` and `predict_cycle_targets()` functions
   (not a reimplementation) against real DB data end-to-end with no mocks -
   this is what would have caught the 2026-09-26 outage before it shipped.
4. Only then upload to Supabase Storage and push/restart the workflows.

Don't wait for the next natural drift trigger to pick up an architecture
change this way - an untriggered station keeps serving the old, now
schema-mismatched file indefinitely, and the mismatch only surfaces the
moment its horizon happens to be requested.

### Acceptance rule for model/feature changes (changed 2026-09-29)

**A change passes if it improves at least one chronological fold and does not
hurt the other** (per-fold mean delta; a mean loss beyond ~0.3pp, or a station
losing noticeably, counts as hurting). This replaces the earlier bar of
">= 0.3pp on BOTH folds independently" used in the `lag_672` and
`week_sin`/`week_cos` tests above - those results describe how the old rule
decided, not the current rule. Reason: a change that helps a lot in a stressed
period (05100's collapse) and is neutral in a calm one is a good trade, and the
both-folds bar rejected it only because the calm fold has nothing left to
improve. Still report per-fold and per-station numbers, and still confirm with
the user before pushing anything with live production effects.

### Ratio-target model (DEPLOYED 2026-10-01 after a retest; first rejected 2026-09-29)

**Current state: every station's `h1` model is a `RatioTargetModel`** (`app/drift.py`):
XGBoost on `log((demand+1)/(lag_1+1))`, exposed as a plain level predictor (`.predict(rows)`
returns levels, column 0 = lag_1), so the two `.predict()` call sites are unchanged. Reason
it was reversed: the 2026-09-29 test used a calm fold and a collapse fold only. The
2026-09-17/18 shock added an UPWARD surge past the training range (02300/05000: 19% of points
above the all-time training max, which a level tree cannot predict), and on three chronological
folds through the full stack (chain + clamp + EWMA/PH) the ratio model won on all of them:
station-mean +1.16/+1.41/+0.94pp, pooled +1.32/+1.84/+2.80pp; +2.1pp on the 7 cycles after the
shock began. Known repeatable loss: low-volume stations whose demand COLLAPSES (03000/05100/09122
in the shock window, mostly at +45/+60min). A seasonal collapse gate (use the level model when the
recent level < 0.5x the typical level for those slots) gained <=+0.15pp pooled and would need two
models per station, so it was NOT built. Deploy note: models are class instances, so any running
job must be on code that defines `RatioTargetModel` before ratio `.joblib` files are uploaded, or
`joblib.load` raises AttributeError - push code and restart both workflows FIRST. A bias/EWMA
history built from the old level model's residuals is briefly mismatched right after the switch.

Original 2026-09-29 rejection (kept for the lessons):

Idea: train XGBoost on `log((demand+1)/(lag_1+1))` (relative move from the last
observed value) and convert back to a level at predict time, because a
level-target tree over-predicts by 40-50% while a station's level collapses
(05100, 03000). The code (a `RatioTargetModel` wrapper in `app/drift.py`, which
would have kept the `.predict(rows) -> levels` contract so no call site
changed) was written and verified locally, then reverted at the user's
decision not to deploy. Don't rebuild it without new evidence.

Offline backtest (12 stations, folds 70-85% and last 15%, recursive h1->h4
chain): raw chain, calm fold -0.04..+0.2pp mean, collapse fold +0.7..+2.0pp
(05100 h1 +7.3pp). **With the full production stack (alarm-gated 50% band
clamp + EWMA/Page-Hinkley bias) the gain nearly vanished:** -0.29/-0.04/+0.06/
+0.09pp (h1-h4) on the calm fold, +0.64/+0.26/+0.24/+0.20pp on the collapse
fold; 05100 gained +1.2pp at h1 but lost 1.0-1.8pp at h2-h4, and 03000 lost
0.3-0.9pp on both folds. The EWMA/clamp/PH stack already corrects most of the
level model's bias, and Page-Hinkley fires less on the ratio model's smaller
residuals (alarm share 7% vs 11% on the collapse fold), so the clamp/boost
that helps 05100 engages less. Slope features (`lag_1-lag_2`, `lag_1/lag_4`)
also did nothing (+0.07/+0.02pp). Lesson: backtest any model-side change
through the whole inference stack (chain + clamp + EWMA/PH), not just raw
model output - the post-processing can absorb, or interact with, the gain.

### Migration to a new Supabase project (2026-10-01)

The pipeline moved to a **new Supabase project** after the old one (ref
`egemorkltmcxqadugotl`) was restricted for exceeding its quota (Storage calls
return HTTP 402 Payment Required; this is also the likely reason collector runs
failed on 2026-09-30/10-01). `scripts/migrate_supabase.py` copied the database,
the model files in Storage, and `web/config.js`, then updated the GitHub repo
secrets (`SUPABASE_DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
rotated 2026-10-01 01:01 UTC).

Things to remember:

- **Local `.env` holds both projects.** The plain `SUPABASE_*` variables are the
  OLD project (frozen: its newest heartbeat is 2026-09-30 22:05 UTC); the NEW one
  is `NEW_SUPABASE_DATABASE_URL`, `NEW_SUPABASE_URL`,
  `NEW_SUPABASE_SERVICE_ROLE_KEY`, `NEW_SUPABASE_PUBLISHABLE_KEY`. For any live
  check or manual retrain, map the `NEW_` values onto `DATABASE_URL` /
  `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` first, or you will read stale data
  and conclude the pipeline is dead when it is fine (this happened once).
- Don't `source .env` in bash - one value breaks shell parsing; parse it in Python.
- A failed collector/submission run around 2026-09-30 21:00 to 2026-10-01 01:00 UTC
  ran on the old credentials; it is a migration artifact, not a code bug.
- After the migration the data feed is still the slow virtual clock (newest
  `observed_at` 2026-09-18 in the new DB at the time of writing).

### Collector heartbeat (`app/health.py`)

Drift detection only ever runs from `collector.yml`. If that workflow gets
disabled, cancelled, or silently stops, nothing else would notice on its
own - `submissions.yml` would keep submitting against an ever-staler model
forever. Fix: the collector writes a success row to `ops.job_runs` after
each run; `submit_xgboost.py` checks that row's freshness *after* a
submission already went out (so a stale collector never blocks a
submission that otherwise succeeded) and raises `CollectorStale` if the
last successful collector run is older than `HEARTBEAT_STALE_AFTER_MINUTES`
(90 min = 3x the collector's cadence). That exception is left uncaught in
`main()` so it exits non-zero and the workflow's fail-counter picks it up,
surfacing in the Actions tab.

### Egress bandwidth incident (2026-09-26) and incremental model sync

Supabase emailed a Fair Use Policy warning: the project's egress bandwidth
had blown well past the free tier's quota. The cause was
`scripts/download_models.sh`, which used to unconditionally re-download all
48 model files (~19MB total) via `curl` on **every single loop iteration**
of both workflows - not just at job start. With collector looping every 30
min (48 times/day) and submissions every 5 min (288 times/day), that's 336
full re-downloads/day - about 6.1GB/day, ~184GB/month, roughly 35x a typical
5GB/month free-tier egress cap, even though most of those downloads fetched
byte-identical files.

Fixed by making the sync incremental (`scripts/download_models.py`, called
by the still-named `scripts/download_models.sh` wrapper so the workflows
didn't need to change): one Storage `list` call per loop iteration fetches
just the metadata (name + `updated_at`) for all 48 files - practically free,
no model bytes transferred - and it's compared against a local manifest
(`models/xgboost/.manifest.json`) of what was downloaded last. Only files
whose `updated_at` actually changed (or that are missing locally) get a real
`GET` and count against egress. A drift retrain still propagates to the
other job within one loop iteration, same as before - only now that
iteration downloads just the files that changed (at the time of this fix,
one station's 4 horizon files; since the 2026-09-29 revert to a single
recursive `h1` model per station - see "Recursive vs. direct multi-horizon
prediction" below - a retrain changes only 1 file), not all 48. The
manifest lives alongside the models in the same job's
filesystem, so it persists across loop iterations within a single ~5.75h
job run and is naturally rebuilt from scratch (one full download) whenever
`actions/checkout` resets the workspace on the next job restart - same
cadence as before, just no longer on every 5-minute tick.

Don't reintroduce an unconditional re-download loop here - if this needs
revisiting (e.g. a different model storage backend), preserve the
"only fetch what actually changed" property, since that's what the whole
fix is protecting.

### Database query egress (2026-09-27): stop pulling entire tables to read a handful of rows

Separate from the Storage-download incident above, three DB queries were
each pulling far more than they used: `submit_current_cycle()` was
`SELECT`ing **all** of `"Original Data"` (~59,000 rows and growing) plus all
of `"Temp"` every ~5min while a cycle is open, just to resolve each
station's lag features; `collector.py`'s `predict_records()` did the same
per-station; and `station_ewma_bias()` pulled a station's entire
prediction/actual history for an EWMA whose weight decays geometrically
(alpha=0.2 makes anything past ~93 rows back numerically irrelevant).
Fixed:

- `submit_current_cycle()` / `predict_records()`: filter by
  `station_id = ANY(%s) AND observed_at = ANY(%s)` with the exact
  timestamps `LAGS` can need, instead of an unfiltered/station-only
  `WHERE`. Verified live: 54 rows instead of 58,992 for the same cutoff.
- `station_ewma_bias()`: `ORDER BY observed_at DESC LIMIT
  _ewma_lookback_rows(alpha)` instead of the full history, then reversed
  back to chronological order for the recursion. `_ewma_lookback_rows()`
  derives the row count from that station's own `alpha` (not a hardcoded
  number) so it stays correct even if a future `STATION_EWMA_PARAMS`
  override picks a much slower-decaying alpha. Verified live: matches the
  full-history computation to ~9 significant figures.

All three queries now scale with the size of one response (one station, a
handful of timestamps), not with the size of the ever-growing accumulated
history - keep it that way; re-adding an unfiltered `SELECT * FROM
"Original Data"`-style query anywhere in the hot path (called every
collector/submission cycle) reintroduces exactly this problem.

### Network resilience (`app/net.py`)

`with_retries()` wraps outbound HTTP calls (both to the Pulso API and, via
`scripts/download_models.py`, to Supabase Storage).
It retries only **transient** failures (DNS, timeout, connection refused) -
3 attempts, exponential backoff 5s/10s/20s - and raises `UpstreamUnavailable`
if still failing. Real application errors (HTTP 4xx/5xx, bad auth,
malformed payload) propagate immediately, unretried, since those are bugs
not flakiness. `run()`/`main()` in `collector.py`/`submit_xgboost.py` catch
`UpstreamUnavailable` specifically and `sys.exit(75)` - the workflow loops
treat exit code 75 as "not a real failure, just sleep 5 min and retry",
distinct from the normal 3-consecutive-failures kill-switch.

### Why workflows self-loop instead of relying on cron

GitHub Actions' `schedule` trigger has **no SLA** - under load it can be
delayed hours past its stated interval (empirically observed: a 30-min cron
firing once in a 5-hour window). Fix: each job internally polls in a bash
while-loop for ~5.75h (`timeout-minutes: 355`), combined with
`concurrency: {group: <name>, cancel-in-progress: false}`, which **queues**
(doesn't skip) any scheduled trigger that fires while the loop is still
running - so as soon as one ~5.75h loop ends, the next queued run picks up
immediately, and the chain self-sustains indefinitely once kicked off once
via `workflow_dispatch`. Collector loop cadence is 30 min (`sleep 1800`),
submissions is 5 min (`sleep 300`).

**Implication for editing these workflows**: if you push a fix to code that
an already-running loop uses, the *currently executing* job won't pick it
up (it already did `actions/checkout` once at job start) - you need to
explicitly cancel and re-trigger (`gh run cancel`, then
`gh workflow run <file>`) to get a fix live immediately, otherwise it only
takes effect the next time that ~5.75h job naturally restarts.

## Recursive vs. direct multi-horizon prediction (critical correctness point)

**Current design (since 2026-09-29): one recursive `h1` model per station,
not one model per (station, horizon).** `predict_cycle_targets()` in
`app/submit_xgboost.py` loads exactly one model per station
(`models[station_id]` from `{MODEL_DIR}/xgboost_{station_id}_h1.joblib`)
and calls it once per horizon **in chronological order**, feeding each
prediction back in as the stand-in for the not-yet-observed value the next
horizon's lags need: for target horizon `h`, `lag_k`'s feature value is the
real observed value at `cutoff + 15*(h-k)` minutes (= k steps before the
target) if `k >= h`, otherwise it's the chain's own prediction for horizon
`h - k`. This is required because `lag_k` means "demand exactly k*15 minutes
before the *target* timestamp", which only equals "k*15 minutes before
`data_cutoff`" for the +15min horizon. **Do not** compute all 4 horizons'
features relative to `data_cutoff` directly - that was a real,
previously-shipped bug that silently broke 75% of every submission (see
"Known past bugs" below). **A second, subtler version of the same mistake
lived in the chain until 2026-10-02**: the real lags for `k >= h` were read
at `cutoff - 15*(k-1)` (anchored at the cutoff, as the old direct models
were), i.e. `h-1` steps staler than "k steps before the target" for h >= 2.
Fixed (`lag_timestamps()` in `submit_xgboost.py` lists the exact points read):
through the full stack with h1 models retrained on each fold's prefix, it
improved all three chronological folds (+1.17 / +0.95 / +1.01pp station mean,
h4 +2 to +3pp, every station gained, none lost >0.3pp). The test that locks it
is `test_predict_cycle_targets_chains_h1_prediction_recursively` (distinct
value per offset) plus `test_chain_matrix_matches_the_recursive_chain_used_in_production`
(the vectorized copy in `app/regime.py` must equal the production chain).
`app/collector.py`'s `predict_records()` already uses the correct pattern
for its own (single-step) prediction - mirror it rather than reinventing.

A station missing any real lag (most likely `lag_96` = one day back, the
longest lag now that `lag_672` is gone - see below) is skipped **entirely**
- all 4 targets, not just one horizon. This differs from the direct
design's per-target skip: every horizon here ultimately depends on the same
real `lag_1` (or a prediction chain built from it), so a gap blocks the
whole recursive chain, not one link of it.

**Why this reverted the 2026-09 direct-per-horizon design** (which had 4
independent models per station, one per horizon, specifically built to
*eliminate* this recursion/compounding): a held-out backtest (train on the
front 85% of each station's history, test on the unseen last 15%, no leakage)
found the direct models were consistently *worse* - not just on stations
with a fresh demand-level shift, but on all 12, sometimes by 10+ accuracy
points at `h4`. An oracle decomposition (feeding the chain ground truth
instead of its own predictions) showed the gap wasn't mainly about
compounding error - that's real but small (1-8pp at h4, sub-additive,
since XGBoost's own noise there is mostly unbiased) - it's that the direct
long-horizon models leaned too heavily on `lag_96` (or, for some stations,
mostly time-of-day), a weaker signal than what the `h1` model already
learns from the freshest, tightest lag. `check_and_retrain()` only trains
and uploads `h1` now; the old `h2`/`h3`/`h4` files stay in Storage
untouched as a rollback path (see `MIN_NEW_FOR_RETRAIN` section above) -
reviving this design means reviving the direct-model *code*, not just
those files.

As of 2026-09-27, every model shares the exact same `LAGS = (1, 2, 4, 96)`
feature set - `collector.py` and `submit_xgboost.py` both import `LAGS`
directly from `app.drift`, there is no per-pair variation to look up (see
"`lag_672` was removed entirely" above).

## Page-Hinkley adaptive drift detection (replaces CUSUM, 2026-09-28)

On top of the EWMA bias correction above, each station's standardized
residuals also feed a two-sided Page-Hinkley change-point test
(`_page_hinkley_step`/`_new_page_hinkley_state` in `app/drift.py`). While no
change point has fired, behavior is identical to the plain fixed EWMA.
Once it fires (a real, sustained deviation, not routine noise), the
station switches to a more reactive `(alpha, damping)` pair for a fixed
cooldown window, then reverts. The original version of this used CUSUM
(reset to a hard zero floor); Page-Hinkley replaced it by comparing against
its own running min/max instead of resetting to zero, matching or slightly
beating CUSUM's win on 05100's real collapse **without** the regressions
CUSUM caused on 09000/09122.

`PH_ADAPTIVE_PARAMS = {boost_alpha: 0.4, boost_damping: 0.7, cooldown: 80,
lambda: 25.0}` is **one shared config across all 12 stations, never tuned
per station** - a per-station version was tried first (2026-09-27) and
overfit: ~500 training points per station isn't enough to pin down 4 free
parameters, and different chronological folds picked different "best"
configs for the same station that didn't hold out-of-fold. What *does*
vary per station is only whether the boost is **on or off**
(`_is_page_hinkley_enabled`, backed by `PH_ENABLED_STATIONS` /
`collector_state`), re-decided every time that station retrains via
`_page_hinkley_would_help` - it must beat the plain fixed EWMA by
`MIN_PH_IMPROVEMENT_PP = 0.5`pp on that station's full history before
flipping on, using the same margin-checked, full-history discipline as the
EWMA override search above. Don't re-tune `PH_ADAPTIVE_PARAMS` per station
without repeating the fold-consistency check that caught the 2026-09-27
overfit - re-pooling the search on just the regressing stations to "fix"
them specifically made both worse, not better, since shrinking the pool
removes the regularizing effect that made the shared search work.

Separately, `_station_has_fresh_page_hinkley_alarm(station_id)` runs the
same state machine as **pure detection** - it never changes any prediction
or correction by itself, and runs for every station regardless of whether
that station's boost flag is on. Two things key off this signal:
`check_and_retrain()`'s alarm-bypass (see `MIN_NEW_FOR_RETRAIN` above) and
the band clamp below.

### Alarm-gated band clamp on the recursive chain (2026-09-29)

While a station's Page-Hinkley alarm is fresh, `predict_cycle_targets()`
clamps every horizon's value - both what gets fed forward as a future
horizon's lag AND what gets emitted/scored - to
`[anchor * (1 - PH_ALARM_BAND_PCT), anchor * (1 + PH_ALARM_BAND_PCT)]`,
where `anchor` is the last real observed value (`lag_1` at cutoff) and
`PH_ALARM_BAND_PCT = 0.5`. Outside an active alarm, nothing changes.

This exists because the recursive chain above can compound badly during a
genuine demand-level shift: one station's `h4` accuracy hit 0% during a
real event before this fix. A **proportional shrink** toward the anchor
(`value = anchor + shrink * (raw - anchor)`) was tried first and backtested
across full history (not just the held-out tail, which happened to overlap
the volatile event and made an unconditional shrink look like a universal
win when it wasn't): it fixed the worst compounding cases but distorted
*every* alarm-time prediction, including legitimately large real moves -
one station whose alarms fire on genuine fast trends (not model drift) lost
up to -26pp at h4 under an aggressive shrink. A **band clamp only touches
predictions that are already outside the band, symmetric for over- and
under-prediction** - it left that station's damage at -2.6pp while keeping
nearly all the gain on the stations with real compounding (one went from
0% to 87%+ h4 accuracy during its real alarm cutoffs). Gated by the causal,
no-lookahead alarm signal (same state machine as production), swept across
several band widths before picking 0.5 as the widest one that still kept
most of the gain while minimizing the one regressor's damage.

## 4-hour wave layer (`app/regime_4h.py`, deployed 2026-10-02) - read before touching submissions

**Finding.** Since virtual ~2026-09-18 11:15 UTC all 12 stations oscillate with a
16-slot (4h) period (~81% of spectral power; the demand is synthetic, so this is a
scripted scenario). The XGBoost chain (lags 1,2,4,96 + 24h/weekly calendar) can't
see it: ~70% on those windows vs ~90% for just repeating `y[target-16]`. Before the
onset `lag16` scored only 12-28%. This is why the user's "last 6 cycles" score
(65.8%) trailed the leaders (90-92%) while the cumulative board had them 6th, 5pp
behind first. An oracle test (true lags instead of the chain's own predictions)
tops out at ~85%, so the leaders' 90% comes from the period, not a better one-step
model.

**What is wired.** `submit_xgboost.submit_current_cycle` (flag `USE_4H_WAVE`, turn
it off to restore the old behavior exactly): after `predict_cycle_targets`, each
(station, horizon) prediction is replaced by `y[target-16]` ONLY while
`regime_4h.wave_active` holds, otherwise the standard stack (chain + Page-Hinkley
band clamp + EWMA) is untouched. Entry: lag-16 accuracy >= 0.80 over the last 16
slots AND >= 0.75 over the last 8 AND >= persistence-at-horizon + 0.10. Exit: last
4 slots < 0.60, or entry conditions stop holding. Stateless, recomputed every
cycle from 32 slots/station (the DB read becomes 33 exact timestamps per station
including `lag_96`; still bounded, never a table scan). When a value comes from
the wave the EWMA correction is skipped (it's measured on chain residuals). The
`prediction` stored in `Temp` for drift monitoring stays the raw chain one-step
output. Any exception inside the layer degrades to the standard chain and never
blocks a submission.

**Adaptive multi-period average (2026-10-02).** The wave value is no longer
`y[target-16]` but the mean of up to `MAX_PERIODS=6` previous periods
(`y[target-16j]`), stopping at the first missing one or one whose 8-slot block
was not itself wave (lag-16 accuracy < 0.70, `_period_was_wave`), so it never
mixes in the pre-wave regime. Why: a single copy carries that period's noise
(error ~ noise of two samples); averaging cancels it. Production functions on
real data, same activations: since onset 90.2 -> 92.0, since 09-19 00:00
90.2 -> 92.4, last 8 cutoffs 89.8 -> 92.7 (all 12 stations improved in the
24h fold; pre-onset the wave is never active, unchanged). `HISTORY_SLOTS` is
now 119 (about 1.4k exact timestamps per query, still bounded). Leaders sit
at ~93 on the rolling 24h board, consistent with a denoised template.

**Evidence (all 12 stations, real cutoffs, station-mean accuracy; chain = full
production stack).** Exact last 6 cycles (hourly cutoffs 02:00-07:00 on 09-19):
69.8% -> 90.3%. Since 09-18 12:00: 69.0% -> 89.8%. Pre-onset: 81.0% -> 81.3%
(active 2% of the time). The 0.80 entry never fired before the onset in 8 weeks
of history. Synthetic stress (wave stops, 4h->6h, inverted phase, noise burst,
amplitude change, no wave): it turns itself off within ~16 slots, worst harm ~-2pp
in those first 16 slots, never activates without a wave.

**Things tried that did NOT work (don't redo without new evidence).**
- Retrained direct per-horizon models (old-style absolute lags, relative-ratio
  features with yesterday's move as ratios, a no-seasonal variant, and a 50/50
  chain+direct blend) on 3 chronological folds incl. the shock: blend
  -0.07/+0.83/+0.39pp, the rest worse; all ~68% on the shock fold. They can't learn
  a cycle absent from their training data.
- Seasonal-ratio and "same slot yesterday" naive forecasts: ~51% since 09-18 12:00.
- A missed-cycle theory for the 65.8%: wrong (the portal showed 6/6 delivered, 100%
  coverage). Cumulative coverage 98.09% = exactly 154/157 cycles, i.e. 3 missed in
  the whole competition.
- The stacked "learn any period" layer, FIRST version (`app/regime.py`, now
  rewritten - see the next paragraph):
  searches lags 6-48 slots, fits `a + b*y[t-L]` (negative b covers inversion) on a
  fit window, scores on a later validation window, and softmax-blends chain / lag
  forecast / chain + periodic error correction. Real backtest: last 6 cycles 89.5%
  (backup 90.3%), since onset 86.2% (backup 89.8%), and it LOST 1.1pp pre-onset
  (3.3pp at h4) by fitting noise (calm-period weights ~0.51 chain / 0.41 correction
  / 0.09 lag). Fails the acceptance rule. It also adapts only with a delay of hours
  (synthetic inversion/period-switch tests were no better than persistence for the
  first ~30 slots). Improvement ideas are in the README (point 15) and below.

**Generic layer, current state (`app/regime.py`, wired behind `USE_GENERIC_REGIME`,
OFF by default).** Runs after the 4h backup, only on predictions it did not claim.
Strict gate: a learned forecast (M1 = `a + b*y[t-L]` over all L in 6-48 slots,
b may be negative; M2 = chain + periodic correction of its own error) only counts
if its correlation is stable in sign and |r| >= 0.5 across the fit and validation
windows, it beats the chain by 0.15 AND reaches 0.80 validation accuracy; the top
3 stable lags are averaged and softmax-blended with the chain; EWMA is scaled by
the chain's weight; any error restores the standard chain. Replay of the full
pipeline vs the live one on 178 hourly cutoffs: calm -0.07pp (0.31% of predictions
changed, worst station 05000 -0.63pp), wave 0.00, exact last 6 cycles 0.00 - i.e.
no real-data benefit because the only real pattern is the 4h wave the backup
already covers; its evidence for other patterns (6h wave, inverted, period
switch) is synthetic only and it loses ~8-12pp for ~30 slots after an abrupt break.
Turning it on is the user's call. `submit_current_cycle` also keeps a per-job marker
file (`.last_submitted_cycle` next to the models) so the later 5-minute wake-ups of
the same cycle return early instead of re-reading history to hit a 409.

**Lessons.** (1) When a score gap vs. peers looks too large to be model quality,
inspect the data's structure (autocorrelation by lag, spectrum) before modeling -
the answer was one lag. (2) The honest ceiling for a model matters: compare against
an oracle and trivial baselines before building. (3) Synthetic "robustness" results
vs persistence are weak evidence; always test against the real stack.

## Data feed: a slow virtual clock, not a permanent stall

**Correction (2026-09-26): an earlier version of this doc said the feed was
"stuck since 2026-09-13" - that framing is wrong and was corrected live
after the user pushed back on it ("how can new data be arriving if the API
is not sending new data?").** Rechecking the live API over several hours
showed `data_cutoff` genuinely advancing (e.g. 03:00 -> 08:00 across one
afternoon) - the upstream stream is a **virtual clock replaying historical
data at roughly real-time pace, chronically ~12 days behind the real
"now"**, not a feed that died on a fixed date. As of the last live check
(2026-09-26), every station's most recent real `(demand, prediction)` point
on record was `2026-09-14 11:30:00 UTC` - confirming the ~12-day lag is
still the right mental model, not a permanent halt. **Re-confirmed
2026-09-29**: every station's most recent recorded `(demand, prediction)`
point was still `2026-09-17 00:30:00 UTC` at that check - same ~12-day lag,
same mental model, not a regression.

Practical implications:

- **A code/model change deployed "today" won't show up in real revealed
  outcomes for a while** - the virtual clock has to crawl forward past the
  deployment moment before any genuinely-new `(demand, prediction)` pair
  exists to judge it by. Don't mistake "no new real data yet" for "the
  change didn't work."
- `has_pending_data()` (see above) still matters independently of this -
  it exists for the cross-process race between `collector.yml` and
  `submissions.yml` (see its own docstring), not because of this clock lag.
- If accuracy or drift behavior looks strange, check the most recent
  `observed_at` across stations before assuming the model or drift logic is
  at fault - if it hasn't moved since your last check, you're looking at
  the same already-explained window, not new evidence.
- **Estimating "the accuracy of the last submission"**: there's no stored
  record of what was actually sent for h2/h3/h4 (only `submit_xgboost.py`'s
  raw, uncorrected h1-equivalent one-step prediction gets persisted, via
  `collector.py`'s `predict_records`, purely for drift monitoring). To
  honestly reconstruct a "what did the last submission likely look like"
  estimate: take the most recent real `(demand, prediction)` pair per
  station, and add the EWMA correction **computed only from the history
  strictly before that point** (causal, no lookahead - mirrors
  `station_ewma_bias`'s own logic) to approximate what was actually sent,
  since the correction itself is never stored. This is still h1-only and a
  single-timestamp sample (noisy, one or two stations can swing the pooled
  number a lot) - always say so rather than presenting it as a clean
  aggregate. It also doesn't account for the alarm-gated band clamp (see
  "Recursive vs. direct multi-horizon prediction" below) - if that
  station's Page-Hinkley alarm was active at that point, the actually-sent
  value may have been clamped, which this reconstruction has no way to
  recover since the clamp's effect isn't stored either.

## Monitoring dashboards (`web/` and `dashboard/app.py`)

Two read-only dashboards exist, both querying Supabase directly and never
writing to it: **`web/`** is a plain HTML/CSS/JS static site (no build, no
framework) deployed to Vercel at
**https://web-sigma-lac-7j9zw6955x.vercel.app** - the "always available"
one, since it needs nothing running locally. **`dashboard/app.py`** is the
same views in Streamlit, for local dev (`streamlit run dashboard/app.py`).
Both compute accuracy (all-time / last 20 / last 4, the same window the
drift gate uses), show a clickable Bogotá map built from `stations`' real
coordinates, the real-vs-predicted series, the collector heartbeat, and the
drift-eligible-stations table. Color palette is red/yellow/amber, sourced
from TransMilenio's actual logo file on Wikimedia Commons (no public brand
manual with documented hex values was found - see `dashboard/app.py`'s
comment for the caveat).

### `web/` has no backend - it calls Supabase from the browser with the public key

This is the one genuinely new architectural pattern in this repo: `web/`
uses `@supabase/supabase-js` with the `anon`/publishable key
(`web/config.js` - safe to hardcode and commit, that key is *meant* to be
public, the same way every Supabase frontend does it) instead of a backend
service. Building it surfaced a real, pre-existing vulnerability
unconnected to the dashboard itself: that public key already had full
`SELECT/INSERT/UPDATE/DELETE/TRUNCATE` on `"Original Data"`, `"Temp"`,
`stations`, and several other tables, with RLS disabled - anyone who ever
obtained it could have wiped production data, before this dashboard
existed.

**Fixed in `database/migrations/009_lock_down_anon_access.sql`**, and this
is the pattern to follow for any future public-facing read: enable RLS on
the table with **zero policies** (RLS-enabled-with-no-policy denies
`anon`/`authenticated` access completely via PostgREST's normal
`/rest/v1/<table>` path, for every operation, on every table it's applied
to - this never touches `service_role`, which the collector/submissions
pipeline uses via `SUPABASE_DATABASE_URL` and which always bypasses RLS
regardless of policies), then expose exactly what's needed through a
**narrow `SECURITY DEFINER` SQL function** instead of a blanket table
grant - PostgREST auto-exposes every function as its own endpoint at
`/rest/v1/rpc/<function_name>`. The four functions
(`api_station_list`, `api_station_summary`, `api_station_series`,
`api_job_runs`) each: run with `SET search_path = ''` and fully
schema-qualify every identifier (the standard SECURITY DEFINER hardening -
otherwise the function trusts the caller's search_path), aggregate data
server-side where possible (`api_station_summary` never ships raw
per-row demand to the client, just the computed accuracy numbers), and
cap any row-returning limit parameter server-side
(`LEAST(GREATEST(p_limit, 1), 2000)` in `api_station_series` - a client
can't request the whole, ever-growing table by passing a huge limit).
`api_job_runs` also avoids exposing the `ops` schema at all (not in
Supabase's default exposed-schema list) by living in `public` and
querying `ops.job_runs` internally.

**Don't add a new public-facing read by granting `SELECT` on a raw table
to `anon`** - even a read-only grant re-opens exactly this class of
problem the moment RLS's default-deny gets bypassed by a permissive
policy, and it ships more data than any dashboard view actually needs.
Add another narrow function instead, following the same four as a
template.

Verified live with the real public key before trusting the fix: a direct
table read now returns `[]`, a direct write is explicitly rejected
(`"new row violates row-level security policy"`), and all four RPC
endpoints return real data.

## The teacher's repo (source of truth for the rules) - read 2026-10-03

**https://github.com/uexternadojz/pulso-transmi** (public, default branch `main`)
is the professor's platform: FastAPI + Postgres + scheduler + portal + docs. It is
NOT a template for this repo and prescribes no folder layout - students own their
own pipeline. Read it with `gh api repos/uexternadojz/pulso-transmi/contents/<path>
--jq .content | base64 -d`. Where it and this skill differ on API behavior, the
live API and its `docs/api-contract.md` win (the teacher's guide says so itself).
Most useful files: `docs/api-contract.md` (endpoints, errors, submission rules),
`docs/fase-final.md` (v2 observations, final-phase evidence), `docs/fase-drift.md`
(what the project must be able to explain), `docs/primer-corte-evaluacion.md`
(Corte 1 definition), `docs/guides/pulso-transmi-guia-operativa-v2.0.md` (the
operational guide and "project is ready when" checklist), `docs/runbook.md`.

Course rules that matter for decisions in this repo:

- **Deadline: Sunday 2026-10-04 23:59 America/Bogota** (final phase, `0.9.0`). The
  gap between the previous close and reopening creates no cycles and no absences.
- **Cycles:** one per hour, 25-minute delivery window, 12 stations x 4 horizons =
  48 targets, always taken from `GET /v1/forecast-cycles/current`, never from a
  local clock or cron. Observations release every 30 min.
- **Submission rules:** max 3 accepted attempts per cycle (the last valid one is
  official; a guardrail rejection does NOT consume an attempt), body <= 64 KB, max
  10 requests/min per key, exact target set only, `training_data_end` <= cutoff,
  stable `Idempotency-Key` reused on retry (same key + different body = 409).
- **Leaderboard:** `cumulative` counts only cycles opened since **Corte 1 =
  2026-09-25 00:00 Bogota** (older history stays in the DB for audit); the portal
  chart's "last 6 cycles" is the last six RESOLVED cycles, not your last six
  submissions. Accepted != evaluated: a cycle only scores once truth is revealed.
- **Observation v2** (after virtual 2026-09-20T12:00Z): `measurement.value` text
  decimal or `null` with `quality: missing` (missing != 0); v1 `demand` rows keep
  coexisting in one page. Evaluation truth stays complete; gaps are training-side only.
- **What the teacher grades on (evidence, not just rank):** the repo must let the
  student explain (1) how ingestion/submission continuity is verified, (2) how
  operational problems are told apart from demand change, (3) what triggers a
  retrain and which data it uses, (4) how versions are compared temporally with no
  future leakage, (5) what justifies keeping/promoting/retiring a version, (6) what
  happened before/during/after a detected change. Changing a model label is explicitly
  NOT evidence of retraining. Ranking is evidence, not an automatic grade; weights are
  set by the teacher. Keep README/decision records answering these.
- **Teacher's "ready" checklist** (guia operativa): no duplicate data, cursor only
  advanced after a confirmed write, champion with version + metadata + stable
  location, cron plus manual run, 404 = green exit, delivered cycle never re-POSTed,
  receipts persisted with model and commit, **training and inference in separate
  workflows**, accuracy/coverage/drift explainable. Known gap vs. this repo: retrain
  runs inside `collector.yml` rather than its own workflow, and receipts store no git
  commit; the loops are self-looping instead of the recommended 10-minute cron.
  Neither is a violation, but be ready to justify (see "Why workflows self-loop").
- Secrets never in the repo (`PULSO_API_KEY` etc. only as Actions secrets). The
  public Supabase key in `web/config.js` is meant to be public (see dashboards).

### Repo hygiene (cleanup 2026-10-03)

The repo is Actions + Supabase only. Removed as dead: the Docker stack (`Dockerfile`,
`docker-compose.yml`, `.dockerignore`, `app/scheduler.py`, `app/prediction_scheduler.py`),
`database/init/000_roles.sh` (Docker-Postgres roles), `app/train_comparison_models.py`
(exploratory), empty `src/ data/ deploy/ notebooks/ workflows/ config/`, and the
unused `statsmodels`/`httpx` requirements; `.env.example` now lists only the 4 real
variables. Don't re-add a Docker/local-scheduler path - it contradicts "no local
component". Keep: `dashboard/` + `.streamlit/` (documented local dashboard),
`web/`, `docs/figures/` (README references), `app/load_original.py` (restore
workflow), `scripts/migrate_supabase.py`, all `database/migrations/*` (applied history).

## Pulso TransMi API quirks worth remembering

- `GET /v1/forecast-cycles/current` - 404 with `no_open_cycle` is a normal
  "between windows" state, not an error; `submit_current_cycle()` returns
  `None` in that case.
- `POST /v1/submissions` - idempotent via the `Idempotency-Key` header.
- `GET /v1/leaderboard?window=cumulative|rolling_24h` - **only these two
  window options exist**. There is no per-cycle or per-N-submissions view.
  Don't try to isolate "the last submission's accuracy" from these without
  flagging to the user that the math is fragile (see below).
- `GET /v1/stream/observations` - cursor pagination; `next_cursor: null`
  means "caught up to the live edge," not an error. Don't raise on it.
- **A missed cycle scores as prediction=0** for all its targets in the
  pooled WAPE. This means the leaderboard's cumulative/rolling accuracy can
  look catastrophically bad (e.g. single-digit %) even when the model's
  actual submitted predictions were fine - always check coverage
  (fraction of targets actually submitted) before concluding the model
  itself is broken.
- Reverse-engineering "true" model accuracy from leaderboard deltas
  (`raw_wape`, `coverage`) is unreliable when the target-pool growth rate
  between two snapshots is uncertain (real vs. virtual competition clock).
  **Prefer** reconstructing predictions against now-revealed ground truth:
  once the virtual clock has passed a past cycle's target timestamps, query
  `Temp`/`Original Data` directly for the real values and compare to what
  the *current* code would have predicted for that exact cutoff. This is
  far more trustworthy than parsing cumulative/rolling aggregates.

## Model payload privacy

The submission payload's `model` field is visible to the competition
(other students/professor), so it must not reveal implementation details:
`version` is a generic string like `"v1"` (not e.g. `"xgboost-per-station:1.0"`),
there is no `git_commit` field (that would point straight at the public
repo), and `client_run_id`/`Idempotency-Key` avoid naming the algorithm.
Keep it this way when touching `submit_current_cycle()`.

## Known past bugs (avoid reintroducing)

1. **Lag features computed relative to `data_cutoff` for all 4 horizons**
   instead of per-target - silently broke +30/+45/+60min (75% of every
   submission). Fixed via the chronological-recursion pattern above.
2. **History query read only `"Original Data"`**, stale for any
   non-drifted station - happened independently in both
   `submit_xgboost.py` and `collector.py`'s `predict_records()`. Fixed by
   always reading `"Original Data"` + `"Temp"` together.
3. **`next_cursor: null` treated as a `RuntimeError`** instead of the
   documented "drained the stream" signal. Fixed in `collect_new_data()`.
4. **Bash `\\` instead of `\` line continuations** in workflow YAML curl
   blocks silently mangled the model-download step into garbage commands.
5. **`np.True_ is True`** - numpy bools fail Python `is` comparisons in
   tests; always wrap statistical-test booleans with `bool(...)` before
   returning/asserting on them.
6. **The collector cursor was never persisted, ever.** `collect_new_data()`
   only saved `next_cursor` `if next_cursor is not None`, but the API
   returns `null` on the page that reaches the live edge - which, once
   caught up, is *every single page of every single run* (steady state, not
   a corner case). Since `collector_state` never accumulated a row, every
   30-min run re-fetched and re-processed the entire stream from scratch.
   Fixed by building a fallback resume cursor from the last record's own
   `(released_at, observed_at, station_id)` - verified byte-for-byte
   identical to the server's own cursor encoding for the same record, and
   confirmed to round-trip correctly - whenever the server doesn't hand
   back a real one. This relies on undocumented server internals (the API
   only guarantees `cursor` is an opaque `string | null`), so `_fetch_page`
   falls back to a full refetch (not a crash) if a saved cursor - real or
   synthetic - ever gets rejected with a 4xx. See `_synthetic_cursor` and
   `_fetch_page` in `app/collector.py`.

   Downstream side effect worth remembering if promotion counts ever look
   odd: while this bug was live, every run refetched the full stream, so
   already-promoted rows kept getting re-inserted into `"Temp"` for
   stations that had already been retrained that day. `ON CONFLICT
   (station_id, observed_at) DO NOTHING` on the `"Temp"` insert silently
   absorbed the exact duplicates, but distinct-looking `Original Data`
   count deltas after a later retrain can still reflect this: a station
   that "should" have inserted N new rows on promotion may show fewer,
   because some of those N were already sitting in `Original Data` from an
   earlier same-day retrain and got silently skipped by the promotion
   step's own `ON CONFLICT DO NOTHING`. This is a one-time artifact of the
   pre-fix duplication, not a bug in the KS-test/promotion logic itself,
   and it stops recurring now that the cursor actually persists.
7. **`app/collector.py`'s `predict_records()` was missed when `lags_for()`
   was introduced (2026-09-26).** It builds the h1 feature vector for
   *every* station's drift-monitoring nowcast, but kept iterating the fixed
   `LAGS` tuple directly instead of calling `lags_for(station_id, 1)`. The
   moment 09122's h1 model (one of the `NO_LAG672_MODELS` overrides, 8
   features) got redeployed, every single call sent it 9 features instead -
   `ValueError: Feature shape mismatch, expected: 8, got 9`, unrecoverable
   (it hits on every record in every batch), which then failed the whole
   `submissions.yml` job after its 3-strikes limit. **Lesson: `lags_for()`
   has more than one call site** - `submit_xgboost.py`'s
   `predict_cycle_targets` AND `collector.py`'s `predict_records` both build
   feature vectors for a saved model, and both must go through it. When
   adding a new per-`(station, horizon)` architecture override, grep for
   every place a `.joblib` model's `.predict()` gets called
   (`grep -rn "\.predict(" app/`) and check each one, not just the one you
   were already editing.
8. **Schema drift between `database/init/001_schema.sql` and the live
   Supabase DB.** The init file is aspirational/only used for fresh
   installs; the real DB is built incrementally via
   `database/migrations/*.sql`, and a migration can silently fail to keep
   up with what the init file already declares. Concretely:
   `"Original Data"` was missing its `prediction` column in production
   (migration 007 added `prediction` to `"Temp"` only, never to
   `"Original Data"`, even though `001_schema.sql` declared it on both)
   until `008_add_original_data_prediction.sql` was written and applied
   directly to Supabase. This crashed `station_accuracy_stats()`'s new
   UNION query (`psycopg.errors.UndefinedColumn`) the moment the 6h-window
   drift trigger shipped and tried to read `"Original Data"` for the first
   time. Unit tests never catch this class of bug since DB access is fully
   mocked - if a query touches a column/table shape, sanity-check it
   against the live schema (or at minimum against every migration file in
   order), not just against `001_schema.sql`.
9. **`web/app.js` declared `const supabase = window.supabase.createClient(...)`.**
   The `@supabase/supabase-js` CDN script (loaded via a plain `<script>` tag
   before `app.js`, not an ES module) already creates its own global
   `window.supabase` - redeclaring that identifier with `const` at another
   `<script>`'s top level throws `Identifier 'supabase' has already been
   declared` and kills the whole script before `main()` (or any function in
   the file) ever runs. Symptom in the browser: the page sat on "Cargando
   datos..." forever, because even the page's own error-handling code never
   got a chance to execute - the error happens before any of it is defined.
   `curl` never shows this class of bug (the HTML/JS files serve fine over
   HTTP; the failure is a JS runtime error, not a network one) - it only
   surfaced once tested with a real browser. Fixed by renaming the local
   variable to `supabaseClient`. When wiring up a third-party script loaded
   via a global `<script>` tag (not a module import), check what globals it
   creates before naming a local variable the same thing.

## Diagnosing a stuck or slow-looking workflow run

`gh api .../actions/jobs/{id}/logs` returns `BlobNotFound` (404) for a job
that is still `in_progress` - GitHub only finalizes/stores logs once a job
completes or is cancelled. So when a run looks stalled and you need to see
*why*, you can't just fetch its logs. Sequence that works:

1. Check `pg_stat_activity` on Supabase first, to tell "genuinely slow/
   CPU-bound" apart from "blocked on a DB lock" - a real deadlock shows an
   active/waiting query, a merely-slow-but-fine run doesn't.
2. If nothing's stuck at the DB level and you still need to see what
   happened, `gh run cancel <run-id>` - this forces the job to finalize and
   its log becomes readable via the API immediately after.
3. Re-trigger the workflow (`gh workflow run <file>.yml`) right after,
   since the cancel was purely diagnostic and the pipeline must not sit
   idle - this is safe even mid-loop-sleep, since a self-looping workflow
   idling between iterations has no in-flight work to lose.

This is also the technique that surfaced bug #7 above (the missing
`prediction` column) - the run looked hung for 30 minutes because it had
already failed once and was sleeping through its retry-backoff wait, not
actually stuck.

## Testing conventions

`pytest tests/` - all DB access is mocked via small `FakeConnection`
classes (see `tests/test_drift.py`, `tests/test_health.py`,
`tests/test_collector.py`) that implement `__enter__`/`__exit__`/`execute`/
`fetchall`/`fetchone` and either return canned rows or route on a substring
of the SQL query. No real Supabase connection is used in tests. Run the
full suite after any change to `app/collector.py`, `app/submit_xgboost.py`,
`app/drift.py`, or `app/health.py` - these are the modules most prone to
the "stale history" and "wrong lag" class of bug above.

**`web/` has no Python test suite - it needs a real browser, not `curl`.**
`curl`/static checks only prove the files serve over HTTP; they say nothing
about whether the page's JavaScript actually runs (see bug #9 above, which
`curl` couldn't have caught). To verify a change to `web/`, launch a real
headless browser and check the DOM after load, e.g. Puppeteer:
`npm install puppeteer` (may need `apt-get download libnspr4 libnss3
libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libxkbcommon0
libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0
libasound2t64` + `dpkg-deb -x <pkg>.deb <dir>` + `LD_LIBRARY_PATH=<dir>/usr/lib/x86_64-linux-gnu`
if Chrome fails to launch with a missing `.so` and there's no root/sudo),
then `puppeteer.launch({args: ["--no-sandbox"]})`, capture both
`page.on("console")` (filter `type() === "error"`) and
`page.on("pageerror")`, and assert on real page state after load (e.g. the
loading spinner actually hid, KPI values aren't still their placeholder
text, the map has marker elements). Test against the deployed Vercel URL
too, not just a local `python -m http.server` copy, before calling a
change verified - a passing local test doesn't guarantee the same files
serve identically once deployed.

## Working conventions the user expects

- **No local execution required to operate the pipeline.** Any fix must
  work unattended via GitHub Actions + Supabase.
- **Verify against live data before trusting an assumption.** The user has
  repeatedly asked to check real Supabase rows / live API responses rather
  than reasoning from code alone, especially for "why is accuracy X"-type
  questions.
- **Flag statistical/design soundness concerns before implementing
  anything speculative** (e.g. the KS-test regime-shift design was
  discussed and confirmed before being built).
- **Ask before pushing changes with real production side effects**
  (triggering workflows, cancelling in-flight runs, changing submission
  cadence) - minor same-session tweaks have sometimes been pushed directly
  without objection, but default to confirming for anything with live
  competition consequences.
- Secrets live in GitHub Actions repo secrets: `SUPABASE_DATABASE_URL`
  (exposed to the app as `DATABASE_URL`), `SUPABASE_URL`,
  `SUPABASE_SERVICE_ROLE_KEY`, `PULSO_API_KEY`. Never print or log these.
