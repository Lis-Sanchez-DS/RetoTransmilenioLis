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
points) falls below `ACCURACY_THRESHOLD=0.85`. A fixed count survives the
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

**One model per station per horizon, not one recursive model.** Since the
direct multi-horizon change (see "Multi-horizon prediction" below), a
drifted station retrains and republishes all 4 horizon models
(`_train_station_horizon_model`, called once per horizon in
`HORIZONS = (1, 2, 3, 4)`), each uploaded to Supabase Storage as
`POST /storage/v1/object/models/xgboost/xgboost_{station}_h{horizon}.joblib`
with `x-upsert: true` - 48 files total across 12 stations, not 12.

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
iteration downloads the ~4 files that changed (one station's 4 horizons),
not all 48. The manifest lives alongside the models in the same job's
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

## Multi-horizon prediction (critical correctness point)

`predict_cycle_targets()` in `app/submit_xgboost.py` predicts each
station's 4 targets (+15/+30/+45/+60 min from `data_cutoff`) **in
chronological order**, feeding each prediction back into a local
per-station history dict as the stand-in for the not-yet-observed value the
next horizon's lags need. This is required because `lag_k` means "demand
exactly k*15 minutes before the *target* timestamp", which only equals
"k*15 minutes before `data_cutoff`" for the +15min horizon. **Do not**
compute all 4 horizons' features relative to `data_cutoff` directly - that
was a real, previously-shipped bug that silently broke 75% of every
submission (see below). `app/collector.py`'s `predict_records()` already
uses the correct pattern for its own (single-step) predictions - mirror it
rather than reinventing.

If a station is missing history for a lag its target horizon actually needs
(most likely `lag_96` = one day back, the longest lag now that `lag_672` is
gone - see below), `predict_cycle_targets` **skips only that target/horizon**
and logs a warning - it does not abort the whole station or cycle. (Before
2026-09-26 this checked all of `LAGS` up front per station and skipped the
*entire station* on any single missing lag; that granularity briefly
mattered while different (station, horizon) pairs used different lag sets,
since a station could have full history for one horizon's lags but not
another's. Preserve the per-target granularity regardless; don't revert to
the coarser per-station check.)

Each horizon is a **separate model file**, loaded as
`models[(station_id, horizon)]` from
`{MODEL_DIR}/xgboost_{station_id}_h{horizon}.joblib` - not one shared model
called four times. `download_models.py` and `drift.py`'s upload step must
stay in sync with this per-`(station, horizon)` naming. As of 2026-09-27,
every model (all stations, all horizons) shares the exact same `LAGS =
(1, 2, 4, 96)` feature set - `collector.py` and `submit_xgboost.py` both
import `LAGS` directly from `app.drift`, there is no more per-pair
variation to look up (see "`lag_672` was removed entirely" above).

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
still the right mental model, not a permanent halt.

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
  aggregate.

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
