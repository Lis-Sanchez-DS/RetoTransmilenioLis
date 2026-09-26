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

Both download models fresh from Supabase Storage on every loop iteration
via `scripts/download_models.sh` (not just once at job start) - this matters
because a drift retrain mid-run needs to reach the *other* long-running job
quickly, not after it happens to restart hours later.

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

### Per-(station, horizon) feature set overrides (`lags_for`)

`lag_672` (a full week back) anchors every model to "what happened at this
same time last week" - a good default normally, but actively counterproductive
while a station's demand pattern is genuinely shifting (05100's collapses
are the clearest case). Tested 2026-09-26 via **two independent
chronological folds** (the 70-85% region, and the standard last-15% test
split), per station per horizon, comparing the untouched 9-feature baseline
against (a) dropping `lag_672` from the feature set entirely and (b)
softly down-weighting it (`feature_weights` in the `XGBRegressor`
constructor, weight 0.2, with `colsample_bynode=0.8` - `feature_weights` is
a silent no-op at the default `colsample=1.0`, since with no column
subsampling every feature is used at every split regardless of its
weight). A candidate only earns a spot in production if it beat the
baseline by **>= 0.3 pp on BOTH folds independently, not just one** - the
same discipline used everywhere else in this project, because plenty of
station/horizon pairs looked like a clear win on a single split and simply
didn't repeat on the second one (the *direction* - down-weighting or
dropping `lag_672` helps - was robust almost everywhere, but the *exact*
optimal weight value bounced around between folds for most stations,
which is why a full drop, not a hand-tuned weight, was preferred wherever
it cleared the bar). Final production config
(`NO_LAG672_MODELS`/`SOFT_DEEMPHASIZE_LAG672_MODELS` in `app/drift.py`):

| station | horizon(s) | treatment |
|---|---|---|
| 05100 | h3 | `lag_672` dropped entirely (8 features) |
| 06111 | h2, h3 | `lag_672` dropped entirely |
| 07111 | h2 | `lag_672` dropped entirely |
| 09122 | h1, h2, h4 | `lag_672` dropped entirely |
| 05000 | h2 | soft down-weight (weight 0.2, `colsample_bynode=0.8`, still 9 features) |
| everyone else | all horizons | untouched 9-feature baseline |

`lags_for(station_id, horizon)` is the single source of truth for which
lags a given saved model expects - both `_train_station_horizon_model` (at
train time) and `predict_cycle_targets` (at inference time) call it, so the
feature vector built for prediction always matches what that specific
`.joblib` file was actually trained on. **Never train or predict for one of
the overridden pairs using the full `LAGS` tuple directly** - a dimension
mismatch there doesn't raise a friendly error, it just silently corrupts
predictions or crashes deep inside XGBoost. If this set is ever revisited,
repeat the two-independent-fold methodology above rather than trusting a
single split, and retrain + re-upload every affected `.joblib` file in the
same change (see "Deploying a feature-set change" below) - leaving the code
change unaccompanied by a retrain corrupts inference immediately, since the
currently-deployed model file still has the old feature count.

### Deploying an architecture change to specific (station, horizon) models

Adding a station/horizon to `NO_LAG672_MODELS` (or any future per-model
architecture change) only changes what a *newly trained* model looks like -
the already-deployed `.joblib` file for that pair is untouched and still has
the old feature count/schema. Since `predict_cycle_targets` now builds its
feature vector via `lags_for()` (the *new* logic), pushing the code change
alone, without retraining, creates an immediate mismatch between what the
code sends and what the old file expects - this fails loudly inside
XGBoost's C++ layer, not as a friendly Python exception. A change like this
is not complete until both steps happen together:

1. Ship the code change (`NO_LAG672_MODELS`/`SOFT_DEEMPHASIZE_LAG672_MODELS`
   in `app/drift.py`, and anywhere else that reads `lags_for()`).
2. Immediately retrain and re-upload every affected `(station, horizon)`
   pair's model on current full history, using the exact same
   `_train_station_horizon_model` / `_upload_model` functions a real drift
   retrain would call (not a hand-rolled equivalent) - loading `.env` into
   `os.environ` first (note: the DB URL there is `SUPABASE_DATABASE_URL`,
   but `app/db.py` reads `os.environ["DATABASE_URL"]`, so it must be copied
   across or the connection call raises `KeyError`).

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

### Network resilience (`app/net.py`)

`with_retries()` wraps outbound HTTP calls (both to the Pulso API and, via
`curl --retry` in `scripts/download_models.sh`, to Supabase Storage).
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
(most likely `lag_672` = one week back, which needs an unbroken week of
data), `predict_cycle_targets` **skips only that target/horizon** and logs
a warning - it does not abort the whole station or cycle. (Before
2026-09-26 this checked all of `LAGS` up front per station and skipped the
*entire station* on any single missing lag; that granularity broke once
some horizons stopped needing `lag_672` at all - see "Per-(station,
horizon) feature set overrides" above - since a station could easily have
full history for the lags one horizon needs but not another's. Preserve the
per-target granularity; don't revert to the coarser per-station check.)

Each horizon is a **separate model file**, loaded as
`models[(station_id, horizon)]` from
`{MODEL_DIR}/xgboost_{station_id}_h{horizon}.joblib` - not one shared model
called four times. `download_models.sh` and `drift.py`'s upload step must
stay in sync with this per-`(station, horizon)` naming. Which lags go into
that file's feature vector is **not** always the same fixed `LAGS` tuple
any more - see `lags_for()` above; always build the prediction feature
vector via `lags_for(station_id, horizon)`, never by assuming every
horizon of every station uses all 5 lags.

## Data feed stalled (ongoing, not a pipeline bug)

Since 2026-09-13 the Pulso API's observation stream has been stuck -
`observed_at` stopped advancing while `released_at`/`server_time` keep
moving normally. This is upstream/server-side, not something to "fix" in
this repo. This stall is what originally motivated the `has_pending_data()`
gate above - a stalled feed that still returns duplicate records every 30
min would otherwise re-run drift checks across all 12 stations for no
reason - but it also directly caused the gate's original ("did this run
insert something new") version to permanently jam retrain for any station
whose data was already sitting in `Temp` when the feed died (see above). If
accuracy or drift behavior looks strange, check whether `observed_at` in
fresh `Temp` rows is actually advancing before assuming the model or the
drift logic is at fault - and check whether `"Temp"` already holds
unpromoted, unretrained data for the station in question.

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
7. **Schema drift between `database/init/001_schema.sql` and the live
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
