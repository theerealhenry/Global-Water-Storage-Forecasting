# Phase 3 Handoff — TWS Forecasting Project

**Read this document first, before touching any code.** It is a complete, self-contained briefing for
picking up this project in a brand-new chat session with no prior context. Written 2026-08-13, immediately
after Project Phase 2's formal closure.

---

## 0. Before anything else: connect to the project folder

This project lives at **`D:\PROJECTS\tws-forecast`** on the user's machine (a git repository, remote at
`https://github.com/theerealhenry/Global-Water-Storage-Forecasting.git`). The very first thing to do in a
new session is confirm you have file access to that folder (in Cowork: the user needs to have selected/
connected it, or you request it) — nothing below is actionable without it. Once connected, read, in this
order: this document in full, then `docs/PROJECT_PLAN.md`, `docs/ARCHITECTURE.md`, and
`docs/ASSUMPTIONS.md` (this document summarizes all three below, but they are the authoritative source if
anything here seems to conflict). Do not start building Phase 3 code before doing that — this document
tells you *what* to build; those three tell you the full governing context *why*.

The raw competition data (`data/raw/Train.csv`, `Test.csv`, `SampleSubmission.csv`) is present on disk and
gitignored; its identity is pinned in `data/raw/dataset_manifest.json` (SHA-256 + row counts). Do not
re-download or modify it. A small committed golden fixture (`tests/data/golden/`) is used by the test
suite instead of the full files.

---

## 1. What this project is (30-second version)

Competition: ITU/UN "AI for Good" — forecast next-month Total Water Storage (TWS) anomaly at 15,715 fixed
global land-only grid locations, from GRACE/GRACE-FO satellite gravimetry data. Training data: 2,154,021
rows, May 2002–Aug 2015. Test data: 280,961 rows, 18 non-contiguous months Sep 2015–Dec 2018. Prize pool
€2,000; score is 50% leaderboard RMSE + 50% a written Trustworthiness/Innovation report for top-10
finishers only. Three goals, in order: win it, build genuine DS/ML skill, produce a senior-level portfolio
piece. Full detail: `docs/ARCHITECTURE.md` §1, `docs/PROJECT_PLAN.md` header.

**The one idea that governs every design decision in this repo:** this is not a simple regression problem
(`TWS_t, SPEI, soil_moisture → TWS_t+1`). It is a **partially-observed state-reconstruction and
state-transition problem**. 66% of test rows have no current TWS observation at all — masked in contiguous
whole-month blackout blocks that mirror the real 2017–2018 GRACE→GRACE-FO satellite gap, not scattered
per-row missingness. A model that only works when `TWS_t` is present fails on two-thirds of the leaderboard
by construction. Everything from validation design to feature architecture to the model ladder exists to
serve this reframing. Full reasoning: `docs/ARCHITECTURE.md` §3, `docs/PROJECT_PLAN.md`'s "Central
hypothesis" section (read this if nothing else — it has the architecture diagram every later phase builds
toward).

**Governing document precedence** (`ARCHITECTURE.md` §2): Zindi's rules > `ARCHITECTURE.md` (what the
system is) > `PROJECT_PLAN.md` (what order, what's done) > `COMPETITIVE_ANALYSIS.md` (why this approach).
Evidence that contradicts a document triggers an ADR (`docs/adr/`), never a silent pivot. Two living
registers sit alongside: `docs/ASSUMPTIONS.md` (things believed, not yet proven — status Active/Validated/
Rejected) and `docs/OPEN_QUESTIONS.md` (things genuinely unresolved).

---

## 2. What has been built and proven so far (Phases 0–2, all formally closed)

### Project Phase 0 — Foundation
Git repo, `src/` layout, pinned Python 3.11 deps, `pandera` schema contracts, `dataset_manifest.json`
(hash-pinned data), flat experiment log (`reports/experiments/experiment_log.csv`). No DVC (ADR-0002 —
static externally-supplied data, manifest+hash is sufficient). Complete.

### Project Phase 1 — Forecastability & data-generating process
Two executed notebooks (`notebooks/01_eda.ipynb`, `notebooks/02_forecastability.ipynb`), 7 ordered
experiments, all complete. **The results every later phase is built on:**

- **Masking is real, not synthetic-competition noise.** Blackout months mask 99.58–99.97% of the grid at
  once (never a flat ~66% row-independent rate); the test set's absent-month pattern matches the *documented*
  GRACE→GRACE-FO mission gap (JPL sources, Landerer et al. 2020) almost exactly. **A-001, Validated.**
- **The 2015 persistence-RMSE anomaly is resolved.** Persistence RMSE is stable at 0.50–0.63 from
  2002–2014, then jumps to 0.898 in 2015 — confirmed genuine (episodic, broad-based, global, directional),
  plausibly El Niño-linked, **not** an artifact to discard. Practical consequence: don't assume 2002–2014
  volatility is representative of 2016–2018. **A-004/A-007.**
- **The blackout-degradation curve** (RMSE vs. months-since-observation, k): pooled RMSE grows 0.537→0.884
  over k=1→9. Per-location ACF(1) quartile is the dominant stratifying factor (0.403 RMSE spread at k=9) —
  bigger than latitude, season, or drought regime combined. **This is why historical-signature/ACF features
  are Phase 4's top priority. A-008, Validated.**
- **Four baselines, measured on the real replayed 18-month test structure** (Method B: direct replay of the
  real FULL/BLACKOUT offset pattern onto 8 windows of the verified clean 2004–2010 span, ground-truth
  scored): **Baseline A (oracle persistence, FULL months only) = 0.5247** (n=747,365). **Baseline B
  (last-known-state, BLACKOUT months only) = 0.7145** (n=1,491,960). **Baseline C (seasonal climatology,
  all 18 months) = 0.8170.** **Baseline D (Hybrid: A when observed, B when masked, all 18 months) =
  0.6573** (n=2,239,325) — **this is the realistic "do nothing clever" floor**, not the in-sample 0.572.
  **A-009, Validated.** These are Phase 1 *preview* numbers (informal replay script, not the harness);
  Phase 3's job is to recompute them *inside* the now-built validation harness for the official record.
- **Staleness × ACF is a real but nonlinear interaction** (AR(1)'s ρᵏ form, not a linear product term) —
  R²=0.448 of the degradation curve's variance explained by a parameter-free AR(1) model. Also: TWS
  volatility (σ) is a largely independent, comparably important signal (r=-0.141 vs ACF). **A-010,
  Validated (nonlinear).**
- **Covariate shift train→test is minimal** on SPEI/soil moisture (KS<0.04); the dominant train/test
  difference is the masking regime itself (0% vs 66.5%), which is the project's whole design focus, not a
  new risk. **A-003 supported (indirect), A-011** (test entirely omits October — a real, actionable
  calendar-coverage gap for Phase 2 fold design and Phase 4 seasonal features).

Exact per-experiment numbers, methods, and full reasoning: `docs/PHASE1_FINDINGS_SYNTHESIS.md` and
`docs/PROJECT_PLAN.md`'s Phase 1 section (both are complete, verbose write-ups — read them if a Phase 3
question needs Phase 1 detail beyond this summary).

### Project Phase 2 — Validation harness
**This is the most important thing to understand before starting Phase 3: a complete, tested,
production-grade validation engine already exists and works.** Do not build any new CV/masking/scoring
logic in Phase 3 — call into what's here.

**Module map** (`src/tws_forecast/`, 211 tests passing, 12 commits-worth of steps 2.1–2.12, each
individually tested and documented in `docs/PHASE2_EXECUTION_PLAN.md`):

- `utils/seeds.py` — `RANDOM_SEED = 42`, `set_seed()`. Every stochastic call in this repo seeds explicitly.
- `utils/config.py` — minimal pydantic loader for `configs/base.yaml`.
- `utils/dates.py` — `month_index()` / `month_index_to_timestamp()` shared calendar-arithmetic helpers.
- `state/reconstruction.py` — `ForecastOrigin` (frozen dataclass: `origin_time, target_time, horizon,
  information_cutoff, location_id, regime`) and `location_id_from_lat_lon()`. Every fold, masked example,
  and decomposition row is keyed by this, never a raw row index. `StateSnapshot` (the state-reconstruction
  layer proper) is added to this **same file** in Phase 4 — don't create a new file for it.
- `validation/phase1_constants.py` — every Phase 1 number above, as typed constants, each with a docstring
  citing its source notebook section: `BASELINE_A/B/C/D`, `BLACKOUT_K_DISTRIBUTION`, `TEST_FULL_OFFSETS`,
  `TEST_BLACKOUT_OFFSETS`, `BLACKOUT_K_BY_OFFSET`, `CLEAN_TRAIN_SPAN_START/END`, `ACF_QUARTILE_AR1_PARAMS`,
  `PROMOTION_THRESHOLDS`. **Import from here, never re-derive or hardcode a competing copy.**
- `validation/splitters.py` — `expanding_window_splits()` (time-respecting only, never random K-fold) and
  `attach_forecast_origin_columns()`. Earliest fold's training always covers the clean 2004–2010 span;
  final fold's validation window reaches into the 2015 anomaly deliberately.
- `validation/masking_simulator.py` — `MaskingScenario` (declarative blackout config) + `apply_masking()`
  (one scenario) + `apply_blackout_curve()` (per-location independently-drawn-k blackout runs, the Tier 2
  mechanism). Streak-aware — never row-independent random masking.
- `validation/scenarios.py` — config-driven registry reading `configs/validation/*.yaml`
  (`expanding_window`, `blackout_curve`, `test_regime_replay`, `2015_like`). `load_scenario(name)`.
- `validation/tiers.py` — the three validation tiers, each a `run_tierN(model, df) -> TierResult`:
  - **Tier 1** (`run_tier1`): standard forecastability, no masking, `expanding_window_splits`.
  - **Tier 2** (`run_tier2`): blackout tier, `apply_blackout_curve` injected into each fold's validation
    window — this is the tier structurally analogous to Baseline D's real mixed regime, and the one
    `promote()` evaluates its ladder against.
  - **Tier 3** (`run_tier3`): real-calendar replay onto historical analogs, **diagnostic/robustness only,
    never a promotion criterion on its own** (enforced in code — `harness.promote()` raises if called
    without Tier 1 + Tier 2). **Important, hard-won limitation to know before Phase 3 touches Tier 3: this
    tier scores each of the 18 replay offsets independently — it never lets a stateful predictor's internal
    "last known value" update from earlier-in-pattern FULL-offset observations within the same replay
    window.** This is deliberate (built for Phase 4's future feature-based models, which read history
    through an explicit row-level column, not internal memory) but it **under-scores internally-stateful
    baselines** like last-known-state persistence. Full write-up, and the workaround (a diagnostic
    sequential-state replay pattern), in **`docs/ASSUMPTIONS.md` A-013** — read it before writing any
    Baseline B/D Tier 3 code. `Predictor` is the protocol every model implements: `fit(train_df) -> None`,
    `predict(df) -> np.ndarray`.
  - `_select_replay_anchors()` was a real, found-and-fixed bug this phase (anchors weren't restricted to
    the verified clean 2004–2010 span) — now fixed and regression-tested (`tests/test_tiers.py`).
- `validation/decomposition.py` — `decompose(tier_result, acf_lookup=None) -> pd.DataFrame`: the standard
  error-decomposition table (overall / regime / staleness_bucket / staleness_x_acf_quartile / hemisphere /
  extreme_target / rapid_change slices). `degradation_slope(decomp_df)`: empirical vs. AR(1)-theoretical
  RMSE(k) comparison per ACF quartile. **Every model in this project gets this table, never a single
  aggregate RMSE** — this is how Phase 3's baselines must be reported.
- `validation/leakage_tests.py` — 4 mechanical leakage checks (`future_row_shuffle_test`,
  `historical_only_check`, `rolling_window_cutoff_check`, `masking_simulator_no_leak_check`) plus a static
  disallowed-feature-name scanner. Proven against both correct and deliberately-leaky toy examples.
- `validation/harness.py` — the orchestrator. `evaluate_candidate(model, df, candidate_id, acf_lookup=None,
  include_tier3=True, n_anchors=3) -> CandidateReport` runs Tier 1+2(+3) and builds all decomposition
  tables/slopes in one call. `promote(report, baseline_report=None) -> PromotionDecision`: evaluates the
  ladder (`PROMOTION_THRESHOLDS`, against Tier 2's overall RMSE) and, if `baseline_report` given, **blocks
  promotion if the candidate regresses on hard staleness buckets k=5/6/7 relative to the baseline, even
  while improving in aggregate** — this safeguard is proven to have teeth (see below).
- `validation/experiment_log.py` — `log_candidate(report, decision=None, model_name=None, notes="", ...)
  -> LoggedExperiment`. Writes one row to `reports/experiments/experiment_log.csv` (continuing the EXP-NNN
  sequence) **and** one real MLflow run (`mlflow.db`/`mlruns/` at repo root, experiment name
  `tws-forecast-validation`) with metrics, params, and the full decomposition tables + degradation slope
  logged as CSV artifacts — in one atomic call, so the two views can never drift apart.

**The phase-defining proof run** (`notebooks/03_validation_harness.ipynb`, real Train.csv, 2,154,021 rows):
ran Baseline D's own logic and a bare LightGBM (raw columns only) through the full harness.
- **Tier 3 reproduces Baseline D's 0.6573 within tolerance** (0.6319 via the sequential-state diagnostic,
  diff 0.0254 < 0.03) — the harness is confirmed to faithfully reproduce Phase 1's measured reality.
- **Bare LightGBM beats Baseline D in aggregate**: Tier 1 0.5798 vs 0.6380, Tier 2 0.5801 vs 0.6381. Clears
  the `naive_floor` promotion rung.
- **But LightGBM is correctly BLOCKED from promotion over Baseline D head-to-head** — it regresses on hard
  staleness buckets k=5/6/7 despite the aggregate win. This is the hard-staleness-bucket safeguard's first
  real, non-synthetic catch, and the sharpest evidence yet that Phase 4's historical-signature/state
  features are the priority, not optional polish.
- Logged for real: `reports/experiments/experiment_log.csv` rows **EXP-008** (Baseline D logic: tier1=
  0.6380, tier2=0.6381, tier3=0.8270) and **EXP-009** (bare LightGBM: 0.5798/0.5801/0.7669); real MLflow
  runs in `mlflow.db`/`mlruns/`.

Full detail, all numbers, all figures: `docs/PROJECT_PLAN.md` Phase 2 section (STATUS: MET),
`docs/ARCHITECTURE.md` §20, `docs/ASSUMPTIONS.md` A-013, `notebooks/03_validation_harness.ipynb` itself
(fully executed, all outputs saved), `notebooks/figures/01–04_*.png`.

---

## 3. Current repository state (as of this handoff)

```
tws-forecast/
├── data/raw/                       Train.csv, Test.csv, SampleSubmission.csv, dataset_manifest.json
├── src/tws_forecast/
│   ├── data/                       loaders.py, contracts.py (pandera)   — Phase 0/1
│   ├── state/reconstruction.py     ForecastOrigin (StateSnapshot arrives Phase 4)
│   ├── validation/                 phase1_constants.py, splitters.py, masking_simulator.py,
│   │                                scenarios.py, tiers.py, decomposition.py, leakage_tests.py,
│   │                                harness.py, experiment_log.py    — all Phase 2, all tested
│   ├── models/                     __init__.py only — EMPTY, Phase 3 builds baselines.py here
│   ├── features/                   __init__.py only — empty until Phase 4
│   ├── pipelines/, serving/        __init__.py only — empty until Phase 10/13
│   └── utils/                      config.py, dates.py, seeds.py
├── configs/base.yaml, configs/validation/*.yaml (4 scenario configs)
├── notebooks/01_eda.ipynb, 02_forecastability.ipynb, 03_validation_harness.ipynb   (all executed, all
│                                    figures in notebooks/figures/)
├── tests/                          211 tests passing, tests/data/golden/ (small fixed fixture set)
├── reports/experiments/experiment_log.csv   EXP-001..EXP-009 (7 Phase-1 forensics rows, 2 Phase-2 proof
│                                    rows — Tier columns populated for EXP-008/009, N/A for EXP-001..007
│                                    since those were data-forensics, not models)
├── mlflow.db, mlruns/               real MLflow tracking backend, live since Phase 2's proof run
├── docs/                            PROJECT_PLAN.md, ARCHITECTURE.md, ASSUMPTIONS.md, OPEN_QUESTIONS.md,
│                                    COMPETITIVE_ANALYSIS.md, DATA_DICTIONARY.md, PHASE1_FINDINGS_SYNTHESIS.md,
│                                    PHASE2_EXECUTION_PLAN.md, SETUP.md, adr/0001-0005, this file
└── (models/, docker/, .github/workflows/ exist as skeleton, not yet used)
```

**Working conventions this repo has followed strictly since Phase 1** (keep following them in Phase 3):
- **One file, one commit, one descriptive message** — never bundle unrelated files into one commit.
  Exception: a batch of figures generated together gets one commit ("docs: add figures for
  notebook_name.ipynb"), and this handoff's own convention below.
- **Every number in code traces to a Phase 1 source** — no re-deriving or eyeballing a constant that
  `phase1_constants.py` already has.
- **Full decomposition table, never a single RMSE** — every model gets `decompose()`, ideally
  `degradation_slope()` too.
- **Time-respecting validation only** — never random K-fold, anywhere, for any reason.
- **Real git history as portfolio evidence** — commit messages explain *why*, not just *what*, and record
  real findings (including bugs found and fixed) rather than presenting a sanitized narrative.
- Local execution environment: conda env `tws-forecast`, Python 3.11, `D:\CONDA\conda_envs\tws-forecast`.
  `pip install -e .` makes `tws_forecast` importable. `pytest -q` from repo root runs the suite.
- The repo is 11 commits ahead of `origin/main` as of this handoff (push access isn't available from every
  execution environment — confirm with the user whether local commits have been pushed since).

---

## 4. Project Phase 3 — State-aware baselines: detailed execution plan

**Objective** (`PROJECT_PLAN.md`, `ARCHITECTURE.md` §13): establish the exact numbers, decomposed by
regime, that every subsequent model (Phase 5's GBM onward) must beat. Four *distinct* baselines, kept
separate rather than collapsed into one with a fallback bolted on, because conflating them obscures what's
actually being measured. Plus a global mean and a Ridge regression as further reference points.

**Why this phase matters, concretely:** without it, "beats the baseline" has no precise meaning. Phase 2's
proof run already showed a bare LightGBM beating *Baseline D* in aggregate while quietly regressing on the
hardest staleness regime — that finding was only possible because Baseline D existed as a harness-scored,
decomposed reference. Phase 3 is what makes that kind of finding available for *every* future candidate,
not just the one Phase 2 happened to build as a proof-run stand-in.

**Key design decision carried over from Phase 2 that shapes how Phase 3 must be built:** Baseline D's own
predictor logic (if `TWS_t` observed, use it; else use last-known-state) was already implemented once, as
a throwaway class (`BaselineDPredictor`) inside `notebooks/03_validation_harness.ipynb`'s build script,
purely as a trivial stand-in to prove the harness works. **Phase 3 promotes this logic into real, tested,
importable `src/` code** rather than reimplementing it — see step 3.2 below.

### 3.0 — Read A-013 before writing any Tier-3-facing baseline code

`docs/ASSUMPTIONS.md` A-013 documents that `run_tier3` under-scores internally-stateful predictors (exactly
what Baselines B and D are) because it scores replay offsets independently rather than letting the
predictor's own "last known" state update across the replay window. This is not a Phase-3 problem to fix in
`tiers.py` — it's a known, deliberate design property, scoped to affect only non-feature-based, stateful
baselines like this phase's own B and D. Decide up front which of two honest options to take, and say so
explicitly in the Phase 3 notebook:
- **(a) Report Tier 3 for B/D using the same diagnostic sequential-state replay pattern** proven in
  `notebooks/03_validation_harness.ipynb` section 7b (walk the real offsets in chronological order,
  manually updating the predictor's own `_last_known` state after each FULL offset before scoring later
  BLACKOUT offsets). Recommended if you want Baseline B/D's Tier 3 number to be genuinely comparable to
  Phase 1's Method B replay number (0.7145 / 0.6573). Consider promoting this pattern from
  notebook-only-diagnostic to a small, tested, reusable helper (e.g.
  `validation/tiers.py::run_tier3_sequential_state` or similar) if it's going to be reused — a second
  ad hoc copy-paste of the same ~30 lines is worth turning into a real function.
- **(b) Report the standard (row-wise) Tier 3 number with the A-013 caveat attached**, and rely on the fact
  that Tier 3 is diagnostic-only by design (`harness.promote()` never uses it for promotion regardless).
  Simpler, defensible, but the Tier 3 number for B/D won't numerically match Phase 1's replay numbers,
  which may read as a discrepancy to a future reader unless clearly footnoted.

Either is acceptable; **do not silently report the row-wise number as if it were comparable to Phase 1's
0.7145/0.6573 without the caveat** — that would misrepresent what was measured, exactly the mistake A-013
exists to prevent.

### 3.1 — `src/tws_forecast/models/baselines.py`

One new file, six predictor classes, each implementing `validation.tiers.Predictor`
(`fit(train_df) -> None`, `predict(df) -> np.ndarray`) so every one of them plugs directly into
`harness.evaluate_candidate()` with no special-casing:

- **`GlobalMeanPredictor`** — `fit`: store `train_df["target"].mean()`. `predict`: broadcast that constant.
  The absolute floor reference (Phase 1 measured ≈0.912 in-sample).
- **`OraclePersistencePredictor`** (**Baseline A**) — `predict`: return `df["TWS_t"]` directly where
  present. **Explicit fallback policy required**: this baseline is only *meaningful* on rows where `TWS_t`
  is observed (that's its whole definition — "how hard is the unmasked problem"). But `harness.run_tier2`/
  `run_tier3` will call `predict()` on masked rows too, where `TWS_t` is `NaN`. Document and implement a
  clear, simple fallback for those rows (e.g. fall back to the fitted global mean) so the harness doesn't
  crash — and make explicit in the write-up that Baseline A's *masked*-regime numbers are a fallback
  artifact, not the quantity this baseline exists to measure. Its real answer is its **Tier 1** number
  (Tier 1 never masks) and its Tier 2/3 **regime=observed** decomposition slice.
- **`LastKnownStatePredictor`** (**Baseline B**) — `fit`: build a `{location_id: last_observed_TWS}` dict
  from the training fold (sorted by time, last value per location), plus a global-mean fallback for any
  location never observed in the training history. `predict`: for each row, return the last-known value as
  of that row's location (this is pure last-observation-carried-forward — **unlike Baseline D, it does NOT
  prefer the row's own current `TWS_t` even when available**, since Baseline B's definition is specifically
  "how far can last-known-state alone get us, with zero use of the current observation, even when it
  exists"). This is subtly different from `BaselineDPredictor`'s masked-branch logic — check this
  distinction carefully; a naive copy-paste of Baseline D's logic minus the observed-branch is *not quite*
  the same as always ignoring `TWS_t`, if the fit-time history-building step differs.
- **`SeasonalClimatologyPredictor`** (**Baseline C**) — `fit`: per-`(location_id, calendar_month)` mean of
  `target` from the training fold, plus a global-mean fallback for unseen combinations. `predict`: look up
  by `(location_id, month of target_time)` — be careful which month (origin vs. target) the climatology
  should key off; Phase 1's original measurement used per-location-per-calendar-month mean of the *target*
  itself, re-derive this precisely rather than guessing. Phase 1 measured this at 0.817 in-sample
  (`notebook 01`) — weaker than intuition suggests, meaning most error in this problem is within-location
  month-to-month deviation, not baseline-level miscalibration.
- **`HybridPersistencePredictor`** (**Baseline D**) — promote `BaselineDPredictor` from
  `notebooks/03_validation_harness.ipynb`'s build script verbatim (or near-verbatim) into this file: if
  `TWS_t` is observed for a row, use it; else use last-known-state; else (never-observed location) fall
  back to the fitted global mean. This is the realistic "no ML at all" floor — Phase 1 measured 0.6573 on
  the real test structure, and Phase 2's proof run already confirmed this logic reproduces that number
  through the harness (with the A-013 caveat on Tier 3). **After building this class, go back and simplify
  `notebooks/03_validation_harness.ipynb`'s next execution (if it's ever re-run) to import from here
  instead of redefining the class inline** — not required for Phase 3's own Definition of Done, but avoids
  the two copies silently drifting apart.
- **`RidgeBaselinePredictor`** — a thin wrapper around `sklearn.linear_model.Ridge` fit on the raw available
  numeric columns (`SPEI_01_t, SPEI_03_t, SPEI_06_t, SPEI_12_t, SOIL_MOISTURE_t, month_sin, month_cos`, plus
  `TWS_t` if present). Since masked rows have `TWS_t = NaN`, decide explicitly: either fit two internal
  Ridge models (one with `TWS_t` as a feature for observed rows, one without for masked rows) or drop
  `TWS_t` from the feature set entirely so one model serves both regimes — document which choice was made
  and why, since it directly affects how to interpret Tier 2's regime-split numbers for this baseline.

Every class's `predict()` must return a plain `np.ndarray`, must never mutate its input `df`, and must
handle being called on a frame that includes both observed and masked rows without raising — the harness
calls each candidate exactly this way across all three tiers.

### 3.2 — `tests/test_baselines.py`

For each of the six classes: `fit` then `predict` returns an array of the right length with no NaNs (a
predictor that returns NaN silently breaks every downstream RMSE calculation — test this explicitly);
determinism (`GlobalMeanPredictor`/`RidgeBaselinePredictor` in particular, given `set_seed()`); the
never-observed-location fallback path for `LastKnownStatePredictor`/`SeasonalClimatologyPredictor` (build a
tiny synthetic frame with a location present in `predict`'s input but absent from `fit`'s training frame,
confirm it falls back to the global mean rather than raising or returning NaN); `OraclePersistencePredictor`
returns exactly `TWS_t` on observed rows and the documented fallback on masked rows, verified against a
hand-built fixture with both regimes present; each class satisfies the `Predictor` protocol structurally
(can be passed to `run_tier1`/`run_tier2`/`run_tier3` against the golden fixture and completes without
error — this is the real integration check, not just a unit test in isolation).

### 3.3 — `notebooks/04_baselines.ipynb`

The Phase 3 proof-run notebook, same pattern as `03_validation_harness.ipynb` (executed top-to-bottom
against the real, full-scale `Train.csv`, all outputs saved before committing). Contents:

1. Load `Train.csv`, compute the real per-location ACF(1) lookup (reuse notebook 03's §3 logic, or better,
   promote it to a small tested helper if it's going to be needed again in Phase 4 anyway — likely worth
   doing now rather than a third copy-paste).
2. Instantiate all six baseline predictors from `models/baselines.py`.
3. For each, call `harness.evaluate_candidate(model, train_df, candidate_id, acf_lookup=acf_lookup,
   include_tier3=True, n_anchors=3)` — this alone gets Tier 1, Tier 2, and (with the A-013 handling decided
   in step 3.0) Tier 3, plus full decomposition tables and degradation slope, in one call per candidate.
4. Print/render every `CandidateReport`'s decomposition tables — **the actual deliverable is these tables,
   not a leaderboard of six RMSE numbers.** Compare against Phase 1's preview numbers (A=0.5247, B=0.7145,
   C=0.8170, D=0.6573) and note any discrepancy with a reason (different validation mechanism — harness CV
   folds vs. Phase 1's 8-window replay — some numeric difference between Tier 1/2's *harness* numbers and
   Phase 1's *replay* numbers is expected and fine; only Tier 3, run through the sequential-state pattern,
   should closely match Phase 1's original replay numbers).
5. Run `promote()` for every baseline against the ladder alone, and head-to-head against Baseline D as the
   reference baseline (mirroring Phase 2's proof run) — confirm the four baselines rank in the expected
   order (D should be the strongest "no ML" floor; A should look artificially strong on Tier 1 alone since
   it's an oracle there, and comparatively weak once Tier 2/3 masked rows are scored under its documented
   fallback).
6. `log_candidate()` every one of the six to the real `experiment_log.csv`/MLflow backend — this is what
   makes these numbers "the official record" per `ARCHITECTURE.md` §13's own phrasing.
7. A closing synthesis section (mirroring notebook 03's section 17): state explicitly, in prose, the exact
   number every Phase 5+ model must beat per regime — this is Phase 3's actual deliverable, spelled out, not
   left implicit in a table.

**Practical execution note, learned the hard way in Phase 2**: running a full-scale notebook against the
real 2.15M-row `Train.csv` inside a sandboxed tool-call environment with a ~45-second wall-clock cap per
call is genuinely difficult — six candidates × three tiers × multiple CV folds each will not fit in single
calls. If executing in a similarly constrained environment, budget for either a chunked/checkpointed
execution approach (Phase 2 improvised one; ask the user or search prior session history if needed — it was
explicitly *not* committed to the repo, since it's scratch tooling, not part of the deliverable) or simply
have the user run the notebook locally in their own Jupyter/VS Code environment (which is what actually
happened for notebook 03 in practice) and hand back the saved `.ipynb` for interpretation. Either way,
**verify the notebook file actually has cell outputs saved before treating it as done** — this tripped up
Phase 2's closeout once already (a notebook execution's side-effect files, like the MLflow run, can succeed
while the `.ipynb` itself fails to save its outputs back to disk if the editor's own save step is missed).

### 3.4 — Documentation closure (mirror Phase 2's step 2.12 pattern exactly)

- `docs/PROJECT_PLAN.md` — check off Phase 3's items, add a STATUS: MET paragraph with the real numbers,
  same style as the existing Phase 1/Phase 2 STATUS notes.
- `docs/ARCHITECTURE.md` §20 — extend the "current status" paragraph to record Phase 3's closure and
  headline numbers, same pattern as the existing Phase 1/Phase 2 paragraphs.
- `docs/ASSUMPTIONS.md` — add a new entry only if something genuinely surprised the Phase 1 expectations
  (e.g. if the harness-computed Tier 1/2 numbers for A/B/C/D differ meaningfully and unexpectedly from
  Phase 1's preview numbers, or if the Ridge baseline does notably better or worse than intuition suggests).
  Don't force an entry if nothing surprising happened — Phase 2's own execution plan explicitly says not to
  manufacture assumptions register entries where there's nothing to record.
- `reports/experiments/experiment_log.csv` — confirm all six new EXP rows have real Tier 1/2/3 numbers (not
  `N/A`), continuing the EXP-NNN sequence from EXP-009 (i.e., new rows start at EXP-010).
- One-file-per-commit throughout: `baselines.py`, its test file, the notebook, its figures (one bundled
  commit), the experiment log, then each doc file separately — matching Phase 2's exact commit pattern (see
  the repo's own git log for the literal precedent, commits `783b202` through `6346346`).

### Definition of Done (verbatim from `PROJECT_PLAN.md`)

> All four baselines decomposed by regime; we know the exact numbers every subsequent model must beat, per
> regime, not just on average.

Concretely: `models/baselines.py` exists, tested, importable; `notebooks/04_baselines.ipynb` is executed
end-to-end against the real data with all six candidates scored through the harness and logged for real;
the decomposition tables (not just aggregate RMSE) are the artifact that gets referenced by Phase 4/5 going
forward; documentation closure is complete, matching Phase 2's own closure pattern.

---

## 5. What comes after Phase 3 (context only, not this phase's job)

Project Phase 4 (state reconstruction & feature engineering) is next, and depends directly on Phase 3's
baselines existing as the thing it must beat. Phase 4 builds the `StateSnapshot` schema (co-located in
`state/reconstruction.py` alongside `ForecastOrigin`), shrinkage-regularized historical location
signatures, and spatial-history features from neighbors' *historical* trajectories (never concurrent-month
values — the 0.981-correlation-but-unusable-during-blackouts finding from Phase 1 rules that out). Full
detail: `ARCHITECTURE.md` §13, `PROJECT_PLAN.md` Phase 4 section. Do not start Phase 4 work inside a Phase
3 session — get sign-off on Phase 3's results first, per this project's established "one phase at a time,
stop for review" working style (`PROJECT_PLAN.md`, "How we'll work").

---

## 6. Quick-reference cheat sheet

| Thing | Value / Location |
|---|---|
| Repo root | `D:\PROJECTS\tws-forecast` |
| Conda env | `tws-forecast`, Python 3.11, `D:\CONDA\conda_envs\tws-forecast` |
| Random seed | 42 (`utils/seeds.py::RANDOM_SEED`) |
| Baseline A / B / C / D | 0.5247 / 0.7145 / 0.8170 / 0.6573 (Phase 1 preview numbers; Phase 3 recomputes officially) |
| Promotion ladder | naive_floor <0.6573 · oracle_ceiling <0.572 · beat_mohar <0.559 · serious_contender <0.53 · exceptional <0.50 |
| Current #1 public leaderboard (MOHAR, at time of `COMPETITIVE_ANALYSIS.md` writing) | 0.55958798 — re-check current standings before treating as current |
| Verified clean training span | 2004-01 through 2010-12 |
| Real test FULL offsets | [0, 4, 9, 15, 34, 38] |
| Real test BLACKOUT offsets | [5, 6, 10, 11, 12, 16, 17, 18, 19, 20, 21, 39] |
| Blackout staleness distribution | k=2(×4), k=3(×3), k=4(×2), k=5/6/7(×1 each) |
| Test suite | `pytest -q` from repo root, 211 tests (pre-Phase-3) |
| Latest closed phase | Project Phase 2 (Validation harness) — formally MET |
| Next phase | Project Phase 3 (State-aware baselines) — this document |
| Experiment log | `reports/experiments/experiment_log.csv`, next ID is EXP-010 |
| MLflow | `mlflow.db` + `mlruns/` at repo root, experiment `tws-forecast-validation` |

---

## 7. Open items worth knowing about but not blocking Phase 3

- Local `git push` may need to be run by the user directly — push access isn't guaranteed from every
  execution environment this project has been worked in.
- Two open decisions still owed from the user (`PROJECT_PLAN.md`, "Open decisions I need from you"):
  Streamlit vs. Gradio for the eventual interactive demo (Phase 13, not urgent), and the Trustworthiness
  Evaluation rubric document (needed before Phase 9, not urgent for Phase 3).
- `docs/OPEN_QUESTIONS.md` has five standing open questions (e.g. how much of the masked regime is
  predictable from environmental forcing alone, independent of TWS history) — worth a glance if Phase 3's
  Ridge baseline result raises any of them concretely.
