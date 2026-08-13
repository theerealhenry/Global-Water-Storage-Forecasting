# Project Phase 2 — Validation Harness: Execution Plan

**Status:** ready to implement. Written after Phase 1's formal closure (ADR-0004) and a module-map
reconciliation pass against `ARCHITECTURE.md` (ADR-0005), which this plan is built against directly.
**Objective (unchanged from `PROJECT_PLAN.md`):** a validation scheme that won't lie to us, built on what
Phase 1 measured, not on assumptions.

**How to use this document:** it's the single authoritative build order for Phase 2. `PROJECT_PLAN.md`'s
Phase 2 section is trimmed to point here rather than duplicate this content. Each step below produces one
file (or one tight file+test pair), gets its own commit per the project's one-file-per-commit workflow, and
has its own concrete deliverable — this phase is done when every deliverable below exists and the
phase-level Definition of Done at the bottom passes.

---

## 0. Design constants (unchanged from ADR-0004 — restated here for one-stop reference)

| Constant | Value | Source |
|---|---|---|
| Real blackout staleness-to-target distribution | k=2 (×4), k=3 (×3), k=4 (×2), k=5 (×1), k=6 (×1), k=7 (×1) | Experiment 4 |
| Real test FULL-month offsets | [0, 4, 9, 15, 34, 38] months from 2015-09 | Experiment 4 |
| Real test BLACKOUT-month offsets | [5, 6, 10, 11, 12, 16, 17, 18, 19, 20, 21, 39] | Experiment 4 |
| Real absent-month structure | main gap 2017-07→2018-06 (12mo); 4 smaller scattered gaps | Experiment 7 |
| Baseline A (oracle persistence) | 0.5247 | Experiment 4 |
| Baseline B (last-known, blackout only) | 0.7145 | Experiment 4 |
| Baseline C (seasonal climatology) | 0.8170 | notebook 01 |
| Baseline D (Hybrid, realistic naive floor) | 0.6573 | Experiment 4 |
| Calendar-month coverage gap | October = 0% of test rows; 7 months ~11% each; 4 months ~5.5% each | Experiment 6, A-011 |
| Verified gap-free training span | 2004-01 through 2010-12 (84 months) | Experiment 3 |
| Missing training months (22 total) | 5 pre-2011 unexplained, 17 battery-management pattern | Experiment 7 |
| Staleness × ACF relationship | nonlinear (ρᵏ form) | Experiment 5, A-010 |
| Promotion ladder | <0.6573 clears naive floor · <0.572 matches oracle ceiling · <0.559 beats MOHAR's public 0.55958798 · <0.53 serious contender · <0.50 exceptional | `COMPETITIVE_ANALYSIS.md` §6 |

These live in exactly one file (`phase1_constants.py`, step 2.1), each value with a comment pointing to its
exact notebook section, so nobody re-derives them and nothing can silently desync from Phase 1.

## 1. Governing invariants (apply to every step below, not just one)

- **Time-respecting only.** No random K-fold anywhere in this phase — confirmed non-stationary trend
  (notebook 01) and the 2015 anomaly (A-004) both make random splits actively misleading.
- **`ForecastOrigin` discipline.** Every fold, every masked example, every decomposition row is keyed by
  `(origin_time, target_time, horizon, information_cutoff, location_id, regime)` — never by raw row index.
- **Tier 3 is diagnostic, never a promotion criterion on its own** (`ARCHITECTURE.md` §11) — enforced in
  code (step 2.10), not just documented.
- **No leakage feature.** No feature derived from test-row position/order is ever permitted — enforced in
  code (step 2.9), not just documented.
- **One file, one commit, one test.** Each numbered step below is sized to be exactly one commit.

---

## 2.1 — Prerequisites: seeds, config loader, Phase 1 constants

**Files:**
- `src/tws_forecast/utils/seeds.py` — `RANDOM_SEED = 42` module constant; `set_seed(seed: int = RANDOM_SEED) -> None` seeding `random`, `numpy`, and (when relevant) LightGBM/XGBoost's own seed args. Every stochastic operation from here on takes an explicit seed, defaulting to this constant, never an unseeded call.
- `src/tws_forecast/utils/config.py` — minimal `pydantic` `BaseSettings`-style loader reading `configs/base.yaml` (paths, `RANDOM_SEED`, data dir). Deliberately small — full config-driven modeling config arrives in Phase 4/5; Phase 2 only needs enough to load scenario YAMLs (step 2.6) and resolve `data/raw/`.
- `src/tws_forecast/validation/phase1_constants.py` — the table in §0 above, as typed constants (`BLACKOUT_K_DISTRIBUTION: list[int]`, `TEST_FULL_OFFSETS: list[int]`, `TEST_BLACKOUT_OFFSETS: list[int]`, `BASELINE_A/B/C/D: float`, `PROMOTION_THRESHOLDS: dict[str, float]`, etc.), each with a docstring comment citing the exact notebook section.
- `configs/base.yaml` — `random_seed: 42`, `data_dir: data/raw`, `train_period: {start: 2002-05, end: 2015-08}`, `test_period: {start: 2015-09, end: 2018-12}`.

**Tests:** `tests/test_seeds.py` (two calls to `set_seed` produce identical `np.random` output); `tests/test_phase1_constants.py` (values match the notebook — e.g. `sum of BLACKOUT_K_DISTRIBUTION == 12`, `BASELINE_A < BASELINE_B < BASELINE_D < BASELINE_C`, an ordering check that would catch a transcription error).

**Deliverable:** running `set_seed()` then any numpy-random operation twice gives identical results; `phase1_constants.py` importable from anywhere in `src/` with every Phase 1 number in one place, test-verified against the source notebook.

## 2.2 — `ForecastOrigin` schema

**File:** `src/tws_forecast/state/reconstruction.py` (created now with only `ForecastOrigin`; `StateSnapshot` is added to this same file in Phase 4 per `ARCHITECTURE.md` §6's co-location — noted in the module docstring so nobody "fixes" this file's apparent incompleteness prematurely).

**Contents:** a frozen dataclass (or pydantic model — pydantic is already pinned, prefer it for free validation) with fields `origin_time: pd.Timestamp, target_time: pd.Timestamp, horizon: int, information_cutoff: pd.Timestamp, location_id: str, regime: Literal["observed", "masked"]`. A `from_row(row, horizon=1)` constructor building one from a `Train.csv`/`Test.csv` row (`origin_time = time`, `target_time = time + 1 month`, `information_cutoff = time`, `regime` derived from whether `TWS_t`/`TWS_t_masked` indicates a real observation).

**Tests:** `tests/test_forecast_origin.py` — `target_time == origin_time + 1 month` always; `information_cutoff <= origin_time` always (the literal no-future-leakage assertion at the schema level); round-trip from a few known golden-fixture rows produces the expected fields.

**Deliverable:** every later step (splitters, masking simulator, tiers, decomposition) consumes `ForecastOrigin` objects/frames, never raw row indices — this is what makes the leakage tests in step 2.9 mechanically checkable rather than a matter of code review trust.

## 2.3 — Expanding-window splitter

**File:** `src/tws_forecast/validation/splitters.py`

**Contents:** `expanding_window_splits(df, n_folds, min_train_months=84, anchor_to_2004=True, seed=RANDOM_SEED) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]` (train fold, validation fold, both as `ForecastOrigin`-indexed frames). Fold boundaries: earliest fold's training portion always includes a full pass through the verified clean 2004-2010 span (§0); folds hold out progressively later years; final fold's validation window sits adjacent to 2015 specifically, per `PROJECT_PLAN.md` §2.1's original design intent — the harness must confront the 2015 anomaly (A-004), not fold around it.

**Tests:** `tests/test_splitters.py` — the literal leakage check (`assert no validation-fold row's origin_time ≤ any same-fold training row's origin_time` — actually the reverse: validation origin_time must be > all training origin_time in the same fold); determinism given a fixed seed; the final fold's validation window includes at least one 2015 month; the earliest fold's training portion covers all of 2004-2010.

**Deliverable:** a fold generator that is provably non-leaking (test-enforced) and provably deterministic, producing folds that don't average away the 2015 anomaly by construction.

## 2.4 — `MaskingScenario` + streak-aware masking simulator

**File:** `src/tws_forecast/validation/masking_simulator.py`

**Contents:** `MaskingScenario` dataclass, fields exactly per `ARCHITECTURE.md` §8 — `blackout_start, blackout_end, affected_locations, exception_rate, streak_length, transition_pattern, source_rationale`. `apply_masking(df, scenario: MaskingScenario, seed=RANDOM_SEED) -> pd.DataFrame` nulls `TWS_t` (and any column derived from it — none yet exist pre-Phase-4, but the function signature reserves a `derived_columns: list[str] = []` parameter so Phase 4 doesn't have to touch this function) for the run of consecutive months the scenario specifies, respecting per-month row-count variability (Experiments 1 & 3's finding that no month has the full 15,715-row grid, in either file) by construction — the function only ever operates on rows that exist.

The simulator itself is **generic** — it doesn't know about "curve mode" or "replay mode"; those become two named `MaskingScenario` instances registered in step 2.5, so `masking_simulator.py` stays a clean, reusable primitive rather than accumulating scenario-specific branches.

**Tests:** `tests/test_masking_simulator.py` (synthetic fixture) — masked rows' `TWS_t` is null but all other columns remain populated (matches the real `Test.csv` contract in `DATA_DICTIONARY.md`); the simulator never masks a row that didn't exist; applying a scenario twice with the same seed produces identical output.

**Deliverable:** a masking primitive that can reproduce any contiguous-block blackout pattern from a declarative config, verified leak-free (masked value is genuinely gone, never recoverable from another column) and grid-irregularity-aware.

## 2.5 — Scenario registry

**File:** `src/tws_forecast/validation/scenarios.py`, plus `configs/validation/*.yaml` (one file per scenario, per `ARCHITECTURE.md` §11's explicit requirement that experiments reference scenarios by identifier rather than re-describing split/masking logic inline).

**Concrete scenarios to define now** (minimum set — more may be added later without touching this file's structure):
- `configs/validation/expanding_window.yaml` — Tier 1, no masking, wraps step 2.3's splitter with default args.
- `configs/validation/blackout_curve.yaml` — Tier 2, `MaskingScenario` instances resampling blackout length with replacement from `BLACKOUT_K_DISTRIBUTION` (§0), reproducing Experiment 3's multi-window design.
- `configs/validation/test_regime_replay.yaml` — Tier 3, a `MaskingScenario` anchored to the real `TEST_FULL_OFFSETS`/`TEST_BLACKOUT_OFFSETS` calendar identities (§0), reproducing Experiment 4's Method B exactly, including A-011's October-gap/2×-1× recurrence pattern.
- `configs/validation/2015_like.yaml` — a scenario isolating validation windows adjacent to the 2015 anomaly specifically, named per `ARCHITECTURE.md` §11's example list ("a scenario approximating 2015-like conditions").

**Contents of `scenarios.py`:** `load_scenario(name: str) -> ScenarioConfig` (pydantic model parsing the YAML), `list_scenarios() -> list[str]`, `SCENARIO_REGISTRY` mapping name → config path.

**Tests:** `tests/test_scenarios.py` — every YAML in `configs/validation/` parses into a valid `ScenarioConfig`; `test_regime_replay.yaml`'s offsets match `phase1_constants.py` exactly (a drift-detection test — if someone edits one but not the other, this fails).

**Deliverable:** every validation scenario used anywhere in the project from this point on is a named, versioned config file, not inline logic — directly satisfying `ARCHITECTURE.md` §11 and giving every future experiment a one-line way to say "run under `test_regime_replay`."

## 2.6 — Three validation tiers

**File:** `src/tws_forecast/validation/tiers.py`

**Contents:** `run_tier1(model, df, scenario="expanding_window") -> TierResult`, `run_tier2(model, df, scenario="blackout_curve") -> TierResult`, `run_tier3(model, df, scenario="test_regime_replay") -> TierResult`, all sharing a `TierResult` container (predictions frame keyed by `ForecastOrigin`, overall RMSE, fold-level RMSEs + variance). Each composes `splitters.py` (fold generation) with `scenarios.py`/`masking_simulator.py` (masking application) — no tier reimplements split or masking logic itself.

**Tests:** `tests/test_tiers.py`, run against a trivial mean-predictor stand-in — confirms each tier executes end-to-end on the golden fixture and returns a well-formed `TierResult`; confirms Tier 1's fold count and Tier 3's calendar anchoring match expectations.

**Deliverable:** three callable, independently-runnable validation tiers, each answering the distinct question `ARCHITECTURE.md` §11 assigns it, sharing no duplicated split/masking code.

## 2.7 — Error decomposition table + degradation slope

**File:** `src/tws_forecast/validation/decomposition.py`

**Contents:** `decompose(tier_result: TierResult, acf_lookup: pd.Series | None = None) -> pd.DataFrame` producing rows for: overall / masked / unmasked / **by the real k=2,3,4,5,6,7 staleness buckets** (§0 — never an invented 1-2mo/3-4mo/5+mo scheme) / **cross-cut by ACF quartile within each staleness bucket** (per A-010) / by hemisphere / on extreme-TWS and rapid-change slices. `degradation_slope(decomp_df) -> pd.DataFrame` computing ΔRMSE/Δk per ACF quartile, returned alongside Experiment 5's already-validated AR(1) theoretical curve (`sigma*sqrt(2*(1-rho**k))`, reusing per-quartile ρ/σ constants carried from `phase1_constants.py`) as a reference column, so any model's degradation is numerically comparable to the mechanistic baseline from day one, not eyeballed against a figure.

**Tests:** `tests/test_decomposition.py` — decomposition table has exactly the expected row set (no silently-invented bucket scheme); degradation slope's reference-curve column matches Experiment 5's numbers within tolerance on the golden fixture; ACF-quartile cross-cut columns sum/align correctly (each staleness bucket's rows cover the full population, no rows silently dropped).

**Deliverable:** every model run from Phase 3 onward can call one function and get the full, standard decomposition table plus a degradation-slope comparison to the AR(1) reference — this is the artifact the promotion rule (step 2.8) reads from, and the artifact `COMPETITIVE_ANALYSIS.md`/`docs/BIAS_AND_EXPLAINABILITY.md` will eventually quote directly.

## 2.8 — Leakage firewall, as executable checks

**File:** `src/tws_forecast/validation/leakage_tests.py` (check *logic*, importable), thin wrappers in `tests/test_leakage_firewall.py` and `tests/test_no_leakage_features.py` (the actual pytest entrypoints).

**Contents — four checks, each named directly in `ARCHITECTURE.md` §7/§11:**
1. `future_row_shuffle_test(pipeline, df)` — shuffle all rows with `origin_time` ≥ some cutoff, confirm predictions for rows before the cutoff are byte-identical.
2. `historical_only_check(signature_fn, df)` — confirm any location-signature/climatology function's output at time `t` is unchanged when rows at/after `t` are removed from its input.
3. `rolling_window_cutoff_check(feature_fn, df)` — confirm a rolling/lag feature computed at origin `t` never reflects a value with `time ≥ t`.
4. `masking_simulator_no_leak_check(scenario, df)` — confirm `apply_masking`'s output never allows the masked `TWS_t` value to be recovered from any other column in the same row.

**Separately, `tests/test_no_leakage_features.py`** scans `src/tws_forecast/features/` module output columns (once Phase 4 exists — this test is written now, and stays green vacuously until Phase 4 adds real feature modules) for disallowed name patterns (`test_row_index`, `relative_test_position`, or anything matching a `row_order`/`file_position`-style pattern) — a standing, mechanical guard, not a one-time promise.

**Deliverable:** the leakage firewall from `ARCHITECTURE.md` §7 is a green pytest suite, not a paragraph of prose — every future modeling phase inherits this check for free.

## 2.9 — Harness orchestrator + promotion rule

**File:** `src/tws_forecast/validation/harness.py` (an addition to `ARCHITECTURE.md`'s explicit file list, recorded in ADR-0005 — the composition point, not a duplicate of any single-tier module).

**Contents:** `evaluate_candidate(model, df, candidate_id) -> CandidateReport` running Tier 1 + Tier 2 (+ Tier 3 diagnostically), producing the full decomposition table and degradation slope, and `promote(report: CandidateReport) -> PromotionDecision` implementing the target ladder as code constants pulled from `phase1_constants.py` (§0's promotion row). **Hard rule enforced here, not just documented:** `promote()` raises `ValueError` if called with a `CandidateReport` missing Tier 1 or Tier 2 results — a Tier-3-only score can never satisfy promotion, directly implementing `ARCHITECTURE.md` §11's stated boundary. A model is only promoted when it clears the relevant threshold on the full decomposition table, not the headline number alone — `promote()` checks the k=5-7 buckets specifically don't regress even if the aggregate improves.

**Tests:** `tests/test_harness.py` — `promote()` raises on a Tier-3-only report (the integrity safeguard, test-enforced); a report that improves overall RMSE but regresses the k=6/k=7 bucket is correctly *not* promoted; a report clearing 0.6573 with no regime regression is promoted at the correct ladder rung.

**Deliverable:** one function is the only legitimate way any future model gets called a "champion" — the promotion rule from `COMPETITIVE_ANALYSIS.md` §6 is executable, not aspirational.

## 2.10 — Experiment log migration + MLflow kickoff

**Files:** extend `reports/experiments/experiment_log.csv` usage (`cv_tier1_rmse`, `cv_tier2_rmse`, `cv_tier3_rmse` columns already exist from Phase 0 — Phase 2 is the first phase to actually populate them, via `harness.py`'s output, rather than leaving them `N/A`); stand up `mlflow.db` (SQLite backend) per `ARCHITECTURE.md` §6/Phase 0's "can catch up" deferral — now is when it catches up, since the harness is the stabilization point Phase 0 named as the trigger.

**Deliverable:** every `harness.py` run logs one row to the flat CSV (kept, for quick grep-able review) and one MLflow run (`mlruns/`) with the full decomposition table as an artifact — the lineage chain `ARCHITECTURE.md` §5/§18 describes starts being real from here on, not just specified.

## 2.11 — Validation notebook: proof run

**File:** `notebooks/03_validation_harness.ipynb`

**Contents:** loads Train.csv, runs `harness.evaluate_candidate()` against two trivial stand-ins — (a) Baseline D's own logic (if-observed-use-it-else-last-known) wrapped as a model, and (b) a bare LightGBM on raw columns (no state features yet — those don't exist until Phase 4) — through all three tiers; renders the decomposition table; plots the degradation slope against the AR(1) reference curve from Experiment 5; states explicitly whether Tier 3's Baseline-D-logic score reproduces 0.6573 within tolerance.

**Deliverable — this is the phase-defining sanity check.** If Tier 3's naive-model score doesn't land near 0.6573, the harness has a bug and is not faithfully reproducing Phase 1's measured reality (this exact validation criterion is already written into ADR-0004 and `PROJECT_PLAN.md`'s Phase 2 Definition of Done — this notebook is where it gets checked, not just promised).

## 2.12 — Documentation closure pass

- [x] `PROJECT_PLAN.md` Phase 2 section trimmed to a short summary + pointer to this document; checkboxes ticked as each step above completes. STATUS: MET note added summarizing step 2.11's real results.
- [x] `ARCHITECTURE.md` §20 status paragraph updated once Phase 2 code exists (previously stated "no modeling code has been written yet") — now records Phase 2's formal closure and step 2.11's proof-run findings.
- [x] `docs/ASSUMPTIONS.md` — a real surprise WAS encountered while building (contrary to this checklist's original expectation that Phase 2 wouldn't generate one): Tier 3's replay-anchor selection had a genuine bug, and separately its row-wise scoring design under-serves stateful baselines. Logged as new entry A-013, not silently patched away.
- [x] `reports/experiments/experiment_log.csv` — confirmed: Tier 1/2/3 columns populated with real RMSEs (not `N/A`) for both step-2.11 proof runs, EXP-008 (Baseline D logic: 0.6380/0.6381/0.8270) and EXP-009 (bare LightGBM: 0.5798/0.5801/0.7669). Several duplicate rows from retries against a transiently-corrupted local `mlflow.db` were cleaned up first.
- [x] ADR-0004 and ADR-0005 follow-up-action checkboxes both closed out.

---

## Phase-level Definition of Done (restated, now step-mapped)

Running the harness (`harness.py`, step 2.9) against a trivial model produces all three tiers (2.6), the
full decomposition table including the ACF-quartile × staleness-bucket cross-cut (2.7), and the degradation
slope with the AR(1) reference overlay (2.7) — all demonstrated in the step-2.11 notebook. Tier 3's
naive-model score reproduces Baseline D's 0.6573 within a small tolerance (2.11). The leakage tests (2.3,
2.8) pass. Every scenario used is a named config file (2.5), not inline logic. We both agree the scheme
can't leak the future and honestly reflects the real blackout structure — at which point Phase 2 is closed
the same way Phase 1 was: an ADR recording closure and binding Phase 3's baseline work to whatever Phase 2
actually measured.

## Suggested build order (dependency-respecting)

2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8 → 2.9 → 2.10 → 2.11 → 2.12. Each arrow is a hard dependency
(e.g. 2.6 cannot be written meaningfully before 2.3/2.4/2.5 exist) — this is not an arbitrary ordering
preference, it's the actual build dependency graph, so it should be followed in order rather than
parallelized.
