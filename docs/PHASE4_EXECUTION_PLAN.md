# Project Phase 4 — State Reconstruction & Feature Engineering: Execution Plan

**Status:** ready to implement. Written after Project Phase 3's formal closure (2026-08-14, real numbers in
`PROJECT_PLAN.md`/`ARCHITECTURE.md` §20/`docs/ASSUMPTIONS.md` A-014), the same way Phase 3's own build
guidance (`docs/PHASE3_HANDOFF.md`) was written against Phase 2's proven harness. **Objective (unchanged
from `PROJECT_PLAN.md`):** build the observation-state reconstruction layer as one coherent, explicit
pipeline stage — the `StateSnapshot` schema `ARCHITECTURE.md` §4 already specifies — rather than scattered,
ad hoc features computed independently by different modules.

**How to use this document:** the single authoritative build order for Phase 4, the same role
`PHASE2_EXECUTION_PLAN.md` played for Phase 2. `PROJECT_PLAN.md`'s Phase 4 section is trimmed to a summary
+ pointer here, matching that precedent. Each step below produces one file (or one tight file+test pair),
gets its own commit per the project's one-file-per-commit workflow, and has its own concrete deliverable.
Read this in full before writing any code — several steps below resolve real ambiguities between
`ARCHITECTURE.md`'s `StateSnapshot` field list and `PROJECT_PLAN.md`'s Phase 4 bullet list that are worth
knowing about up front (step 4.1).

---

## 0. What Phase 4 inherits — every number, taxonomy, and finding it must build on

Nothing below is re-derived or re-argued; it's carried forward exactly, the same discipline
`phase1_constants.py` established for Phase 1's numbers.

| Input | What it says | Why Phase 4 needs it | Source |
|---|---|---|---|
| Central hypothesis | This is a state-reconstruction problem, not `TWS_t → TWS_t+1` regression | The entire reason this phase exists | `PROJECT_PLAN.md` "Central hypothesis"; `ARCHITECTURE.md` §3 |
| `StateSnapshot` field list | `last_known_tws, last_known_time, months_since_observation, previous_known_tws, historical_delta, local_trend, seasonal_position, acf_1_3_6_12, observation_density, blackout_streak_length, location_signature, state_status` (`OBSERVED`/`RECONSTRUCTED`/`PARTIALLY_RECONSTRUCTED`) | The canonical schema this phase implements — every downstream feature/model/diagnostic consumes this, never a competing notion of "last known value" | `ARCHITECTURE.md` §4 |
| Signature-indexing invariant | A signature used for a forecast at origin `t` must be `signature(location, t)`, built only from history strictly before `t` — fold-level OOF is **not** sufficient, origin-time indexing is required | Governs `state/signatures.py`'s entire design (step 4.2) | `ARCHITECTURE.md` §4 |
| Shrinkage formula | `θ̂_location = w·θ_location + (1−w)·θ_global`, `w` increasing with location-level evidence (empirical-Bayes style, e.g. `w = n/(n+k)`) | The specific fix for the failure Phase 3 just measured directly (A-014) | `PROJECT_PLAN.md` Phase 4; `ARCHITECTURE.md` §10/§17 |
| **A-014 — climatology overfits without shrinkage** | Naive per-`(location, month)` climatology (Baseline C) scored 1.0796 out-of-fold, *worse* than a plain global mean (0.8740) — the in-sample 0.817 preview was optimistic | This phase's shrinkage-regularized signature is the direct, evidence-backed fix; step 4.9's proof notebook owes A-014 a direct confirmation, not just a design that sounds right | `docs/ASSUMPTIONS.md` A-014 |
| **A-008 — ACF quartile is the dominant staleness-degradation driver** | Per-location ACF(1) quartile produces the widest RMSE spread at k=9 (0.403) of any tested cut — bigger than latitude, season, or drought regime combined | Directly prioritizes `acf_1_3_6_12` as a `StateSnapshot` field over region/season features | `docs/ASSUMPTIONS.md` A-008 |
| **A-010 — staleness × ACF is nonlinear** | The true `k`-vs-degradation relationship follows AR(1)'s `ρᵏ` form, not a linear product term; volatility (σ) is a largely independent signal from ACF (r=-0.141) and adds real explanatory power on its own | `StateSnapshot` keeps `acf_1_3_6_12` and `location_signature` (which carries σ) as distinct fields, never collapsed into one composite — and step 4.9 should test the AR(1)-motivated composite `sigma*sqrt(2*(1-rho**k))` as one explicit engineered feature, not just hope the GBM learns the interaction | `docs/ASSUMPTIONS.md` A-010 |
| **A-011 — test entirely omits October** | 0/18 test months are October; 2× row-share imbalance between recurring and non-recurring calendar months | Any seasonal/calendar feature (step 4.5) must be validated on under/zero-represented months specifically, not just aggregate CV | `docs/ASSUMPTIONS.md` A-011 |
| **A-003 — spatial stability (still Active, not yet directly tested)** | Same-month spatial correlation is 0.981, but *temporal* stability of that relationship hasn't been directly tested — only indirect covariate-shift evidence exists so far | Phase 4 **builds** spatial-history features; the direct ablation that would resolve A-003 is explicitly scoped to Project Phase 6 (`docs/OPEN_QUESTIONS.md`), not this phase — don't over-claim what Phase 4's own feature-importance pass proves about A-003 | `docs/ASSUMPTIONS.md` A-003; `docs/OPEN_QUESTIONS.md` |
| S1-S4 spatial-feature taxonomy | S1 concurrent (same-month, ruled out for blackout regimes — 0.981-correlation-but-unusable finding), S2 historical (neighbor trajectories/trends/anomalies — Phase 4's actual mechanism), S3 signature (static basin/location characteristics), S4 residual (post-hoc spatial correction, not this phase's job) | Every spatial feature this phase adds must declare which of these four it is | `ARCHITECTURE.md` §9 |
| Four distinct temporal quantities | Calendar lag (`TWS_t-k`, itself often missing) ≠ last-observed lag (`last_known_tws`) ≠ observation age (`months_since_observation`) ≠ observation trajectory (last/previous/second-previous known, velocity/acceleration) — genuinely different things, not interchangeable | The core design constraint for `StateSnapshot` (step 4.1) | `PROJECT_PLAN.md` Phase 4 |
| ACF-quartile AR(1) params | `Q1_low_ACF ρ=0.510 σ=0.830`, `Q2 ρ=0.734 σ=0.822`, `Q3 ρ=0.826 σ=0.830`, `Q4_high_ACF ρ=0.929 σ=0.760` | Reused for the AR(1)-motivated composite feature (A-010) and for sanity-checking the new real per-fold ACF computation against Experiment 5's numbers, the same cross-check notebook 03/04 both already did | `phase1_constants.ACF_QUARTILE_AR1_PARAMS` |
| Baseline D floor | Tier 2 RMSE 0.6381 (harness), 0.6573 (Phase 1 replay) — every k=5/6/7 bucket value in `reports/experiments/experiment_log.csv` EXP-014 | Not this phase's promotion bar directly (Phase 4 doesn't promote a champion — Phase 5 does), but the sanity floor step 4.9's proof-run GBM must not be *worse* than, to prove the features aren't actively harmful | Project Phase 3, EXP-014 |
| Real per-location ACF(1) | Computed twice already (notebooks 03 and 04, both from real `Train.csv`, both cross-checked against Experiment 5's stored ρ) | Step 4.2 promotes this into real, tested `src/` code instead of a third notebook copy-paste — the handoff pattern this project has now used twice (`run_tier3_sequential_state` was the last one) | `notebooks/04_baselines.ipynb` §2 |

---

## 1. Governing invariants (apply to every step below, not just one)

- **Origin-time indexing, not fold-level OOF.** `signature(location, t)` must be built only from rows with
  `time < t`. A signature built from the whole 2002-2015 record and merely excluded from one CV fold is
  **not** the same guarantee and is **not** acceptable — this is `ARCHITECTURE.md` §4's explicit,
  mechanically-checked invariant, and it is the single easiest thing to get subtly wrong in this phase.
- **No naive per-location statistics.** Every location-level aggregate (signature mean/std/trend/seasonality/
  ACF, and any spatial-neighbor aggregate) must be shrinkage-regularized. A-014 is not a hypothetical risk
  here — it is a measured failure of the naive version of exactly this kind of feature.
- **S1 (concurrent-neighbor) features are out of scope for this phase.** They're real but nearly useless in
  the regime that matters (blackout months mask the whole grid together) — `ARCHITECTURE.md` §9/§3. Every
  spatial feature built in step 4.3 is S2 (historical) or S3 (signature); if anything resembling same-month
  neighbor TWS is ever added, it must be explicitly labeled S1 and justified for why it survives despite the
  0.981-correlation-but-unusable finding.
- **Leakage-safe transformers, fit-only-on-train.** Every feature-producing object in this phase (signatures,
  spatial history, temporal, environmental, target transforms) implements the same `fit(train_df)` /
  `transform(df)` shape (step 4.4's `Transformer` protocol) — never a bare function that silently computes
  over whatever frame it's handed, train or validation alike.
- **Every new feature module gets a real leakage test**, not just a unit test of its arithmetic — `tests/
  test_no_leakage_features.py` was written in Phase 2 as a *vacuous* pass (no feature modules existed yet to
  scan); step 4.8 is where it stops being vacuous.
- **Config-driven, not hardcoded.** Shrinkage's `k`, trailing windows (12/24 months), neighbor count/radius —
  all live in `configs/features/*.yaml`, matching the project's existing `configs/validation/*.yaml`
  discipline, not as magic numbers in Python.
- **One file, one commit, one test.** Each numbered step below is sized to be exactly one commit (or one
  tight file+test pair), per the project's standing convention.

---

## 4.1 — Reconcile and build the `StateSnapshot` schema

**A real spec gap, worth resolving explicitly before writing any code — don't silently pick one document
over the other.** `ARCHITECTURE.md` §4's `StateSnapshot` field list has `previous_known_tws` and
`historical_delta` but no explicit `second_previous_known_tws` or `state_acceleration` field.
`PROJECT_PLAN.md`'s Phase 4 bullet asks for "observation trajectory (last_known, previous_known,
second_previous_known, from which state velocity **and acceleration** are derived)" — acceleration needs a
third point. Per `ARCHITECTURE.md` §2's own governance rule (architecture ranks above the project plan, and
evidence that contradicts a document triggers an ADR, never a silent pivot), this gets a short ADR:

**ADR-0006 (write first, before any code in this step):** extend `StateSnapshot` by exactly two fields —
`second_previous_known_tws: float | None` and `state_acceleration: float | None` — reasoning: `historical_delta`
already serves as state *velocity* (`last_known_tws - previous_known_tws`), so only acceleration and the
one extra trajectory point it requires are genuinely missing from the existing spec; this is a minimal,
justified extension, not a redesign. Record it, then implement against the extended schema.

**File:** `src/tws_forecast/state/reconstruction.py` (same file `ForecastOrigin` already lives in, per
`ARCHITECTURE.md` §6's explicit co-location and this file's own docstring, written in Phase 2 specifically
to say "not added until Phase 4" — that line gets deleted this step, not left stale).

**Contents:**
- `StateSnapshot` — frozen dataclass (or pydantic model, consistent with `ForecastOrigin`), fields exactly
  per `ARCHITECTURE.md` §4 plus ADR-0006's two additions: `last_known_tws, last_known_time,
  months_since_observation, previous_known_tws, second_previous_known_tws, historical_delta,
  state_acceleration, local_trend, seasonal_position, acf_1_3_6_12, observation_density,
  blackout_streak_length, location_signature, state_status`.
- `state_status` derivation, made precise (the field's meaning was named but not yet operationalized):
  `OBSERVED` — `TWS_t` is directly present at the forecast origin. `RECONSTRUCTED` — `TWS_t` is masked, but
  `last_known_tws` exists within a reasonable recency window (config-driven threshold) and the location's
  signature has adequate evidence (`w` above a config-driven floor). `PARTIALLY_RECONSTRUCTED` — `TWS_t` is
  masked **and** either the location has no prior observation at all (a never-before-seen location, or one
  whose only history predates `min_train_months`), or the signature's shrinkage weight `w` is below the
  floor (too little evidence to trust it) — the case every downstream consumer (uncertainty architecture,
  MoE gating, the eventual deployment UI) most needs a red flag for.
- `build_state_snapshot(df: pd.DataFrame, as_of: pd.Timestamp, location_id: str, trailing_windows=(12, 24)) ->
  StateSnapshot` — the actual reconstruction function, built strictly from `df[df["time"] < as_of]` for the
  given location (never `<=`, matching `ForecastOrigin.information_cutoff`'s existing `<=` origin_time
  semantics carefully — the *state* snapshot's own cutoff is exactly `information_cutoff`, i.e. everything
  strictly before the *next* month being forecast, which for horizon=1 is everything up to and including
  `origin_time` itself. State the exact boundary condition in the docstring and pin it with a test — this is
  precisely the kind of off-by-one that `ARCHITECTURE.md` §4's leakage discipline exists to catch).
- A **vectorized** batch variant, `build_state_snapshots(df, as_of_column="time") -> pd.DataFrame`, computing
  the same fields for every row of a frame at once (groupby + shift/rolling, not a per-row Python loop over
  millions of rows) — the same design lesson `attach_forecast_origin_columns` already applied in Phase 2, and
  the one that will actually get used in the feature-assembly pipeline (step 4.9); `build_state_snapshot`
  (singular) stays available for single-row diagnostics and tests, exactly mirroring `ForecastOrigin.from_row`
  vs. `attach_forecast_origin_columns`'s existing relationship.

**Tests:** `tests/test_state_snapshot.py` — the boundary condition (a row exactly at `as_of` is included or
excluded per the documented rule, tested explicitly, not just implied); `state_status` classifies correctly
on three hand-built fixtures (one per status); the vectorized batch variant produces results identical to
calling the per-row function on every row (a consistency test, mirroring the one `test_splitters.py` already
runs for `ForecastOrigin`); a never-observed location correctly resolves to `PARTIALLY_RECONSTRUCTED` with
`last_known_tws=None` rather than raising or silently defaulting to zero.

**Deliverable:** the single canonical `StateSnapshot` implementation every later step in this phase — and
every model from Project Phase 5 onward — consumes, with no competing definition of "months since
observation" anywhere else in the codebase.

## 4.2 — Historical location signatures, with shrinkage

**File:** `src/tws_forecast/state/signatures.py`

**Contents:**
- `LocationSignature` — small typed container: `mean, std, trend, seasonality_amplitude, acf_1, acf_3,
  acf_6, acf_12, spei_response, soil_moisture_response, n_observations, shrinkage_weight`.
- `compute_location_signature(df: pd.DataFrame, location_id: str, as_of: pd.Timestamp) -> LocationSignature`
  — built strictly from `df[(df["location_id"] == location_id) & (df["time"] < as_of)]`. Mean/std/trend/
  seasonality amplitude computed directly; ACF(1/3/6/12) via the same per-location autocorrelation logic
  notebooks 03/04 already used twice (`groupby + shift + corr`) — **promoted into this real module now**,
  rather than becoming a third notebook copy-paste, matching this project's own established pattern
  (`run_tier3_sequential_state` was the last thing promoted this way). SPEI/soil-moisture response: simple
  per-location correlation between `target` (or `historical_delta`) and each covariate, over the same
  historical window.
- **Shrinkage, applied to every one of the above except `n_observations`/`shrinkage_weight` themselves:**
  `θ̂_location = w · θ_location + (1−w) · θ_global`, `w = n / (n + k)`, `k` a config-driven constant
  (`configs/features/signatures.yaml`, step 4.4) defaulting to a value chosen by the ablation in step 4.9
  (start from k such that a location needs roughly 2-3 years of monthly history, ~24-36 observations, before
  its own signal outweighs the global prior — tune empirically, don't hardcode a guess). `θ_global` is
  computed once per fold from `df[df["time"] < as_of]` globally (the same information-cutoff discipline,
  just unconditioned on location) — **never** the full-record global mean, which would itself leak future
  information into early folds.
- **Vectorized batch variant**, `compute_location_signatures(df, as_of_column="time") -> pd.DataFrame`, same
  reasoning as step 4.1's batch function — this is the one actually used at scale; do not compute 15,715
  signatures with a Python loop calling the single-location function 15,715 times per fold.

**Tests:** `tests/test_signatures.py` — origin-time-indexing is enforced (a signature at `t` is unchanged
when rows at/after `t` are removed — the exact `historical_only_check` shape from `leakage_tests.py`,
run here directly as a unit test before step 4.8 wires it into the standing suite); shrinkage weight `w`
increases monotonically with `n_observations` and asymptotes toward 1; a location with `n=0` history returns
`θ_global` exactly (`w=0`), not `NaN` or a raised exception; **the A-014 regression test** — a synthetic
fixture with many sparse `(location, month)` cells (the exact failure condition A-014 measured) shows the
shrunk signature's out-of-fold RMSE beating naive per-cell means, the direct confirmation A-014's own
"validation experiment" column asked for.

**Deliverable:** the `location_signature` field of every `StateSnapshot` from here on is shrinkage-regularized
and origin-time-indexed by construction — directly closing A-014's outstanding validation experiment, not
just implementing what the architecture always said to build.

## 4.3 — Historical spatial-history features

**File:** `src/tws_forecast/state/spatial_history.py`

**Contents:** every feature here is **S2 (historical) or S3 (signature)** per the taxonomy in `ARCHITECTURE.md`
§9 — never S1 (concurrent same-month neighbor values), consistent with the verified 0.981-correlation-but-
unusable-during-blackouts finding (`PROJECT_PLAN.md` "Key findings"). Concretely:
- **Neighbor selection**: k-nearest-neighbor by great-circle distance on `(lat, lon)` (the grid is
  ~1-degree, so a simple haversine or even planar approximation is adequate — document which was used and
  why), `k` config-driven (`configs/features/spatial_history.yaml`). No basin dataset is sourced as of this
  phase (`PROJECT_PLAN.md`'s "basin-aware aggregation if a basin dataset is sourced" is explicitly
  conditional) — implement geographic-distance neighbors now, leave a documented extension point for
  basin-aware aggregation rather than blocking this phase on sourcing external basin data.
- `neighbor_TWS_last_known` (S2) — for each of a location's k neighbors, that neighbor's own `last_known_tws`
  from its `StateSnapshot` at the same origin (i.e. this function composes with step 4.1's output, it doesn't
  recompute last-known logic independently — no second copy of that logic anywhere in the codebase).
- `neighbor_TWS_lag_3`, `neighbor_TWS_lag_6` (S2) — neighbors' calendar-lag values (step 4.1's `calendar_lag`
  quantity), aggregated (mean, and optionally the closest neighbor's own value) across the k neighbors.
- `neighbor_historical_anomaly`, `neighbor_trend`, `neighbor_seasonal_signature`, `neighbor_ACF` (S2/S3) —
  aggregates of neighbors' own `LocationSignature` fields (step 4.2's output) — again, composing existing
  signature logic rather than reimplementing it for neighbors.
- All neighbor aggregates use **inverse-distance weighting**, not a flat mean, so a k=8 neighborhood doesn't
  treat a neighbor 50km away identically to one 300km away — config-driven whether to use this or a flat
  mean, defaulting to inverse-distance per basic spatial-statistics practice, but this is an ablatable choice
  step 4.9's proof notebook should sanity-check, not an unexamined default.

**Tests:** `tests/test_spatial_history.py` — every feature this module produces is explicitly tagged with its
S-category in a small registry (`SPATIAL_FEATURE_TAXONOMY: dict[str, Literal["S1","S2","S3","S4"]]`) and a
test asserts **no S1 feature exists in this module** — a standing, mechanical guard against the exact failure
mode `ARCHITECTURE.md` §9 warns about (a concurrent-neighbor feature silently contributing nothing in
blackout regimes); origin-time indexing (neighbor features at `t` unchanged by rows at/after `t`, same
pattern as step 4.2); a synthetic 3-location fixture where two locations are geographically close and one
far, confirming the close neighbors dominate the inverse-distance-weighted aggregate as expected; a location
with zero neighbors within any reasonable radius (an edge-of-grid or isolated point) falls back gracefully
(e.g. to the shrinkage-regularized global signature) rather than raising or returning NaN.

**Deliverable:** a spatial-history feature set that is provably S2/S3-only (mechanically checked, not just
documented), ready for step 4.9's feature-importance pass and Project Phase 6's later direct ablation of
A-003.

## 4.4 — `Transformer` protocol + `configs/features/` registry

**Files:** `src/tws_forecast/features/base.py`; `configs/features/signatures.yaml`,
`configs/features/spatial_history.yaml`, `configs/features/temporal.yaml`, `configs/features/environmental.yaml`.

**Contents:**
- `Transformer` protocol (`features/base.py`) — deliberately mirrors `validation.tiers.Predictor`'s existing
  shape: `fit(train_df: pd.DataFrame) -> None`, `transform(df: pd.DataFrame) -> pd.DataFrame`. Every
  feature-producing class from steps 4.2, 4.3, 4.5, 4.6 implements this, so a future feature pipeline
  (Project Phase 5+) can compose an arbitrary list of transformers uniformly, the same way the harness
  composes an arbitrary `Predictor`.
- Config files carry every tunable this phase introduces: shrinkage `k`, neighbor count and distance-weighting
  choice, trailing window lengths (12/24 months), SPEI-differencing lags, drought-run-length thresholds —
  matching the existing `configs/validation/*.yaml` discipline (`validation.scenarios.load_scenario`'s
  pattern), not hardcoded Python constants scattered across feature modules.
- `features/registry.py` (small, mirrors `validation/scenarios.py`'s `load_scenario`/`list_scenarios` shape)
  — `load_feature_config(name: str) -> FeatureConfig`.

**Tests:** `tests/test_feature_base.py` — every config YAML parses into a valid typed config; a stub
`Transformer` satisfies the protocol structurally (mirrors the existing `Predictor` protocol test pattern).

**Deliverable:** every feature module built from this step forward is config-driven and structurally
uniform — no feature-specific one-off calling convention for anything downstream to special-case.

## 4.5 — Temporal features (seasonal, trend)

**File:** `src/tws_forecast/features/temporal.py`

**Contents:** `Transformer` implementations for: trailing linear-trend slope per location (a rolling OLS slope
over the trailing 12/24-month window, origin-time-indexed — this is a *feature*, distinct from
`StateSnapshot.local_trend`, which may reuse the same underlying computation; don't duplicate the math, do
expose it through both the canonical `StateSnapshot` field and a standalone feature column if downstream
code needs it as an explicit model input); month × hemisphere interaction (categorical/cyclical cross
feature, motivated directly by the grid's Northern-Hemisphere skew — `ARCHITECTURE.md` §7 — and the risk
that a single global seasonal signal misrepresents the Southern Hemisphere's opposite phase).

**Tests:** `tests/test_temporal_features.py` — trend slope on a synthetic linearly-trending fixture recovers
the known slope within tolerance; **A-011's specific risk gets its own test**: performance/behavior on the
zero-representation month (October) and the under-represented months (April/May/Aug/Nov) is checked
explicitly — not just implicitly covered by a general CV pass, since A-011 flagged this exact gap as
"real and actionable, not yet acted on."

**Deliverable:** the seasonal/trend feature set `PROJECT_PLAN.md` names, built with A-011's calendar-coverage
risk checked directly rather than hoped away.

## 4.6 — Environmental features

**File:** `src/tws_forecast/features/environmental.py`

**Contents:** SPEI differencing (`SPEI_03_t - SPEI_03_t-3`-style deltas across the four available timescales);
drought-persistence run-length (consecutive months below a config-driven SPEI drought threshold, per
location); soil-moisture trajectory (the same lag/trend treatment `StateSnapshot` gives TWS, applied to
`SOIL_MOISTURE_t`, since Phase 1 established SPEI_12 as by far the strongest single covariate and soil
moisture as a meaningful secondary one — `PROJECT_PLAN.md` "Key findings").

**Tests:** `tests/test_environmental_features.py` — differencing/run-length arithmetic verified on a small
hand-built fixture; leakage check (a differencing feature at `t` never reads a covariate value at `time >= t`).

**Deliverable:** the environmental-feature set `PROJECT_PLAN.md` names, on the same leakage-safe `Transformer`
footing as everything else in this phase.

## 4.7 — Target transformation comparison

**File:** `src/tws_forecast/features/targets.py`, plus `notebooks/05_state_features.ipynb` §N (the controlled
experiment itself lives in the proof notebook, step 4.9 — this file just implements the five candidate
transforms as reusable, invertible functions).

**Contents:** five transforms, each with a `forward(df) -> pd.Series` and an `inverse(predictions, df) ->
pd.Series` (predictions must be invertible back to the raw `TWS(t+1)` level the competition actually scores
against — RMSE is computed on the real target, not whatever transformed space a model happens to train in):
**level** (`target` as-is, the current default / control condition); **delta** (`target - TWS_t`, undefined
when `TWS_t` is masked — document the fallback, e.g. `target - last_known_tws` when masked, consistent with
how `HybridPersistencePredictor` itself resolves the same ambiguity); **anomaly** (`target -
location_signature.mean`, i.e. relative to the shrinkage-regularized climatology from step 4.2, **not** the
naive climatology A-014 just showed is actively harmful — this is a direct, concrete way this phase's own
earlier work feeds its later work); **trend-residual** (`target - (local_trend extrapolated one month
forward)`); **volatility-normalized delta** (`(target - TWS_t) / location_signature.std`, again the
shrinkage-regularized std, not a naive per-location std).

**Tests:** `tests/test_target_transforms.py` — `inverse(forward(df)) == df["target"]` exactly, for all five
transforms, on a fixture with both observed and masked rows (the round-trip/invertibility property is the
one thing that must never silently break, since a bug here would corrupt every downstream RMSE without
necessarily looking wrong).

**Deliverable:** five interchangeable, leakage-safe, round-trip-verified target framings, ready for the
controlled head-to-head comparison in step 4.9 — per `ARCHITECTURE.md` §9, "the persistence-plus-delta
baseline is the floor this comparison has to clear, not a preordained conclusion about which framing wins."

## 4.8 — Leakage tests, made real

**File:** extend `tests/test_no_leakage_features.py` (written in Phase 2 as a deliberately vacuous pass — this
is where it stops being vacuous) and `src/tws_forecast/validation/leakage_tests.py` if any of its four
generic checks need a small extension to accept the new module shapes.

**Contents:** run all four of Phase 2's generic leakage checks (`future_row_shuffle_test`,
`historical_only_check`, `rolling_window_cutoff_check`, `masking_simulator_no_leak_check` — `PHASE2_
EXECUTION_PLAN.md` §2.8) against every real `Transformer` this phase built: `state/signatures.py`,
`state/spatial_history.py`, `features/temporal.py`, `features/environmental.py`. Also run the disallowed-
feature-name scan (already implemented in Phase 2) against this phase's actual output column names for the
first time — confirm nothing resembling `test_row_index`/`relative_test_position` slipped in.

**Deliverable:** the leakage firewall (`ARCHITECTURE.md` §7) is exercised against every real feature this
project has ever built, not just golden-fixture toy examples — the standing guard Phase 2 built now has
real, current-phase work to actually guard.

## 4.9 — Feature-assembly pipeline + proof notebook

**Files:** `src/tws_forecast/features/assemble.py` (a single `build_feature_matrix(df, ...) -> pd.DataFrame`
composing steps 4.1-4.6's outputs into one flat, model-ready frame — the first piece of what
`ARCHITECTURE.md` §6 eventually calls `pipelines/train.py`, though the full pipeline module is Project
Phase 10's job; this is deliberately the minimum needed for this phase's own proof run, not an early start
on Phase 10); `notebooks/05_state_features.ipynb`.

**Notebook contents**, mirroring notebooks 03/04's now-established pattern (setup → real-data run → findings
→ closing synthesis, executed locally against the real `Train.csv`, not in the cloud session per the same
wall-clock-budget reasoning notebooks 03/04 both documented):
1. Build the full feature matrix via `build_feature_matrix()` across a few CV folds.
2. **Leakage proof**: the literal future-row-shuffle test, run end-to-end against the assembled matrix, not
   just the individual transformers in isolation (step 4.8 covers the units; this covers the composed whole).
3. **Target-transformation comparison** (step 4.7): a single LightGBM (or the existing `RidgeBaselinePredictor`
   as a cheaper first pass, LightGBM for the real comparison), trained once per transform, scored through Tier
   1 + Tier 2 with identical folds and identical features otherwise — report which transform wins and by how
   much, per `ARCHITECTURE.md` §9's instruction that this is "a first-class experimental question," not a
   foregone conclusion. **Adopt the winner as this project's default target framing from here on** — record
   the decision as a short ADR if it's not "level" (i.e. if the default actually changes), since that's
   exactly the kind of downstream-affecting decision `ARCHITECTURE.md` §2's ADR mechanism exists for.
4. **A-014 direct confirmation**: shrinkage-regularized signature-based prediction (step 4.2) vs. Phase 3's
   naive `SeasonalClimatologyPredictor` (Baseline C), both scored on the identical folds — confirm the shrunk
   version beats 1.0796 and ideally beats the global mean (0.8740) too, closing A-014's "direct confirmation
   still pending" status to `Validated`.
5. **Feature-importance pass** (this phase's actual Definition of Done requirement): a LightGBM trained on
   the full assembled feature matrix (raw columns + `StateSnapshot` fields + signatures + spatial-history +
   temporal + environmental), run through `harness.evaluate_candidate()` exactly like Project Phase 3's six
   baselines were. Report SHAP or native feature-importance, **decomposed by regime** (masked vs. observed,
   per the Definition of Done's own wording) — does `acf_1_3_6_12`/`months_since_observation` actually
   dominate importance in the masked regime the way A-008/A-010 predict? Sanity-check against Baseline D's
   floor (0.6381 Tier 2) and specifically against the k=5/6/7 buckets — **this is not a promotion decision**
   (Project Phase 5 owns that, against the full champion ladder), it's this phase's own proof that the
   features it built are not actively harmful and behave the way the architecture predicts they should.
6. Closing synthesis, matching notebooks 03/04's now-established closing-section pattern.

**Deliverable:** `ARCHITECTURE.md` §4's leakage invariant demonstrated end-to-end on the real assembled
pipeline (not just unit-tested per-module); the target-transformation question answered with real numbers,
not left as an open architectural question; A-014 moved from "surprising finding" to "confirmed, and fixed";
a real feature-importance ranking, decomposed by regime, ready for Project Phase 5 to build its GBM on top
of with evidence about which features actually matter, not just architectural intuition about which ones
should.

## 4.10 — Documentation closure

- [ ] `PROJECT_PLAN.md` Phase 4 section trimmed to a short summary + pointer to this document (this step's
  own doc edit, done alongside publishing this plan — see below).
- [ ] `PROJECT_PLAN.md` Phase 4 checkboxes ticked as each step above completes; STATUS: MET note added once
  step 4.9's notebook is executed, summarizing the target-transformation winner, the A-014 confirmation, and
  the feature-importance findings — the same pattern Phase 2/3's STATUS notes already established.
- [ ] `ARCHITECTURE.md` §20 status paragraph extended with Phase 4's closure and headline findings.
- [ ] `docs/ASSUMPTIONS.md` — A-014's status updated from "Validated" (Phase 3's surprise finding) to
  reflect step 4.9's direct confirmation experiment; a new entry only if step 4.9 surfaces something
  genuinely unanticipated (don't force one — Phase 2/3's own closure notes are explicit that a phase without
  a real surprise doesn't need a manufactured entry).
- [ ] `docs/OPEN_QUESTIONS.md` — the spatial-stability question (A-003) stays open, explicitly noted as still
  scoped to Project Phase 6's direct ablation, not silently marked resolved just because Phase 4 built the
  spatial-history features that ablation will eventually test.
- [ ] `docs/adr/0006-*.md` — ADR-0006 (step 4.1's `StateSnapshot` field extension) and, if step 4.9's target-
  transformation comparison changes the project's default away from "level," a second ADR recording that
  decision.
- [ ] `reports/experiments/experiment_log.csv` — confirm the feature-importance proof run (step 4.9) is
  logged with real Tier 1/2/3 numbers, continuing the EXP-NNN sequence from EXP-015.

---

## Phase-level Definition of Done (restated, now step-mapped)

The state-reconstruction pipeline (`StateSnapshot`, step 4.1; signatures, step 4.2; spatial history, step
4.3; temporal/environmental features, steps 4.5-4.6) runs end-to-end inside CV folds with no leakage —
verified mechanically (shuffling future data must not change past predictions), both per-module (step 4.8)
and against the fully assembled pipeline (step 4.9). A feature-importance pass shows sensible results,
decomposed by regime (step 4.9) — meaning it's checked against A-008/A-010's own predictions about what
should matter, not just eyeballed. At that point Phase 4 is closed the same way Phases 1-3 were: real
numbers in `PROJECT_PLAN.md`/`ARCHITECTURE.md`, any genuine surprise logged to `docs/ASSUMPTIONS.md`, and
Project Phase 5 (core gradient-boosted forecasting) begins against a proven feature layer rather than
starting from raw columns the way Project Phase 2's own bare-LightGBM stand-in had to.

## Suggested build order (dependency-respecting)

4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6 → 4.7 → 4.8 → 4.9 → 4.10. Steps 4.2 and 4.3 both depend on 4.1
(`StateSnapshot`/its batch builder); 4.5-4.7 are independent of each other and of 4.2/4.3, so may be
reordered or built in parallel with each other, but all of 4.1-4.7 must exist before 4.8 (which scans all of
them) and 4.9 (which assembles and proves all of them together). 4.4 (the `Transformer` protocol and config
registry) is listed after 4.1-4.3 above only because those are the steps whose ambiguity (the ADR-0006
schema question) needed resolving first — in practice, write 4.4 immediately after 4.1, before 4.2/4.3, so
those two steps have the protocol and config pattern to implement against from the start rather than
retrofitting it.
