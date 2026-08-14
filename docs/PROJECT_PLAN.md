# Global Water Storage Forecasting — Project Plan

**Competition:** ITU/UN "AI for Good" — Forecasting Global Water Storage (GRACE TWS, one-month-ahead)
**Goals (in priority order):** (1) win first prize, (2) build genuine DS/ML skill, (3) produce a senior-level portfolio piece.
**Scoring:** 50% leaderboard RMSE (Competition Phase 1) + 50% written report on Trustworthiness (30%) / Innovation & Practicality (20%), judged only for top-10 finishers (Competition Phase 2).

**Terminology note:** this document's phases ("Project Phase 0", "Project Phase 1", ...) are our own work breakdown and are unrelated to the competition's own "Phase 1 / Phase 2" scoring structure named above. Written as "Project Phase N" wherever the two could be confused.

## How we'll work

One phase at a time. I implement a phase, explain what I did and why, show you the results, and stop for your sign-off before starting the next one. Each phase below has a **Definition of Done**. This is a living roadmap, now on its second major restructuring after `COMPETITIVE_ANALYSIS.md` matured past the plan's original "EDA → features → GBM → advanced model → explainability → production" shape. Everything is built in a real git repo (`tws-forecast/`) so the commit history is portfolio evidence of process, not just a final artifact.

---

## Central hypothesis (read this before anything else)

**This is not primarily a one-month-ahead regression problem. It's a partially-observed state-reconstruction and state-transition forecasting problem.** Original framing: `TWS_t + SPEI + soil moisture → TWS_t+1`. Actual framing, supported by everything found so far: `historical observations → reconstruct hydrological state → estimate how that state evolves → forecast TWS(t+1)`. This explains persistence's unusual strength, why the delta is much harder to predict than the level, why 66% of test rows lack current TWS, why masking arrives in contiguous month-level blocks rather than scattered per-row, why location history matters, why staleness matters, why same-month spatial neighbors go dark together during blackouts, and why external hydrological forcing should matter *more*, not less, exactly when TWS_t is missing. **Caveat worth stating precisely in any report:** "hydrological state" is our modeling hypothesis, not something directly observed — what we actually observe is TWS, SPEI, soil moisture, and the mask indicator. Full reasoning and evidence: `COMPETITIVE_ANALYSIS.md` §4.

```
                    Raw observations (TWS, SPEI, soil moisture, mask)
                                     │
                                     ▼
                     Observation-state reconstruction layer
        (last-known value, observation age, state velocity/
         acceleration, ACF, historical location signature)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                             ▼
   Temporal-history            Environmental forcing         Spatial-history
   reconstruction              (SPEI, soil moisture,         reconstruction
   (lags, trend, ACF)          external hydrology)           (historical, not concurrent)
        │                            │                             │
        └────────────────────────────┼─────────────────────────────┘
                                     ▼
                       Specialist predictors (persistence/
                       state-transition, environmental,
                       historical-location, spatial-history, analog)
                                     │
                                     ▼
                  Conditional gate / OOF-optimized blend
                (built only once specialists earn it — Phase 7)
                                     │
                                     ▼
                     Residual / bias / physics-informed correction (OOF only)
                                     │
                                     ▼
                                TWS(t+1)
```

Our competitive advantage should not be "a better LightGBM" — it should be a better representation of hydrological state when the most informative observation is missing. Everything below follows from that.

---

## Key findings driving the design (from EDA, all verified — full detail in `COMPETITIVE_ANALYSIS.md` §3)

- **Target = TWS at t+1, exactly** (verified programmatically, zero leakage ambiguity).
- **Persistence (predict "no change") already gets RMSE 0.572** vs a target std of 0.912. Any model must decisively beat 0.572 to be worth anything, and per the delta-correlation finding below, most of the room to beat it lies in the masked regime, not the easy one.
- **66% of test rows have `TWS_t` itself missing**, clustered into whole "blackout" months mirroring the real 2017-2018 GRACE→GRACE-FO gap. A model that only works well when TWS_t is present will fail on 2/3 of the leaderboard.
- **Non-stationary trend**: mean target drifts from +0.23 (2002) to ~-0.13 (2012-2015) — random K-fold CV would leak the future.
- **Grid is land-only, Northern-Hemisphere-heavy** (5,573 locations 30-60°N vs 570 at 60-30°S). Real generalization/bias question, not just a footnote.
- **SPEI_12 >> SPEI_01 in predictive power** (r=0.38 vs 0.20); SPEI features are collinear with each other (up to r≈0.7).
- **Nearest-neighbor TWS is extremely correlated within a month (r=0.981)**, but nearly unusable during blackouts since the whole grid is masked in lockstep — spatial signal in the hard regime must come from history, not concurrent readings.
- **The delta (target − TWS_t) is weakly explained by every given feature** — TWS_t itself correlates −0.32 with the delta (mean-reversion), best SPEI feature only −0.065. Caps how much "persistence + correction" can achieve on the *unmasked* third alone.
- **Persistence RMSE by year is stable (~0.50–0.63) 2002–2014, then jumps to 0.898 in 2015** — unresolved; treated as a hard gate in Project Phase 1, not just an open question, because it could mean the years immediately preceding test are qualitatively different from the training bulk.

---

## Project Phase 0 — Minimal reproducible foundation

**Objective:** just enough engineering to run reproducible experiments — not full production polish. Split explicitly so engineering doesn't compete with modeling insight for time this early.

**Required before experimentation:**
- [x] Git repo initialized, standard `src/` layout, `configs/`, `notebooks/`, `tests/`, `docker/`, `.github/workflows/`, `docs/`, `reports/figures/`
- [x] Data access (symlinked from the uploads mount; see data policy below)
- [ ] `requirements.txt` pinning exact versions, Python 3.11
- [ ] Minimal experiment logging: a flat table (csv/json is fine to start) recording experiment name, data version, training cutoff, blackout-simulation parameters, model, seed, and CV RMSE per tier — the substance of reproducibility, not the MLflow UI
- [ ] `docs/DATA_DICTIONARY.md`

**Can catch up (not a blocker for Project Phase 1):**
- [ ] Full MLflow (SQLite backend, Model Registry) — upgrade the flat log above once the validation harness (Project Phase 2) stabilizes
- [ ] Pre-commit hooks (`black`, `ruff`, `isort`, `mypy`)
- [ ] `pydantic` config validation
- [ ] Polished `README.md`

**Data versioning decision (resolved, not left open):** no DVC. Train.csv is large, externally supplied, static, and not something we redistribute — the portfolio value of "reproducible data acquisition + documented contract" is comparable to "DVC used" here without the setup overhead. Instead: `data/raw/` gitignored, `data/README.md` (tracked, download instructions), `dataset_manifest.json` (tracked, row counts + SHA256 hash of each file) so anyone reproducing the repo can verify they have the exact same data. Revisit DVC only if multiple dataset versions actually emerge.

**Definition of Done:** a clean checkout plus the manifest/download instructions reproduces the exact dataset (hash-verified); experiments are logged reproducibly even before MLflow is fully wired up.

---

## Project Phase 1 — Forecastability & data-generating process

**Objective:** understand exactly what we're being asked to forecast, and how hard each regime actually is, before finalizing validation design or writing modeling code. (This phase absorbs an earlier proposal for a separate "forensics" stage — folded in here rather than given a new number, since this *is* what Project Phase 1 is for.)

**Ordered experiment sequence** — each builds on the last, run in this order:

1. [x] **Reproduce the masking process.** Now rigorously reproduced end-to-end in `notebooks/02_forecastability.ipynb` (Project Phase 0's informal check upgraded to a real, executed notebook). Confirmed and refined: 18 non-contiguous test months split cleanly bimodal — 6 fully observed (0% masked) and 12 blackout months (99.58%-99.97% masked, no in-between); every test month is short of the full 15,715-location grid as rows (38-195 locations absent, not merely masked — a new finding, not previously stated per-month); blackout-month partial recovery is exhaustively confirmed scattered/non-recurring (all 66 month-pairs checked, overlap 0-29, mean 1.53, zero always-unmasked locations), refining the earlier "0-2 overlap" claim and surfacing a weak same-calendar-month echo (Feb 2016 vs Feb 2017) that doesn't change the overall conclusion; and a preview cross-check (later formally confirmed by Experiment 7, item 7 below) showed the test set's entirely-absent months closely matching the documented GRACE→GRACE-FO hard gap.
2. [x] **Persistence ceiling and the 2015 anomaly (hard gate).** Resolved in `notebooks/02_forecastability.ipynb` §7-8: NOT a partial-year/seasonal-composition artifact (ruled out directly — pooled 2002-2014 Jan-Aug vs Sep-Dec ratio only 1.03) and NOT a data-quality/outlier artifact (broad-based, not outlier-driven; global, not regional). Genuine regime characteristic — episodic within 2015 (Jan/Feb/Mar/Jun/Jul elevated, Apr/May/Aug normal), directional (systematic negative residual shift), plausibly explained by the sourced 2015-16 El Nino event (`docs/ASSUMPTIONS.md` A-007, not yet verified month-by-month against an ENSO index). Decision: don't discard/down-weight 2015, but don't assume 2002-2014 volatility is representative of the test period either — informs Project Phase 2 validation-fold design and Project Phase 4 trend-feature robustness.
3. [x] **Blackout-degradation curve — the highest-priority single experiment in the project.** Complete: `notebooks/02_forecastability.ipynb` §9-10. Simulated multiple sampled contiguous blackouts (15 overlapping windows, K=9 months, verified against an independent non-overlapping subset for robustness) on the confirmed gap-free 2004-2010 span, using last-observation-carried-forward as the baseline. Pooled RMSE grows 0.537→0.884 over 9 months (1.65x). **Stratified as required** — the pooled curve does hide different subpopulations, but not quite the illustrative pattern originally sketched here: per-location ACF(1) quartile is the dominant factor (0.403 RMSE spread at k=9), but low-ACF locations are worse in *absolute* terms at every horizon while high-ACF locations degrade *proportionally* faster (2.15x vs 1.32x growth) — both matter for feature design. Latitude band is next (0.191 spread, driven by an 0-30S/tropical-Southern-Hemisphere outlier plausibly linked to Experiment 2's El Nino finding), then season (0.079, weak) and onset drought regime (0.066, weak but directionally consistent with notebook 01's mean-reversion finding). An AR(1) theoretical model validates the curves' shape but under-predicts real predictability, pointing at exploitable structure beyond simple lag-1 persistence. Directly reprioritizes Project Phase 4: ACF/historical-signature features first, region-aware features second (`docs/ASSUMPTIONS.md` A-008).
4. [x] **Last-known-state baseline** (not climatology) — see the four distinct baselines in Project Phase 3. Complete: `notebooks/02_forecastability.ipynb` §11-12. Corrected a subtle staleness-definition bug before computing anything: staleness must be measured to the *target* month (row_month+1, per `docs/DATA_DICTIONARY.md`'s target definition), not the row's own month. Reconstructed the real 18-month test temporal structure (6 FULL + 12 BLACKOUT offsets) and found the 12 real blackout months carry staleness-to-target k=2 through k=7 (not a single fixed value: k=2×4, k=3×3, k=4×2, k=5/6/7×1 each). Two independent methods agree closely: Method A (reweighting Experiment 3's already-validated RMSE(k) curve by this real distribution) gives RMSE≈0.709; Method B (direct replay of the exact real temporal pattern onto 8 independent windows of the verified clean 2004-2010 span, ground-truth scored) gives RMSE=0.7145 — agreement within 0.005 pooled and 0.028 mean-abs per-offset, strong cross-validation. Full four-baseline picture computed on the real/replayed test structure: **A** (oracle persistence, FULL months only) 0.5247; **B** (last-known-state, BLACKOUT months only) 0.7145; **C** (seasonal climatology, all months) 0.817; **D** (Hybrid, all 18 months) 0.6573. **Reframes the internal target ladder** (`COMPETITIVE_ANALYSIS.md` §6): those targets were calibrated against Baseline A's in-sample 0.572, which is unreachable on the real test set since 12/18 months have no current observation — Baseline D (0.6573) is the actual realistic naive floor. These are Project Phase 1 forecastability numbers, not yet Project Phase 3's formal, harness-validated baselines (those get recomputed inside the Project Phase 2 validation harness for the official record) — but they're a strong, cross-validated preview.
5. [x] **Staleness × location-dynamics interaction** — does ACF explain the shape of the degradation curve from step 3? Directly informs whether "staleness" alone is the right state variable or whether "staleness × location persistence" is needed. **Complete: `notebooks/02_forecastability.ipynb` §13-14.** Verdict: **PARTIALLY — real, but not linear.** A direct linear-interaction regression (error² ~ k + acf1 + k·acf1) found the interaction term added essentially zero R² (0.0645→0.0645) — but the AR(1) theoretical model (parameter-free, driven only by each quartile's ρ and σ) explains R²=0.448 of the empirical curve's variance, and a finer 10-way ACF-decile stratification is monotonic at 9/9 values of k. Resolution: k and ACF/σ genuinely interact, but the true relationship is nonlinear (AR(1)'s ρᵏ form), so a raw `k×acf1` product term is the wrong functional form and understates the effect — a linear-test null result should not be read as "no interaction." A confound check also found TWS volatility (σ) is a largely independent signal from ACF (r=-0.141) and adds substantially more explanatory power (+0.0462 R², +71.6% relative) than the ACF-only interaction did. Direct implication for `StateSnapshot`: `months_since_observation` alone is not the right state variable — its meaning depends on the location's ACF/volatility profile, justifying keeping `acf_1_3_6_12` and `location_signature` (mean/std) distinct rather than collapsed. Recommends Project Phase 4 test an explicit AR(1)-motivated composite feature (`sigma*sqrt(2*(1-rho**k))`), not just a raw multiplicative term, alongside letting the GBM learn the interaction natively.
6. [x] **Covariate shift** — P(X_train) vs. P(X_test) for SPEI, soil moisture, month, masking rate (KS statistic or histogram comparison). **Complete: `notebooks/02_forecastability.ipynb` §15-16.** Four findings: (1) masking rate is 0% (train) vs. 66.5% (test) — by far the largest covariate-regime difference, but it's the project's existing central design focus, not a new risk. (2) SPEI (all 4 timescales)/soil moisture, train vs. all of test: minimal shift, largest KS statistic 0.0352 (SOIL_MOISTURE_t). (3) SPEI/soil moisture, test-unmasked vs. test-masked rows within the same period: also minimal (largest KS 0.0569) — the blackout regime is not meaningfully environmentally distinct from the observed regime within test, consistent with A-001's account of masking as a hardware/mission-timeline cause, not an environmental one. (4) Calendar-month coverage: test entirely omits October (0 of the 18 test months fall in October) while train has full coverage — a genuine, actionable generalization risk for Phase 4's seasonal features and Phase 2's fold design. Relationship to A-003: this experiment provides indirect evidence only (marginal-covariate stability, not spatial relationship stability) and does not contradict A-003 as measured, but is not a direct test of it.
7. [x] **Real GRACE/GRACE-FO mission timeline (external research, not leakage).** NASA/JPL publish the documented mission gap history — grounds "months since the gap started" features in the actual mission calendar instead of inferring it purely from our sparse sample of masked test months. **Complete: `notebooks/02_forecastability.ipynb` §17-18.** Sourced directly from JPL's GRACE Tellus site, JPL's mission pages, and a peer-reviewed paper (Landerer et al. 2020, *Geophysical Research Letters*, DOI:10.1029/2020GL088306) — not a single secondary summary. Key dates: GRACE launched 2002-03-17, last usable ranging data June 2017 (JPL states "GRACE ended its science mission in October 2017" for the formal end-of-operations milestone — both reported, not conflated); GRACE-FO launched 2018-05-22, first monthly fields June 2018; Landerer et al. describe an "11-month gap" (Jul 2017-May 2018) with no detectable intermission bias between the missions. Three cross-checks: (1) the test set's single longest absent run (2017-07 to 2018-06, 12 months) matches the documented 11-month gap exactly plus one extra month (2018-06, GRACE-FO's own commissioning-adjacent first month, plausibly excluded by the competition's creators) — a near-exact, well-evidenced match. (2) 17 of the 22 missing TRAINING months (2011 onward, in 8 distinct ~2/year events of 1-2 months each) match the documented "battery management" outage cadence (since 2011, ~every 6 months, 4-8 weeks) — resolving a gap A-001 previously flagged as unexplained; 5 pre-2011 missing months remain unexplained. (3) The test set also has 4 smaller scattered absent runs beyond the main gap — one (2016-10 to 2016-11) coincides with GRACE-2's accelerometer shutdown and is plausibly real; the other 3 don't align with any specific sourced event and are flagged as an open item (possibly Zindi's own month-selection choices, not proven satellite unavailability) rather than force-fit into the narrative. **A-001 formally upgraded from Active to Validated** on the strength of finding (1) alone; findings (2)-(3) are bonus resolutions/honest open items, not required for A-001's core claim.

**Engineering foundation (parallel, not blocking the sequence above):**
- [x] `src/tws_forecast/data/loaders.py`, `pandera` schema contract
- [x] Formal EDA notebook (`notebooks/01_eda.ipynb`) and forecastability notebook (`notebooks/02_forecastability.ipynb`, Experiment 1 section) reproducing the above with saved figures
- [x] Unit tests for loaders/schema (`tests/test_loaders.py`, `tests/test_contracts.py`, golden fixtures — 20/20 passing)

**Definition of Done:** all 7 experiments answered or explicitly deferred with a reason; the 2015 anomaly specifically is resolved (or we have a documented, evidence-based decision on how to handle it if it can't be fully explained); both notebooks run top-to-bottom and regenerate all figures.

**STATUS: MET.** All 7 experiments complete with full findings (above). The 2015 anomaly is resolved (Experiment 2: genuine regime characteristic, plausibly El Niño-linked, not an artifact — `docs/ASSUMPTIONS.md` A-004/A-007). Both `notebooks/01_eda.ipynb` and `notebooks/02_forecastability.ipynb` execute top-to-bottom with 0 errors and all figures regenerated (verified `2026-08-11`, see the Phase 1 review note in `reports/experiments/experiment_log.csv` / this file's revision history). Project Phase 1 is formally complete — see `docs/adr/` for the Phase 2 kickoff decision record.

---

## Project Phase 2 — Validation harness

**Objective:** a validation scheme that won't lie to us, built on what Project Phase 1 established rather
than assumptions. **Full step-by-step build plan moved to `docs/PHASE2_EXECUTION_PLAN.md`** (written
2026-08-11, after a module-map reconciliation pass against `ARCHITECTURE.md` — see ADR-0005,
`docs/adr/0005-validation-module-map-reconciliation.md`) — that document is now the authoritative,
single source of truth for Phase 2's build order, so it isn't duplicated and independently maintained here.
This section stays as a summary and status tracker only.

**Design constants** (the exact Phase 1 numbers every Phase 2 mechanism must reproduce, not re-derive):
real blackout staleness-to-target distribution k=2(×4)/3(×3)/4(×2)/5,6,7(×1 each); real test FULL/BLACKOUT
calendar offsets; Baselines A/B/C/D = 0.5247/0.7145/0.8170/0.6573; the October calendar-coverage gap
(A-011); the verified gap-free 2004-2010 training span; the nonlinear staleness×ACF relationship (A-010).
Full table with sources: `docs/PHASE2_EXECUTION_PLAN.md` §0. Binding rationale: ADR-0004.

**Module map** (per ADR-0005, reconciled against `ARCHITECTURE.md` §6/§11 — supersedes any earlier draft
naming): `src/tws_forecast/state/reconstruction.py` (`ForecastOrigin`, Phase 2; `StateSnapshot` added
Phase 4), `validation/phase1_constants.py`, `validation/splitters.py`, `validation/masking_simulator.py`
(`MaskingScenario` objects), `validation/scenarios.py` (config-driven registry backed by
`configs/validation/*.yaml`), `validation/tiers.py`, `validation/decomposition.py`,
`validation/leakage_tests.py`, `validation/harness.py` (orchestrator + promotion rule), plus
`utils/seeds.py` and `utils/config.py`.

**Build order** (2.1 → 2.12, each one commit — full detail per step, including exact function signatures,
tests, and deliverables, in `docs/PHASE2_EXECUTION_PLAN.md`):

- [x] 2.1 Prerequisites — seeds, minimal config loader, `phase1_constants.py` (7 commits, 42/42 tests passing; constants re-verified byte-exact against `notebooks/02_forecastability.ipynb`'s executed output cells before being hardcoded)
- [x] 2.2 `ForecastOrigin` schema (2 commits, 52/52 tests passing; leakage invariants — `target_time == origin + horizon`, `information_cutoff <= origin_time` — enforced in `__post_init__`, not just by convention)
- [x] 2.3 Expanding-window splitter (time-respecting, 2015-anomaly-confronting fold design) — `validation/splitters.py`, 2 commits, 64/64 tests passing; verified by direct inspection that fold 1's training portion ends exactly at the clean-span boundary (2010-12) and the final fold validates on 2015-03→2015-08, squarely inside the anomaly window
- [x] 2.4 `MaskingScenario` + streak-aware masking simulator — `validation/masking_simulator.py`, 2 commits, 81/81 tests passing; `TWS_t_masked == TWS_t.isna()` enforced by construction (never a documentation promise); `transition_pattern` values beyond "abrupt" raise `NotImplementedError` rather than being silently treated as abrupt, since only "abrupt" is backed by Phase 1 evidence
- [x] 2.5 Config-driven scenario registry (`configs/validation/*.yaml`) — `validation/scenarios.py`, 4 scenario YAMLs (`expanding_window`, `blackout_curve`, `test_regime_replay`, `2015_like`) + registry, 6 commits, 98/98 tests passing; `test_regime_replay`/`blackout_curve`'s offsets and k-values checked for drift against `phase1_constants.py` directly
- [x] 2.6 Three validation tiers (forecastability / blackout / test-regime) — `validation/tiers.py`, 9 commits total (incl. `utils/dates.py` extraction, `apply_blackout_curve` extension to `masking_simulator.py`, promoting `attach_forecast_origin_columns` to public API), 141/141 tests passing; manually verified Tier 3's replay offsets alternate FULL/BLACKOUT exactly matching the real calendar pattern, and Tier 1/Tier 2 RMSE are identical for a TWS_t-blind mean predictor (correct, not a bug — confirms masking only affects models that actually read the masked column)
- [x] 2.7 Error decomposition table + degradation slope (real k-buckets × ACF quartile, AR(1) reference overlay) — `validation/decomposition.py`, 4 commits total (incl. `true_tws_t` column added to `tiers.py`), 160/160 tests passing; manually verified the "no rows silently dropped" invariant, graceful handling of an empty slice (0 Northern-hemisphere rows in the golden fixture), and the AR(1) reference curve's shape qualitatively matches Experiment 5 (Q1/low-ACF starts high & flat, Q4/high-ACF starts low & steep)
- [x] 2.8 Leakage firewall as executable checks (future-row shuffle, historical-only, rolling-cutoff, masking no-leak, disallowed-feature-name scan) — `validation/leakage_tests.py`, 3 commits, 178/178 tests passing; each of the three generic checks proven against both a correct and a deliberately leaky toy example (not just exercised on code that already passes) before Project Phase 4 has any real feature/signature functions to check
- [x] 2.9 Harness orchestrator + promotion rule (Tier-3-only promotion hard-blocked in code) — `validation/harness.py`, 2 commits, 195/195 tests passing; promotion ladder evaluated against Tier 2's overall RMSE (the tier structurally analogous to Baseline D's mixed regime, without relying on Tier 3); manually verified a weak mean-only baseline clears exactly the `naive_floor` rung on the golden fixture, and self-comparison against its own report as baseline shows zero regression
- [x] 2.10 Experiment log migration + MLflow kickoff — `validation/experiment_log.py`, 2 commits, 208/208 tests passing; single `log_candidate()` entrypoint writes both halves atomically (flat CSV row continuing the EXP-NNN sequence from Phase 1's EXP-001..EXP-007, plus one MLflow run against a SQLite-backed `mlflow.db` with the full decomposition tables + degradation slope logged as CSV artifacts); `mlflow==3.15.1` (already pinned) verified installable and working end-to-end in the dev sandbox; manually verified a real `evaluate_candidate()`/`promote()` result on the golden fixture logs correctly to both a temp CSV and a temp MLflow run (metrics, params, and 3 decomposition artifacts all confirmed present via `MlflowClient`); real project `mlflow.db`/`mlruns/` deliberately left uncreated until the step-2.11 notebook's first real run, so this step doesn't itself add a spurious history entry
- [x] 2.11 Validation notebook (`notebooks/03_validation_harness.ipynb`) — proof run against Baseline D logic and a bare LightGBM, all three tiers, decomposition table, degradation-slope plot; 3 commits (notebook, figures, EXP-008/EXP-009 experiment log rows), real MLflow runs (`mlflow.db`/`mlruns/`) logged for the first time. **Phase-defining sanity check PASSES**: Tier 3 reproduces Baseline D's 0.6573 within tolerance (0.6319, diff 0.0254) — but getting there required finding and fixing two real issues along the way, not zero. (1) A genuine bug in `_select_replay_anchors` (anchors weren't restricted to the verified clean 2004-2010 span, letting replay windows run into documented post-2010 gaps) — fixed in `tiers.py` and pinned with 3 regression tests, own commits. (2) `run_tier3`'s row-wise, stateless-between-offsets design (deliberate, built for Phase 4's feature-based models) under-scores internally-stateful baselines like Baseline D's own logic — worked around via a diagnostic-only sequential-state replay, not a source change. Full write-up: `docs/ASSUMPTIONS.md` A-013. Substantive finding: bare LightGBM (raw columns only) beats Baseline D on Tier 1 (0.5798 vs 0.6380) and Tier 2 (0.5801 vs 0.6381) and clears the naive_floor rung, but is correctly BLOCKED from promotion head-to-head against Baseline D by `harness.promote()`'s hard-staleness-bucket regression safeguard (regresses on k=5/6/7) — the safeguard's first real, non-synthetic catch, and the sharpest evidence yet that Project Phase 4's historical-signature features (A-008, A-010) are the priority.
- [x] 2.12 Documentation closure pass (this section, `ARCHITECTURE.md` §20, ADR follow-ups) — `PROJECT_PLAN.md`/`ARCHITECTURE.md` §20 updated to reflect step 2.11's real results; `docs/ASSUMPTIONS.md` A-013 added (Tier 3 anchor-span bug + row-wise-scoring limitation); `experiment_log.csv`'s Tier 1/2/3 columns confirmed populated (not `N/A`) for EXP-008/EXP-009; ADR-0004/ADR-0005 follow-up checkboxes closed

**Definition of Done:** running the harness against a trivial model produces all three tiers, the full
decomposition table (including the ACF-quartile × staleness-bucket cross-cut), and the degradation slope
with the AR(1) reference overlay; Tier 3's naive-model score reproduces Baseline D's 0.6573 within a small
tolerance (validating the harness faithfully reproduces Phase 1's measured reality, per ADR-0004); the
leakage tests pass; every scenario used is a named config file, not inline logic; we both agree the scheme
can't leak the future and honestly reflects the real blackout structure. Full detail: `docs/PHASE2_EXECUTION_PLAN.md`.

**STATUS: MET.** All 12 steps complete (above), all built against Phase 1's measured constants, not
re-derived. `notebooks/03_validation_harness.ipynb` (step 2.11) is the phase-defining proof: run against
real Train.csv (2,154,021 rows), Tier 3's naive-model score reproduces Baseline D's 0.6573 within tolerance
(0.6319, diff 0.0254) once a genuine anchor-span bug is fixed (`tiers.py`, regression-tested) and
`run_tier3`'s deliberate row-wise/stateless-between-offsets design is accounted for via a diagnostic
sequential replay — full write-up `docs/ASSUMPTIONS.md` A-013. The leakage tests pass (step 2.8, 18 tests
against both correct and deliberately-leaky toy examples). Every scenario used is a named
`configs/validation/*.yaml` file, never inline logic (step 2.5). The harness's promotion safeguards are
proven against real candidates, not just toy fixtures: bare LightGBM beats Baseline D in aggregate
(Tier 1/2) but is correctly blocked from promotion by the hard-staleness-bucket regression check (k=5/6/7)
— exactly the failure mode that safeguard exists to catch. Project Phase 2 is formally complete; Project
Phase 3 (state-aware baselines) is next.

---

## Project Phase 3 — State-aware baselines (the bar to beat)

Full build plan: `docs/PHASE3_HANDOFF.md` (written 2026-08-13, at Project Phase 2's formal closure). Four
*distinct* baselines, not one baseline with a fallback bolted on — conflating them (as an earlier draft did,
calling a persistence/climatology hybrid "naive persistence") obscures what's actually being measured:

- [x] **Baseline A — Oracle persistence**: ŷ = TWS_t, computed only on rows where TWS_t exists. Answers: how hard is the unmasked problem? `OraclePersistencePredictor`, `src/tws_forecast/models/baselines.py`.
- [x] **Baseline B — Last-observation-carried-forward**: ŷ = TWS at the most recently observed month, for masked rows. Answers: how far can we get from historical state reconstruction alone, with zero learned correction? `LastKnownStatePredictor` — explicitly verified (`tests/test_baselines.py`) to never read the predict-time frame's own `TWS_t`, even when populated, which is the precise distinction from Baseline D's own forward-fill logic (see that class's docstring).
- [x] **Baseline C — Seasonal climatology**: independent fallback, per-location per-calendar-month mean (already measured: 0.817, weak). `SeasonalClimatologyPredictor`, keyed off the *target's* calendar month, matching Phase 1's original measurement precisely.
- [x] **Baseline D — Hybrid**: if TWS_t available, use it; else use last-known. The realistic "no ML" reference point for the actual test structure. `HybridPersistencePredictor`, promoted near-verbatim from `notebooks/03_validation_harness.ipynb`'s throwaway `BaselineDPredictor`, per the handoff's explicit instruction not to reimplement this logic a second time.
- [x] Global mean and Ridge regression as further reference points. `GlobalMeanPredictor`; `RidgeBaselinePredictor` (two internal Ridge models, one per observed/masked regime — see its docstring for why TWS_t isn't simply dropped).
- [x] **A-013 handling (handoff step 3.0):** `validation.tiers.run_tier3_sequential_state` promoted from notebook 03 §7b into tested `src/` code (`tests/test_tiers_sequential_state.py`) — the diagnostic-only, chronologically-ordered replay that lets Baselines B/D's internal state accumulate across a Tier 3 replay window, giving a number genuinely comparable to Phase 1's own replay measurements (B=0.7145, D=0.6573), reported *alongside*, never in place of, the standard harness-faithful `run_tier3` score.
- [x] All six candidates wired to run through `harness.evaluate_candidate()`/`promote()`/`log_candidate()` unmodified — `notebooks/04_baselines.ipynb`, executed end-to-end 2026-08-14 against the real 2,154,021-row `Train.csv` (13.2s load, 12.8s ACF computation over 15,715 locations, ~65-92s per candidate's Tier 1+2+3 evaluation).
- [x] Notebook executed end-to-end against the real `Train.csv`, decomposition tables reviewed, all six candidates logged for real (**EXP-010 through EXP-015**, `reports/experiments/experiment_log.csv` and real MLflow runs in `mlflow.db`/`mlruns/`), documentation closure with the real numbers (this section, `ARCHITECTURE.md` §20, `docs/ASSUMPTIONS.md` A-014).

**Definition of Done:** all four baselines decomposed by regime; we know the exact numbers every subsequent model must beat, per regime, not just on average.

**STATUS: MET.** All six candidates evaluated through the full three-tier harness against the real data,
decomposed by regime, logged for real. Headline numbers (Tier 2 overall RMSE — the tier `promote()`
evaluates the ladder against):

| Candidate | Tier 1 | Tier 2 | Tier 3 (standard) | Tier 3 (sequential-state) | Ladder rung | vs. Baseline D |
|---|---|---|---|---|---|---|
| Global mean | 0.8740 | 0.8740 | 0.8957 | n/a | none | regressed k=5/6/7 |
| **Baseline A** — oracle persistence | 0.6380 | 0.6383 | 0.7961 | n/a | naive_floor | regressed k=5/6/7 |
| **Baseline B** — last-known-state | 0.8532 | 0.8532 | 0.9369 | 0.7258 | none | regressed k=5 |
| **Baseline C** — seasonal climatology | 1.0796 | 1.0796 | 0.9403 | n/a | none | regressed k=5/6/7 |
| **Baseline D** — hybrid persistence | 0.6380 | **0.6381** | 0.8270 | **0.6319** | naive_floor | — (reference) |
| Ridge (SPEI/soil-moisture, 2-regime) | 0.5878 | 0.5880 | 0.7342 | n/a | naive_floor | regressed k=5/7 |

**Baseline D remains the undisputed realistic floor.** Its harness Tier 2 RMSE (0.6381) reproduces Phase 1's
own replay measurement (0.6573) closely, and its A-013-correct sequential-state Tier 3 number (0.6319) lands
even closer — both expected, cross-validating the harness against Phase 1 a second time (after Project
Phase 2's own proof run) on genuinely new candidates. **Every other candidate that clears the naive-floor
ladder rung in aggregate (Baseline A, Ridge) is still correctly blocked from promotion against Baseline D**
by `harness.promote()`'s hard-staleness-bucket safeguard — Ridge in particular is a real learned model using
SPEI/soil-moisture features and beats Baseline D by 0.05 RMSE in aggregate (0.588 vs. 0.638), yet regresses
on k=5 and k=7. This is the *second* time this exact failure mode has been caught by the same mechanism
(after bare LightGBM in Project Phase 2's proof run, EXP-009) — now demonstrated across two structurally
different model families (tree-based and linear), which is stronger evidence the safeguard is catching a
real property of this problem, not a LightGBM-specific quirk.

**One genuine surprise, not anticipated by Phase 1's preview:** Baseline C (seasonal climatology) is the
*worst* candidate evaluated — worse even than the global mean (Tier 2 1.0796 vs. 0.8740) — reversing
Phase 1's in-sample measurement (0.817, `notebooks/01_eda.ipynb`), which never held out the data it was
fit on. Naive per-`(location, calendar-month)` climatology overfits badly out-of-fold, because most of the
~15,715 × 12 cells it estimates have very little data per fold. This is the first real, out-of-fold evidence
(not just architectural reasoning) that Project Phase 4's planned shrinkage-regularized location signatures
are necessary, not merely a stylistic preference — see `docs/ASSUMPTIONS.md` A-014.

**A methodological limitation surfaced by this run, worth carrying into Project Phase 4/5's own reading of
Tier 2 diagnostics:** the per-ACF-quartile degradation-slope curves (section 7 of the notebook, all six
`degradation_slope_*.png` figures) are visibly jagged and non-monotonic for every one of the six candidates
— because Tier 2's synthetic blackout-curve scenario only samples 16-40 rows per staleness bucket per fold
set, split four ways by ACF quartile. All six candidates are scored against the *identical* masked rows
(the blackout simulator's seed doesn't depend on the model), so the noise is shared, not candidate-specific
— but it means individual quartile×k data points in this diagnostic are not statistically reliable at
Project Phase 3's sample size. Tier 3's decomposition (46,700+ rows per staleness bucket) is far more
stable and should be preferred for any claim that depends on a specific staleness bucket's exact value.

Full decomposition tables, promotion-decision reasoning, and the closing synthesis are in
`notebooks/04_baselines.ipynb` itself (executed, all outputs and figures saved) — including a detailed
executive-summary section appended after this closure.

---

## Project Phase 4 — State reconstruction & feature engineering

**Objective:** build the observation-state reconstruction layer as one coherent, explicit pipeline stage (see the central-hypothesis diagram above), not scattered ad hoc features — every downstream model consumes the same consistently-defined state representation. **Full step-by-step build plan moved to `docs/PHASE4_EXECUTION_PLAN.md`** (written 2026-08-14, at Project Phase 3's formal closure, following the same authoritative-single-source pattern established for Phase 2 by `docs/PHASE2_EXECUTION_PLAN.md`) — that document is now the authoritative source of truth for Phase 4's build order (steps 4.1–4.10), module map, `StateSnapshot` schema reconciliation (ADR-0006), and per-step deliverables/tests, so it isn't duplicated and independently maintained here. This section stays as a summary and status tracker only.

**Inherited from Phase 3 (what Phase 4 exists to fix):** Baseline D's realistic no-ML floor (Tier 2 0.6381, sequential-state Tier 3 0.6319) is unbeaten by any candidate that clears promotion — Ridge came closest in aggregate (0.588) but was correctly blocked twice over by the hard-staleness-bucket safeguard (k=5/6/7, then k=5/7). Baseline C's out-of-fold collapse (`docs/ASSUMPTIONS.md` A-014) is direct, real-data proof that naive per-location statistics overfit and that shrinkage-regularized signatures are necessary, not stylistic. Phase 4's job is to build the `StateSnapshot` representation and the S2/S3 historical-signature and spatial-history features that give Phase 5's GBM models something better than raw columns to work with — concurrent-month (S1) features remain explicitly out of scope (unusable during blackouts, per Phase 1's 0.981-correlation finding).

- [ ] **State reconstruction layer** (`StateSnapshot`, step 4.1) — calendar lag vs. last-observed lag vs. observation age vs. observation trajectory (velocity/acceleration), plus observation density and blackout-streak length. ADR-0006 reconciles the `ARCHITECTURE.md`/`PROJECT_PLAN.md` field lists.
- [ ] **Historical location signatures with explicit shrinkage** (step 4.2) — mean, std, trend, seasonality amplitude, ACF(1/3/6/12), SPEI/soil-moisture response, shrunk via θ̂ = w·θ_location + (1−w)·θ_global, w = n/(n+k). Includes a direct A-014 regression test.
- [ ] **Historical spatial-history features** (step 4.3) — k-NN by geographic distance, S2/S3 only (`neighbor_TWS_last_known`, `neighbor_TWS_lag_3/6`, `neighbor_historical_anomaly`, `neighbor_trend`, `neighbor_seasonal_signature`, `neighbor_ACF`); mechanically guarded against any S1 feature.
- [ ] **`Transformer` protocol + config-driven feature registry** (step 4.4).
- [ ] **Temporal features** (step 4.5) — trailing trend slope, month×hemisphere interaction, explicit A-011 (October gap) test.
- [ ] **Environmental features** (step 4.6) — SPEI differencing, drought-persistence run-length, soil-moisture trajectory.
- [ ] **Target transformation comparison** (step 4.7) — level/delta/anomaly/trend-residual/volatility-normalized delta, each with round-trip-invertible `forward`/`inverse`.
- [ ] **Real leakage tests against every new module** (step 4.8).
- [ ] **Feature-assembly pipeline + proof notebook** (`notebooks/05_state_features.ipynb`, step 4.9) — leakage proof, target-transform head-to-head, A-014 direct confirmation, regime-decomposed feature-importance pass (diagnostic only — not a Phase 5 promotion decision).
- [ ] **Documentation closure** (step 4.10).

**Definition of Done:** state-reconstruction pipeline runs end-to-end inside CV folds with no leakage (verified: shuffling future data must not change past predictions); feature-importance pass shows sensible results, decomposed by regime. Full detail: `docs/PHASE4_EXECUTION_PLAN.md`.

---

## Project Phase 5 — Core GBM forecasting

- [ ] LightGBM as primary candidate (native NaN handling) plus XGBoost/CatBoost for comparison
- [ ] **Streak-aware masking-augmentation training** — corrected wording from an earlier draft, which said "null out TWS_t for a matching fraction of rows" (row-independent, inconsistent with Project Phase 2's own streak-aware design). Training-time masking must simulate the same contiguous-block process as validation, or the model is optimized for an easier synthetic problem than the real one.
- [ ] Feature/state representation prioritized over hyperparameter tuning — representation matters far more here than squeezing a few more tree splits; don't spend days on Optuna before Project Phase 4's feature set is settled
- [ ] Champion selected against the full error-decomposition table (Project Phase 2), not overall RMSE alone

**Definition of Done:** a GBM model registered in the experiment log, beating all Project Phase 3 baselines across every regime in the decomposition table, full run reproducibility (fixed seeds, logged config).

---

## Project Phase 6 — External data & spatial-history forecasting

- [ ] External-data provenance and **vintage** gate before any feature enters the champion — table includes source, temporal resolution, release latency, **revision history** (reanalysis products get reprocessed; using a later-revised value that wasn't actually available in near-real-time is a subtle leakage risk), prediction-time availability, leakage risk
- [ ] Sharper hypothesis than "add ERA5": external forcing data's value should be concentrated in the *masked* regime specifically, since it's the best remaining signal about state transition when TWS itself is unavailable — test this directly (masked-regime RMSE with/without external features), don't just add features and hope
- [ ] Water-balance composites (P, ET, P−ET at 1/3/6/12-month windows) if precipitation/ET data is sourced
- [ ] Historical spatial-history features from Project Phase 4, extended with any external spatial/static data (elevation, land cover, climate zone) if sourced

**Definition of Done:** every external feature passes the provenance/vintage gate; masked-regime RMSE impact measured and documented separately from unmasked-regime impact.

---

## Project Phase 7 — Specialists, MoE & OOF ensemble

**Objective:** chase the "innovation" score and remaining RMSE without building unearned architectural complexity.

- [ ] **Named specialists first, no gating network yet**: persistence/state-transition expert, environmental-forcing expert, historical-location expert, spatial-history expert. Measure each one's error by regime (decomposition table) before deciding anything about blending them.
- [ ] **Conditional gate / mixture-of-experts — built only if the specialists show real regime-conditional advantages** (e.g. "expert A wins when TWS_t known, expert B wins in long blackouts"). If they don't, a simple OOF-optimized linear blend is enough, and that's fine — the competition rewards RMSE, not architectural sophistication.
- [ ] Analog forecasting (find historically similar SPEI/soil-moisture/season conditions at the same location, use what followed) as a secondary ensemble member, evaluated for whether it improves the blend
- [ ] OOF-optimized constrained blend weights over simple averaging
- [ ] Kill criteria applied throughout (`COMPETITIVE_ANALYSIS.md` §9) — e.g. drop the spatial-history branch if it doesn't move blackout-tier CV by ~0.002+; drop MoE if specialists don't earn it

**Definition of Done:** documented comparison of specialists vs. blend vs. (if built) gated MoE, with a decision justified by the decomposition table, not intuition.

---

## Project Phase 8 — Residual, bias & physics-informed correction

All strictly OOF — never fit on in-sample residuals.

- [ ] Residual modeling: second model on OOF residuals (location, season, SPEI, months-since-TWS, prediction as inputs), added back to the champion's prediction
- [ ] Systematic bias correction by location/region/season — doubles as direct evidence for Project Phase 9's Trustworthiness work, since the rubric explicitly asks about bias
- [ ] Physics-informed blend: if Project Phase 6 sourced water-balance data, blend the ML prediction with a physically-estimated ΔTWS (≈ P−ET) at an OOF-learned weight — a genuinely strong "physics-guided ML" narrative for Innovation & Practicality, contingent on external data actually being sourced
- [ ] Prediction reconciliation (temporal/spatial smoothing of implausible jumps) — tested, not assumed beneficial; RMSE doesn't automatically reward smoothing

**Definition of Done:** each correction's effect isolated and measured against the decomposition table; only kept if it improves the relevant tier without degrading another.

---

## Project Phase 9 — Trustworthiness, uncertainty & explainability

Feeds Competition Phase 2 directly (30% Trustworthiness + 20% Innovation & Practicality of final score). Broader than SHAP — SHAP is useful but isn't synonymous with explainability here.

- [ ] SHAP analysis (global + regional), but as one input among several, not the centerpiece
- [ ] **Stability**: does the model behave consistently across time, latitude, drought regime, blackout duration?
- [ ] **Sensitivity**: how do predictions respond to small changes in SPEI, last-known TWS, or staleness?
- [ ] **Failure modes**: where does it systematically fail, and why?
- [ ] **Bias**: systematic over/under-prediction by hemisphere, wet vs. dry regime, high vs. low TWS, short vs. long blackout — pulling directly from Project Phase 8's bias-correction analysis
- [ ] **Uncertainty signals**: ensemble variance, residual distribution by staleness/regime, and the degradation slope from Project Phase 2 — "the model knows when its state estimate is becoming stale" is a compelling, concrete finding even though the competition only scores point predictions
- [ ] `docs/BIAS_AND_EXPLAINABILITY.md` written directly against the rubric themes: bias, transparency, reusability, sustainability/efficiency, practicality

**Definition of Done:** a document we could hand to a judge today that honestly represents where the model is strong, where it isn't, and how confidently it knows the difference.

---

## Project Phase 10 — Production pipeline engineering

- [ ] All notebook logic promoted into tested `src/` modules — data, features, models, pipelines
- [ ] Config-driven single entrypoint
- [ ] `pytest` coverage for data loaders, feature transforms (including the state-reconstruction layer), model wrapper, prediction consistency
- [ ] Structured logging, full seed control

**Definition of Done:** a clean checkout plus one command reproduces the champion model's metrics within floating-point tolerance.

---

## Project Phase 11 — MLflow model registry & packaging

- [ ] Full MLflow Model Registry (Staging → Production), upgrading Project Phase 0's flat experiment log
- [ ] Packaged as an MLflow `pyfunc` model bundling the entire preprocessing/state-reconstruction pipeline
- [ ] `docs/MODEL_CARD.md`

**Definition of Done:** `mlflow.pyfunc.load_model(...).predict(raw_dataframe)` works with zero manual preprocessing.

---

## Project Phase 12 — CI/CD

- [ ] GitHub Actions: lint + test on every PR; build & smoke-test the Docker image on merge to main
- [ ] Pre-commit hooks mirrored exactly in CI

**Definition of Done:** a deliberately broken PR is blocked; a clean PR goes green end-to-end.

---

## Project Phase 13 — Deployment: interactive app

Sandbox has no cloud credentials, so the deliverable is a **fully working, containerized app runnable locally today, deployable by you in ~10 minutes** to a free host.

- [ ] FastAPI backend: `/predict` (single + batch), `/health`
- [ ] Interactive frontend (Streamlit or Gradio): pick a location + month, see the state-reconstruction inputs (last-known TWS, staleness, SPEI, soil moisture), get the predicted next-month TWS with an explanation and an uncertainty indicator (Project Phase 9), plus a world-map view
- [ ] `docker-compose.yml`; Hugging Face Spaces recommended deployment target

**Definition of Done:** `docker compose up` gives a working local demo; a documented `git push` deploys it live under your own account.

---

## Project Phase 14 — Documentation & portfolio packaging

- [ ] Final `README.md`: problem, approach, architecture diagram, results table, how to run, demo link
- [ ] `docs/CASE_STUDY.md` — the narrative arc is genuinely strong here and worth telling explicitly: not "I trained LightGBM on GRACE data" but "I discovered the core challenge wasn't forecasting TWS, it was reconstructing hydrological state during systematic GRACE observation blackouts" — EDA → persistence discovery → masking-process discovery → forecastability analysis → state reconstruction → regime-aware validation → conditional forecasting → external hydrological forcing → OOF ensemble. That sequence demonstrates statistical reasoning, missing-data modeling, competition strategy, leakage prevention, feature engineering, experiment design, and MLOps — a stronger signal for the senior-level goal than the model itself.

**Definition of Done:** a stranger could read the README in 5 minutes and understand what was built, why, and how well it works.

---

## Project Phase 15 — Competition submission & Competition Phase 2 report

- [ ] Final Test predictions validated against `SampleSubmission.csv` schema exactly
- [ ] Final 2 submissions selected using error correlation between candidates (`COMPETITIVE_ANALYSIS.md` §13), not just the two lowest CV RMSEs
- [ ] Competition Phase 2 written report addressing the rubric: AI Trustworthiness (pulling from Project Phase 9), Innovation & Practicality (state-reconstruction framing, streak-aware masking-augmentation, physics-informed blending if built)
- [ ] **Waiting on you** to upload the official Trustworthiness Evaluation rubric doc so this is written against the real criteria

**Definition of Done:** submission uploaded; report drafted against the real rubric.

---

## Project Phase 16 — Final QA & retrospective

- [ ] Full dry run: raw data → trained model → deployed prediction, end to end, on a clean checkout
- [ ] Retrospective notes — genuine learning artifact, good interview material

---

## Tech stack summary

| Concern | Choice | Why |
|---|---|---|
| Modeling | LightGBM/XGBoost/CatBoost primary; deep spatiotemporal models explicitly optional (Project Phase 7/`COMPETITIVE_ANALYSIS.md` §11) | Native NaN handling fits the masking problem; representation matters more than architecture here |
| Experiment tracking | Flat log first (Project Phase 0), MLflow (SQLite backend) once validation stabilizes | Reproducibility now, registry later — don't let tooling block modeling |
| HPO | Optuna, deprioritized until feature/state representation is settled | Representation > tuning for this problem |
| Data validation | pandera | Codifies exactly the invariants found in EDA |
| Data versioning | Gitignored raw data + manifest + hash (no DVC) | Sufficient for a static, externally-supplied dataset |
| Testing | pytest | Standard |
| API | FastAPI | Standard for ML serving |
| Interactive demo | Streamlit or Gradio | Fast, portfolio-friendly, HF Spaces native support |
| CI/CD | GitHub Actions | Free, standard |
| Deployment | Docker + Hugging Face Spaces | Free, no cloud creds needed |
| Explainability | SHAP + stability/sensitivity/failure-mode/bias/uncertainty analysis | SHAP alone isn't sufficient explainability for this problem (Project Phase 9) |

---

## Open decisions I need from you

1. **Streamlit vs Gradio** for the interactive demo — Gradio is faster to stand up and HF Spaces' native SDK; Streamlit gives more layout control. (Project Phase 13)
2. **Trustworthiness Evaluation rubric doc** — please upload when you can, ideally before Project Phase 9. (Project Phase 15)

(DVC is resolved — see Project Phase 0 — not left open.)

---

**Next immediate step:** Project Phase 1, experiment 2 — resolve the 2015 persistence-RMSE anomaly (hard gate), then experiment 3, the stratified blackout-degradation curve. Say go and I'll run experiment 2 first, explain the result, and stop for your review before the degradation curve.
