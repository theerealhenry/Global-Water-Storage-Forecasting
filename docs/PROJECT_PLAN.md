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

1. [x] **Reproduce the masking process.** Now rigorously reproduced end-to-end in `notebooks/02_forecastability.ipynb` (Project Phase 0's informal check upgraded to a real, executed notebook). Confirmed and refined: 18 non-contiguous test months split cleanly bimodal — 6 fully observed (0% masked) and 12 blackout months (99.58%-99.97% masked, no in-between); every test month is short of the full 15,715-location grid as rows (38-195 locations absent, not merely masked — a new finding, not previously stated per-month); blackout-month partial recovery is exhaustively confirmed scattered/non-recurring (all 66 month-pairs checked, overlap 0-29, mean 1.53, zero always-unmasked locations), refining the earlier "0-2 overlap" claim and surfacing a weak same-calendar-month echo (Feb 2016 vs Feb 2017) that doesn't change the overall conclusion; and a sourced (not yet formally Experiment-7-verified) preview cross-check shows the test set's entirely-absent months (Jul 2017-Jun 2018) closely matching the documented GRACE→GRACE-FO hard gap.
2. [x] **Persistence ceiling and the 2015 anomaly (hard gate).** Resolved in `notebooks/02_forecastability.ipynb` §7-8: NOT a partial-year/seasonal-composition artifact (ruled out directly — pooled 2002-2014 Jan-Aug vs Sep-Dec ratio only 1.03) and NOT a data-quality/outlier artifact (broad-based, not outlier-driven; global, not regional). Genuine regime characteristic — episodic within 2015 (Jan/Feb/Mar/Jun/Jul elevated, Apr/May/Aug normal), directional (systematic negative residual shift), plausibly explained by the sourced 2015-16 El Nino event (`docs/ASSUMPTIONS.md` A-007, not yet verified month-by-month against an ENSO index). Decision: don't discard/down-weight 2015, but don't assume 2002-2014 volatility is representative of the test period either — informs Project Phase 2 validation-fold design and Project Phase 4 trend-feature robustness.
3. [x] **Blackout-degradation curve — the highest-priority single experiment in the project.** Complete: `notebooks/02_forecastability.ipynb` §9-10. Simulated multiple sampled contiguous blackouts (15 overlapping windows, K=9 months, verified against an independent non-overlapping subset for robustness) on the confirmed gap-free 2004-2010 span, using last-observation-carried-forward as the baseline. Pooled RMSE grows 0.537→0.884 over 9 months (1.65x). **Stratified as required** — the pooled curve does hide different subpopulations, but not quite the illustrative pattern originally sketched here: per-location ACF(1) quartile is the dominant factor (0.403 RMSE spread at k=9), but low-ACF locations are worse in *absolute* terms at every horizon while high-ACF locations degrade *proportionally* faster (2.15x vs 1.32x growth) — both matter for feature design. Latitude band is next (0.191 spread, driven by an 0-30S/tropical-Southern-Hemisphere outlier plausibly linked to Experiment 2's El Nino finding), then season (0.079, weak) and onset drought regime (0.066, weak but directionally consistent with notebook 01's mean-reversion finding). An AR(1) theoretical model validates the curves' shape but under-predicts real predictability, pointing at exploitable structure beyond simple lag-1 persistence. Directly reprioritizes Project Phase 4: ACF/historical-signature features first, region-aware features second (`docs/ASSUMPTIONS.md` A-008).
4. [x] **Last-known-state baseline** (not climatology) — see the four distinct baselines in Project Phase 3. Complete: `notebooks/02_forecastability.ipynb` §11-12. Corrected a subtle staleness-definition bug before computing anything: staleness must be measured to the *target* month (row_month+1, per `docs/DATA_DICTIONARY.md`'s target definition), not the row's own month. Reconstructed the real 18-month test temporal structure (6 FULL + 12 BLACKOUT offsets) and found the 12 real blackout months carry staleness-to-target k=2 through k=7 (not a single fixed value: k=2×4, k=3×3, k=4×2, k=5/6/7×1 each). Two independent methods agree closely: Method A (reweighting Experiment 3's already-validated RMSE(k) curve by this real distribution) gives RMSE≈0.709; Method B (direct replay of the exact real temporal pattern onto 8 independent windows of the verified clean 2004-2010 span, ground-truth scored) gives RMSE=0.7145 — agreement within 0.005 pooled and 0.028 mean-abs per-offset, strong cross-validation. Full four-baseline picture computed on the real/replayed test structure: **A** (oracle persistence, FULL months only) 0.5247; **B** (last-known-state, BLACKOUT months only) 0.7145; **C** (seasonal climatology, all months) 0.817; **D** (Hybrid, all 18 months) 0.6573. **Reframes the internal target ladder** (`COMPETITIVE_ANALYSIS.md` §6): those targets were calibrated against Baseline A's in-sample 0.572, which is unreachable on the real test set since 12/18 months have no current observation — Baseline D (0.6573) is the actual realistic naive floor. These are Project Phase 1 forecastability numbers, not yet Project Phase 3's formal, harness-validated baselines (those get recomputed inside the Project Phase 2 validation harness for the official record) — but they're a strong, cross-validated preview.
5. [ ] **Staleness × location-dynamics interaction** — does ACF explain the shape of the degradation curve from step 3? Directly informs whether "staleness" alone is the right state variable or whether "staleness × location persistence" is needed.
6. [ ] **Covariate shift** — P(X_train) vs. P(X_test) for SPEI, soil moisture, month, masking rate (KS statistic or histogram comparison).
7. [ ] **Real GRACE/GRACE-FO mission timeline (external research, not leakage).** NASA/JPL publish the documented mission gap history — grounds "months since the gap started" features in the actual mission calendar instead of inferring it purely from our sparse sample of masked test months.

**Engineering foundation (parallel, not blocking the sequence above):**
- [x] `src/tws_forecast/data/loaders.py`, `pandera` schema contract
- [x] Formal EDA notebook (`notebooks/01_eda.ipynb`) and forecastability notebook (`notebooks/02_forecastability.ipynb`, Experiment 1 section) reproducing the above with saved figures
- [x] Unit tests for loaders/schema (`tests/test_loaders.py`, `tests/test_contracts.py`, golden fixtures — 20/20 passing)

**Definition of Done:** all 7 experiments answered or explicitly deferred with a reason; the 2015 anomaly specifically is resolved (or we have a documented, evidence-based decision on how to handle it if it can't be fully explained); both notebooks run top-to-bottom and regenerate all figures.

---

## Project Phase 2 — Validation harness

**Objective:** a validation scheme that won't lie to us, built on what Project Phase 1 established rather than assumptions.

- [ ] **Time-respecting CV**: expanding-window splits, not random K-fold
- [ ] **Streak-aware masking simulator**: nulls `TWS_t` (and dependent features) in contiguous multi-month blocks per location, matching the real structure — never a flat row-independent probability
- [ ] **Three validation tiers, each answering a different question, not just three sets of folds:** Tier 1 (forecastability) — can we predict a future month under normal observation conditions? Tier 2 (blackout) — can we forecast after losing the current TWS observation, and how gracefully does accuracy degrade with staleness? Tier 3 (test-regime) — can we reproduce the exact observation/masking/environmental conditions the actual test months represent?
- [ ] **Two integrity safeguards, as explicit rules, not just intentions:** (1) Tier 3 (test-matched) validation is for diagnosis and robustness assessment only — final model selection uses predefined historical folds, not repeated tuning against test-specific analogs. (2) Test row ordering is descriptive metadata only — no `test_row_index`/`relative_test_position` features, no inference about the public/private split from row order.
- [ ] **Error decomposition table**, filled in for every model from here on (standing artifact, not a one-off): overall / masked / unmasked / by staleness bucket (1-2mo, 3-4mo, 5+mo) / by hemisphere / on extreme-TWS and rapid-change slices
- [ ] **Degradation slope** (ΔRMSE / Δmonths-since-observation) tracked as a fourth metric alongside RMSE — how gracefully, not just how accurately, the model handles staleness
- [ ] Internal target ladder (< 0.572 / < 0.559 / < 0.53 / < 0.50), used as a promotion rule against the *full* error-decomposition table, not the headline number alone — a model that wins on aggregate while being fragile in one regime doesn't get promoted
- [ ] Every experiment run logged (Project Phase 0's flat log, migrating to MLflow): params, CV fold scores per tier, full decomposition, git commit hash, data version

**Definition of Done:** running the harness against a trivial model produces all three tiers, the full decomposition table, and the degradation slope; we both agree the scheme can't leak the future and honestly reflects the real blackout structure.

---

## Project Phase 3 — State-aware baselines (the bar to beat)

Four *distinct* baselines, not one baseline with a fallback bolted on — conflating them (as an earlier draft did, calling a persistence/climatology hybrid "naive persistence") obscures what's actually being measured:

- [ ] **Baseline A — Oracle persistence**: ŷ = TWS_t, computed only on rows where TWS_t exists. Answers: how hard is the unmasked problem?
- [ ] **Baseline B — Last-observation-carried-forward**: ŷ = TWS at the most recently observed month, for masked rows. Answers: how far can we get from historical state reconstruction alone, with zero learned correction?
- [ ] **Baseline C — Seasonal climatology**: independent fallback, per-location per-calendar-month mean (already measured: 0.817, weak).
- [ ] **Baseline D — Hybrid**: if TWS_t available, use it; else use last-known. The realistic "no ML" reference point for the actual test structure.
- [ ] Global mean and Ridge regression as further reference points
- [ ] All logged with the full error-decomposition table (Project Phase 2)

**Definition of Done:** all four baselines decomposed by regime; we know the exact numbers every subsequent model must beat, per regime, not just on average.

---

## Project Phase 4 — State reconstruction & feature engineering

**Objective:** build the observation-state reconstruction layer as one coherent, explicit pipeline stage (see the central-hypothesis diagram above), not scattered ad hoc features — every downstream model consumes the same consistently-defined state representation.

- [ ] **State reconstruction layer**, precisely defined (these are four different things, not interchangeable): calendar lag (TWS at exactly t−k, itself often missing inside a blackout streak) vs. last-observed lag (TWS at the most recent *actually observed* month) vs. observation age (t − t_last_observed) vs. observation trajectory (last_known, previous_known, second_previous_known, from which state velocity and acceleration are derived) — plus observation_count/density over trailing 12/24 months.
- [ ] **Historical location signature features, with explicit shrinkage** — mean, std, trend, seasonality amplitude, ACF(1/3/6/12), SPEI/soil-moisture response, computed out-of-fold. Shrunk toward the global estimate as θ̂_location = w·θ_location + (1−w)·θ_global, w increasing with the amount/quality of location-level evidence (empirical-Bayes style, e.g. w = n/(n+k)) — not naive per-location statistics, which overfit given only ~150 monthly observations per location.
- [ ] **Historical spatial-history features — corrected from an earlier draft.** Not concurrent-month k-NN (same-month neighbors are almost always masked together during blackouts, per the verified 0.981-correlation-but-unusable finding). Instead: `neighbor_TWS_last_known`, `neighbor_TWS_lag_3/6`, `neighbor_historical_anomaly`, `neighbor_trend`, `neighbor_seasonal_signature`, `neighbor_ACF`. Basin-aware aggregation if a basin dataset is sourced.
- [ ] Seasonal/trend features: trailing linear-trend slope per location, month × hemisphere interaction
- [ ] SPEI differencing, drought-persistence run-length features, soil-moisture trajectory
- [ ] Target transformation comparison in one controlled experiment: delta / anomaly / volatility-normalized delta / trend-residual
- [ ] All transforms as testable, leakage-safe transformers (fit only on train fold)

**Definition of Done:** state-reconstruction pipeline runs end-to-end inside CV folds with no leakage (verified: shuffling future data must not change past predictions); feature-importance pass shows sensible results, decomposed by regime.

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
