# TWS Forecasting — Architectural Blueprint

**Status:** authoritative. **Version:** v3.0. **Scope note:** this is the last planned revision of this document before implementation begins. Further architecture changes from here are expected to come from ADRs triggered by actual experiment evidence (per §0's own rule), not another review pass.

## 0. Governance: source-of-truth hierarchy and the no-drift rule

**Authority, highest to lowest, when documents conflict:**
```
1. Zindi competition rules          (absolute — never overridden)
2. This document (ARCHITECTURE.md)  (what the system is)
3. PROJECT_PLAN.md                  (execution order)
4. COMPETITIVE_ANALYSIS.md          (strategic reasoning / hypotheses)
5. Experiments                      (empirical evidence)
6. Implementation (code)            (what actually runs)
```
Evidence flows back *up* this chain through ADRs: an experiment result can invalidate an assumption (§13), which triggers an ADR, which updates this document, which may require a `PROJECT_PLAN.md` edit. Nothing skips a level — code should never silently diverge from the architecture, and the architecture should never silently diverge from what an experiment actually showed.

```
New finding → contradicts this architecture? → No: normal experiment, proceed.
                                               → Yes: ADR → evaluate impact →
                                                 update this doc → update
                                                 PROJECT_PLAN.md if needed →
                                                 implement → validate → record outcome.
```

## 1. Core architectural thesis

Not "predict TWS(t+1) from TWS(t) and environmental variables." Instead: **reconstruct the hydrological state of a known location from its historical observations and environmental context, then forecast that state's evolution one month forward.** Full evidence trail: `COMPETITIVE_ANALYSIS.md` §3-4. Caveat worth repeating: "hydrological state" is a modeling hypothesis, not something directly observed.

## 2. The information-set discipline (what "time" means here)

Every prediction is implicitly `ŷ(t+1) = f(I_t)`, where `I_t` is *everything knowable at the forecast origin t and nothing else*. This is stricter than "timestamp ordering" — it's why the leakage firewall (§4) exists and why external data needs a vintage rule, not just a source citation: a value labeled "January 2017" in an external dataset is not automatically something a January 2017 forecast could have used, since reanalysis products get revised and reprocessed after the fact. Two lightweight schemas make this concrete rather than aspirational, without requiring a new heavyweight pipeline stage:

**ForecastOrigin** (one row per training/validation/prediction example): `origin_time, target_time, horizon, information_cutoff, location_id, regime`. This is what lets us say "this model predicted September 2015 from information available at August 2015" instead of vaguely "this was a 2015 row" — and it's the join key everything else (state, features, OOF predictions) keys off.

**StateSnapshot** (the canonical, single definition of "what we know about a location's water state," computed once in `state/reconstruction.py` and consumed everywhere downstream — no feature module gets to compute its own version of "months since observation"): `last_known_tws, last_known_time, months_since_observation, previous_known_tws, historical_delta, local_trend, seasonal_position, acf_1_3_6_12, observation_density, blackout_streak_length, location_signature, state_status`. `state_status ∈ {OBSERVED, RECONSTRUCTED, PARTIALLY_RECONSTRUCTED}` — this one field is what lets MoE gating, uncertainty, explainability, and the deployment UI all ask the same question ("how much do we actually know right now?") without four different ad hoc implementations.

**Time-indexed signatures — a correction, not just a refinement.** Location signatures (`state/signatures.py`) were already specified as "computed out-of-fold," which is necessary but not sufficient: for a forecast at 2012, the signature must be `signature(location, forecast_origin)` built only from history *before* 2012, not `signature(location)` built from all of 2002-2015 and merely excluded from that one CV fold. Fold-level OOF and origin-time-indexed computation are different guarantees, and only the second one is actually correct here. This is now a stated architectural invariant, checked by `validation/leakage_tests.py`.

## 3. System architecture — nine layers plus governance spine

```
┌──────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION — Train/Test/external, dataset manifest   │
├──────────────────────────────────────────────────────────────┤
│ 2. DATA CONTRACT & VERSIONING — schema, hashes, invariants    │
├──────────────────────────────────────────────────────────────┤
│ 3. FORECAST INFORMATION SET — origin/cutoff enforcement,       │
│    external-data vintage lock (§2, §7)                        │
├──────────────────────────────────────────────────────────────┤
│ 4. STATE RECONSTRUCTION — StateSnapshot, observed/             │
│    reconstructed status, signatures, spatial history           │
├──────────────────────────────────────────────────────────────┤
│ 5. FEATURE ENGINE — temporal / environmental / spatial-history │
│    (S1-S4 taxonomy, §7) / external / target transforms         │
├──────────────────────────────────────────────────────────────┤
│ 6. VALIDATION ENGINE — 3-tier CV, masking scenarios, leakage   │
│    firewall, decomposition table, degradation slope            │
├──────────────────────────────────────────────────────────────┤
│ 7. MODEL RESEARCH — baselines → GBM → specialists → MoE →      │
│    OOF ensemble → residual/bias → physics-informed correction  │
├──────────────────────────────────────────────────────────────┤
│ 8. EVIDENCE & CANDIDATE SYSTEM — OOF store, candidate           │
│    registry, champion/challenger, uncertainty (§9)              │
├──────────────────────────────────────────────────────────────┤
│ 9. COMPETITION / PRODUCTION — submission subsystem ‖ serving    │
└──────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────────────────────┐
              │  GOVERNANCE SPINE               │
              │  Git · MLflow · ADRs ·           │
              │  Assumptions · configs · seeds   │
              └───────────────────────────────┘
```

Two research/evidence and production/competition views worth having as separate diagrams, since they answer different questions than the layer stack above:

**Research-evidence loop** (how any single finding turns into a decision): `Hypothesis → Experiment (Run) → Validation → Evidence (decomposition table + OOF) → Decision → Candidate → Submission → new evidence feeds back in`.

**Lineage** (how any single prediction traces backward): `Submission → Candidate → MLflow run → model version + feature version + validation version → git commit → dataset manifest`. Worked example already in §11.

## 4. Repository / module map

```
tws-forecast/
├── data/{raw,interim,processed,external}/     manifest-hash verified, raw gitignored
├── src/tws_forecast/
│   ├── data/            loaders.py, contracts.py (pandera), versioning.py
│   ├── state/            reconstruction.py (StateSnapshot, ForecastOrigin), signatures.py
│   │                      (time-indexed + shrinkage), spatial_history.py
│   ├── features/          temporal.py, environmental.py, external.py (vintage lock),
│   │                       targets.py, masking.py
│   ├── validation/         splitters.py, masking_simulator.py (MaskingScenario objects),
│   │                        scenarios.py (registry, config-driven), tiers.py,
│   │                        decomposition.py, leakage_tests.py
│   ├── models/               baselines.py, gbm.py, specialists.py, moe.py, ensemble.py,
│   │                          correction.py, research/deep.py (sandboxed — no core
│   │                          pipeline dependency on it unless it wins)
│   ├── pipelines/               train.py, evaluate.py, predict.py, submit.py, deploy.py
│   ├── serving/                   api.py, schemas.py, app.py
│   └── utils/                       config.py (pydantic), logging.py, seeds.py, reproducibility.py
├── artifacts/oof/<candidate_id>/     OOF prediction store — first-class, not notebook arrays
├── submissions/                       submission_log.csv, candidates.csv, files/, manifests/
├── configs/           base.yaml, validation/ (scenario configs), features/, models/, champion.yaml
├── notebooks/, tests/ (incl. tests/data/golden/ — small fixed dataset for regression tests)
├── docs/              PROJECT_PLAN.md, COMPETITIVE_ANALYSIS.md, ARCHITECTURE.md,
│                       ASSUMPTIONS.md, OPEN_QUESTIONS.md, DATA_DICTIONARY.md,
│                       MODEL_SELECTION.md, BIAS_AND_EXPLAINABILITY.md, MODEL_CARD.md,
│                       CASE_STUDY.md, adr/
├── reports/{figures,experiments,submissions}/, models/, docker/, .github/workflows/
├── mlruns/, mlflow.db   (live from Project Phase 0, see §8 — not deferred)
└── requirements.txt (pinned), requirements.lock.txt, README.md, docker-compose.yml
```

## 5. Data architecture and the leakage firewall

Unchanged in substance from v2.0, tightened: pandera contract organized by **temporal** (valid dates, known ordering, monthly frequency), **spatial** (fixed 15,715-cell grid), **target** (`target == next-month TWS_t`), **masking** (`TWS_t_masked ⟺ TWS_t is NaN`, never silently transformed).

| Forbidden | Allowed |
|---|---|
| Future TWS values | Test rows' own covariates |
| Target-derived test information | Historical observations, any depth |
| Future-released/reprocessed external covariates (vintage lock, §7) | Correctly-vintaged external data |
| Signatures/climatology using future rows (§2 — must be origin-time-indexed) | Historical spatial/temporal aggregates, origin-time-indexed |
| Public/private split exploitation | Model ensembles, OOF meta-features |
| Manual prediction editing | Config-driven, reproducible pipelines |
| Leaderboard-driven feature selection | Moderate public-LB sanity-checking |

`validation/leakage_tests.py` checks this mechanically: future-row shuffle test, signature/climatology historical-only check, rolling features stop exactly at the information cutoff, masking simulation can't reveal a hidden value.

**Environment capture, not just `requirements.txt`.** Zindi explicitly requires documented packages/environment for reviewed solutions. `requirements.txt` is pinned; `requirements.lock.txt` captures the exact resolved environment (Python version included) so "reconstructable from scratch" is a checkable claim, not an aspiration.

## 6. Masking architecture

Confirmed real structure (`COMPETITIVE_ANALYSIS.md` §3): blackout months mask ~99.7%+ of the grid simultaneously, with a handful of scattered, non-recurring exceptions — never a flat 66% row probability. Two distinctions worth making explicit rather than implicit in code:

**MaskingScenario, not a scalar rate.** `masking_simulator.py` takes a config object — `blackout_start, blackout_end, affected_locations, exception_rate, streak_length, transition_pattern, source/rationale` — not a bare `mask_rate=0.66`. This is what "streak-aware" concretely means as code, not just as a design phrase.

**Synthetic vs. historical blackout, never conflated.** *Historical* masking describes what actually happened to the real test months (used to characterize the problem, in Project Phase 1). *Synthetic* masking is artificially applied to historical periods where TWS is actually known, to create training/validation examples (Project Phase 2 onward). Mixing these up risks treating the test set's own mask pattern as if it were a reusable training feature, which it isn't — it's the thing we're being evaluated against.

## 7. Feature architecture refinements

**Spatial features get a declared information regime**, not one undifferentiated `spatial_history.py` grab-bag: **S1 concurrent** (same-month neighbor TWS — valid only when TWS is observed, i.e. rarely useful precisely when it would matter most); **S2 historical** (neighbor trajectories/trends/anomalies — the primary blackout-regime mechanism); **S3 signature** (basin/location static characteristics); **S4 residual** (post-model spatial residual correction). Every spatial feature declares which of these it is, preventing the exact bug class already caught once (a concurrent-neighbor feature silently being useless in the regime it was meant to help).

**External data gets a vintage lock**, extending the provenance table (`COMPETITIVE_ANALYSIS.md` §7-F): once a source/vintage is approved and pinned, it is never silently swapped for a newer/reprocessed version without a new ADR. Approval flow: scientific usefulness → temporal availability → revision risk → competition-rule compliance → operational accessibility → documentation complete → approved. Fails any step → rejected before it reaches feature generation.

**GRACE mission-calendar knowledge is DGP metadata by default, not a feature by default.** The real GRACE→GRACE-FO timeline (Project Phase 1) is used to *understand* the masking mechanism and design realistic blackout scenarios. Turning something like `months_since_GRACE_end` into an actual model feature is a separate decision requiring its own experiment and ADR — used carelessly, it risks encoding "which specific competition test month is this" rather than a genuinely generalizable relationship.

**Target strategy is a first-class experiment gate, not a baseline footnote.** Level (`TWS(t+1)`), delta (`TWS(t+1) − reconstructed_state`), anomaly, trend-residual, and volatility-normalized-delta all compete under identical folds before one is chosen — the "persistence + delta" baseline in Project Phase 3 is the floor this comparison has to beat, not the final word on target framing.

## 8. Validation engine — scenario registry, MoE gate contract, nested OOF discipline

**Scenario registry**, kept lightweight (config files, not a new subsystem): each validation scenario gets an ID and lives in `configs/validation/` — e.g. `V001` standard expanding window, `V002` 7-month synthetic blackout, `V003` 2015-like conditions, `V004` drought-regime episode. Experiments reference scenario IDs (`validation_scenarios: [V001, V003]`) instead of re-describing the split inline every time. One `validation_version` tag (bumped whenever `validation/` meaningfully changes) is enough at this project's scale — a separate framework-version/scenario-version hierarchy on top of that is more bookkeeping than the team size justifies right now.

**Test-matched CV's exact boundary, stated as a rule:** test-matched validation may use test *structure* (dates, covariates, missingness pattern) — never test *outcomes* (targets, inferred private-split membership, leaderboard feedback used to shape which historical episodes get chosen). This is the difference between legitimate diagnostic design and a soft form of indirect test-set tuning.

**MoE gate input contract:** the gate receives exactly the same information set the experts receive at prediction time — no privileged signal (e.g. the gate can't see true `TWS_t` when deciding how to weight the "missing-TWS expert" unless `TWS_t` is genuinely available in that regime). Stated explicitly because it's an easy, subtle way for a mixture-of-experts architecture to leak.

**Nested OOF discipline, made mechanical rather than just a stated principle:** ensemble meta-model training uses OOF predictions from base models never trained on that row, and is itself evaluated on a further held-out split — a meta-model fit and evaluated on the same OOF predictions produces deceptively strong CV. Residual correction follows the identical chain: base model → OOF prediction → OOF residual → residual model → OOF-corrected prediction, never training-set predictions feeding a residual model.

**Promotion requires a range, not a point estimate.** Candidate comparison reports `mean CV RMSE ± fold/seed variance`, not a bare number — a 0.001 improvement inside the noise band isn't a promotion case. Seed-stability checks (3-4 different reasonable seeds) apply before a candidate reaches champion status, not to every experiment along the way — a model that only wins with one specific seed isn't robust.

## 9. Uncertainty architecture

Three distinct sources, not one blended "confidence": **model uncertainty** (do specialists/ensemble members disagree?), **state uncertainty** (how reconstructed vs. observed is the current `StateSnapshot`?), **regime uncertainty** (how far outside the training distribution is this input — covariate shift territory?). Combined into a prediction confidence signal, but **only called "confidence" after validation**: bucket predictions into confidence deciles and check that observed RMSE actually rises monotonically as confidence falls. If it doesn't, it's reported as an "uncertainty proxy," not confidence — an honesty requirement that matters directly for the Trustworthiness report, not just internal QA.

## 10. Phase-by-phase architecture (updated from v2.0)

Full checklists in `PROJECT_PLAN.md`; changes from v2.0 noted inline, everything else unchanged.

**Project Phase 0.** *Change:* lightweight MLflow (SQLite backend) is live from the start, not deferred behind a flat-log-then-migrate step — experiment tracking is itself part of reproducibility, and Phase 1's findings shouldn't live outside the eventual lineage chain. The full Model Registry (staging/production promotion) still waits for Project Phase 11. *Also:* `requirements.lock.txt` alongside `requirements.txt`.

**Project Phase 1.** Unchanged — 7-experiment sequence, 2015 anomaly as a hard gate, stratified degradation curve. *Addition:* the degradation-curve result becomes a machine-readable table (not just a figure), since it directly parameterizes `MaskingScenario` configs in Project Phase 2.

**Project Phase 4.** *Change:* `state/reconstruction.py` implements the canonical `StateSnapshot` (§2) as one definition consumed everywhere, and `signatures.py` is origin-time-indexed, not just fold-level OOF (§2). Spatial features carry the S1-S4 declaration (§7).

**Project Phase 5.** *Addition:* resource/efficiency logged per model (training time, inference time, model size) from here on — directly serves the Innovation/Practicality and Sustainability rubric criteria, and is the actual evidence needed later to justify *not* using deep learning if GBM wins on both accuracy and cost.

**Project Phase 7.** *Addition:* MoE gate input contract (§8) enforced by a specific leakage test, not just a design note. Deep models live in `models/research/deep/` — the core pipeline has no PyTorch dependency unless a deep model actually clears the champion bar (§10 module map).

**Project Phase 9.** *Addition:* the three-part uncertainty architecture (§9) and its calibration check.

**Project Phase 13.** *Additions, described now, built when this phase starts:* a pinned `production_model_version` in config (the deployed app never silently loads "latest" — a competition experiment shouldn't be able to change the live portfolio app by accident); the UI surfaces model provenance (candidate ID, validation version, state regime/staleness, confidence); input guardrails (reject invalid coordinates, warn on out-of-distribution covariates, explicit degraded mode if state reconstruction fails) with a validated fallback hierarchy (full model → reduced-feature model → state-persistence → climatology); basic API hygiene (request validation, payload limits, timeouts, structured error handling, logging) — not because this is a high-risk app, but because "production-ready" should mean something concrete.

## 11. Submission & candidate subsystem

**Candidate vs. submission — a real distinction.** A candidate (one trained model/config combination) can generate several submissions over time (hyperparameter reruns, minor feature tweaks). `submissions/candidates.csv` tracks the candidate level (`candidate_id, model_family, model_version, feature_version, validation_version, config, mlflow_run_id, cv_overall, cv_masked, cv_unmasked, status, recommendation`); `submission_log.csv` gets one new column, `candidate_id`, linking each submission back to the candidate that produced it — this is the minimum structure needed for real traceability, not a separate heavyweight registry service.

**OOF predictions are a first-class artifact**, not a notebook variable — required for ensembling, residual correction, error-correlation-based final selection, and uncertainty, all of which silently produce wrong results if OOF predictions aren't reliably persisted and reused: `artifacts/oof/<candidate_id>/fold_NN.parquet`, columns `location_id, origin_time, target, prediction, regime`.

**Submission decision, not just a logged number.** Every `submission_log.csv` row gets a `decision` field (`retain / discard / investigate`) and a `submission_type` (`SANITY / ARCHITECTURAL / MODEL / FEATURE / ENSEMBLE / FINAL`) — the log should show *what we learned*, not just accumulate 100+ rows nobody revisits. Before spending a submission slot on a candidate that looks structurally very similar to one already tried, check OOF prediction correlation against recent submissions first — a soft guideline enforced by review, not automated blocking (the automation isn't worth building at this budget: 200 submissions, not 200,000).

**Prediction manifest per submission** (`submissions/manifests/<id>.json`) — mostly a serialization of the log row plus a schema hash, so each submission file has a self-contained reproducibility record sitting next to it, directly answering Zindi's "document data, outputs, features, package versions, environment" requirement at the individual-submission level, not just project-wide.

**Final selection, near competition close**, formalized as: for each shortlisted candidate, report CV (with variance), public LB, masked/unmasked/regime-specific decomposition, OOF error correlation against other shortlisted candidates, and a risk note (does it depend on external data or a more complex architecture with more that could go wrong under distribution shift). Choose two genuinely different candidates by error correlation, not the two best CV numbers.

## 12. Definition of "champion," and champion/challenger framing

**Not** the model with the best public leaderboard score. **The champion is the reproducible model or ensemble with the strongest expected private-leaderboard performance under the most realistic validation architecture, subject to leakage, robustness, reproducibility, and competition-rule constraints.** The public leaderboard is evidence toward that judgment, not the authority. The current champion holds its position until a challenger clears the same bar (validation, masked-regime requirement, reproducibility, robustness, complexity justified by evidence) — a useful framing for a multi-week competition where many candidates will be tried.

Champion is also not automatically the *deployed* model — a "research release" (scientifically validated), a "competition release" (submission-ready), and a "production release" (deployment-ready, passed Project Phase 13's eligibility checks) are allowed to be different points in time, even different candidates, without that being treated as inconsistency.

## 13. Assumptions register and open-questions register — new, distinct from ADRs

An ADR records a *decision already made*. These two registers record things we currently *believe* (and act on) or currently *don't know* — both are honest inputs to the Trustworthiness report, and both are exactly the kind of thing that quietly disappears into notebooks if not written down deliberately.

`docs/ASSUMPTIONS.md` — each entry: statement, confidence, supporting evidence, status (active/validated/rejected), and what experiment would test it. Seeded now with the assumptions this blueprint already depends on (test masking represents the real GRACE gap; external reanalysis data would be genuinely available at prediction time if sourced; historical spatial relationships are stable enough to be useful; the 2015 anomaly isn't evidence of a fundamental regime break — currently unresolved). If an assumption is later rejected by evidence, that's an explicit trigger for a new ADR, not a silent pivot.

`docs/OPEN_QUESTIONS.md` — questions we expect to remain unresolved for a while: why exactly 2015 degrades, how much of the masked regime is truly predictable from environmental forcing alone, how stable spatial relationships are across the full record, whether external data improves genuine out-of-time performance, whether the public/private split is uniformly random. Each entry: why it matters, current evidence, planned experiment, status.

## 14. What's explicitly out of scope

Full list in `COMPETITIVE_ANALYSIS.md` §12 (anti-goals). Adding two items surfaced this round: **manual prediction editing** (any hand-adjustment of a generated submission file, for any reason, breaks the reproducibility chain) and **leaderboard-driven feature selection** (choosing features because they moved the public score, rather than because they moved honest CV) — both already implied by existing rules but worth stating outright.

## 15. Reproducibility, four distinct kinds

Worth distinguishing since they have different audiences and different failure modes: **scientific** (can someone reproduce the *conclusion* — e.g. "masked-regime state reconstruction beats naive imputation"?), **computational** (can someone reproduce the *exact prediction*, bit-for-bit, given the same seed?), **software** (can someone reinstall the environment and run the system at all?), **competition** (can Zindi specifically re-run our submitted top-10 code and reproduce our leaderboard score?). The environment lock (§5), seed centralization (already planned), and the submission manifest (§11) exist mainly to satisfy the last two; the ADR/assumptions/experiment-record system (§13, `PROJECT_PLAN.md`) exists mainly to satisfy the first.

## 16. The governing rule

For any final prediction: (1) where did the data come from — data lineage; (2) how was the state reconstructed — feature/state lineage; (3) why was this model selected — experiment and validation lineage; (4) why should we trust it — validation, explainability, robustness, provenance; (5) can someone reproduce it — git + config + data version + MLflow + a deterministic pipeline. If we can't answer all five, the system isn't finished.

## 17. Final architectural statement

A reproducible, regime-aware global water-storage forecasting system, not a competition notebook: reconstruct hydrological state → characterize its temporal and spatial behavior → simulate the real observation/masking process → validate under realistic conditions, honestly → model state evolution with progressively stronger, evidence-gated experts → combine complementary models through disciplined OOF methods → quantify bias and uncertainty → register a reproducible champion → generate traceable submissions → expose the validated model through an API and an interactive application. The core design decision was never a specific model — it's the state-reconstruction-plus-process-aware-validation architecture. Everything else evolves around that core when evidence justifies it, and every time it does, an ADR says so.

## 18. Scope discipline — read this before proposing v4.0

This is the third major revision of this document and the seventh major planning artifact across this project (two prior `COMPETITIVE_ANALYSIS.md` merges, one `PROJECT_PLAN.md` restructuring, three `ARCHITECTURE.md` revisions). Everything in it is now defensible and most of it is genuinely cheap to build. But none of it has been tested against real code or a real experiment yet — the 2015 anomaly is still unresolved and the blackout-degradation curve, priority #1 since two revisions ago, still hasn't been run. Per §0's own governance rule, the next legitimate reason to change this document is an ADR triggered by an actual finding — not another architectural review. **The next action is code, not another document.**

---

**Next immediate step (unchanged, now overdue): Project Phase 1, experiment 2 — resolve the 2015 persistence-RMSE anomaly — then experiment 3, the stratified blackout-degradation curve.**
