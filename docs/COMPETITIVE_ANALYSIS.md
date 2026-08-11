# Competitive Analysis — "A Step Ahead of Drought" (Zindi/ITU)

Snapshot date: 2026-08-09. Fifth revision. This version incorporates a critique focused on statistical precision, architectural clarity, and separating what's verified from what's still hypothesis. Where a claim was checkable, we checked it before adopting it. Terminology note: **"Phase 1/Phase 2" below always refers to *Competition* Phase 1 (leaderboard RMSE) and Competition Phase 2 (trustworthiness/innovation report)** unless explicitly written as "Project Phase," to avoid collision with `PROJECT_PLAN.md`'s own phase numbering.

---

## 1. Competition facts (verified against the live Zindi page)

- "A Step Ahead of Drought: Forecasting Global Water Storage Challenge" (ITU / AI for Good). Prize pool €2,000: 1st €1,000, 2nd €600, 3rd €400, plus Zindi points.
- Opens 09 Jul 26, closes 13 Sep 26, private leaderboard revealed same day.
- Scoring: Competition Phase 1 (50%) = leaderboard RMSE. Competition Phase 2 (50%) = report on AI Trustworthiness (30%) + Innovation & Practicality (20%), top 10 only.
- **Public leaderboard ≈ 30% of the 280,961-row test set (~84,288 rows); private = remaining 70%, revealed only at close.** The number on screen today is not the number that decides the prize.
- Rules: open-source only, no AutoML, pretrained models OK if openly available, external data allowed only if "freely and operationally available... within one month of acquisition" and fully documented. Seeds must be fixed — Zindi re-runs top-10 code and adjusts rank to whatever it actually reproduces. Top 10 get a 72-hour code/model/report request at close; top 3 must make their solution public. Zindi reserves the right to disqualify "practices that compromise the inherent value of your solution." **You must select 2 submissions for private-leaderboard judging before close** (best-2-public used by default).
- 793 joined, 247 active, max 200 submissions (5/day).
- Ranks 1-8 are from your screenshot; ranks 10/50 were supplied without a cited source in an earlier round and remain unverified.

## 2. Public leaderboard snapshot and corrected math

| Rank | User | Public RMSE | Gap vs MOHAR |
|---|---|---|---|
| 1 | MOHAR | 0.55958798 | — |
| 2 | Shankar | 0.58930553 | +0.02972 |
| 3 | GIrum | 0.62338361 | +0.06380 |
| 4 | CalebEmelike | 0.63192768 | +0.07234 |
| 5 | awxlong | 0.65338823 | +0.09380 |
| 10* | lewisdzuda | 0.69241382 | +0.13283 |
| 50* | ezeshedrack | 0.73990026 | +0.18031 |

\* unverified. MOHAR is 5.04% below #2, 19.2% below #10's own score, 24.4% below #50 — all recomputed and confirmed. Test.csv is confirmed sorted chronologically; no evidence this affects the public/private split.

## 3. Empirical reference numbers and verified findings

In-sample numbers on 2002-2015 training data — an order-of-magnitude anchor, not a like-for-like comparison to test (Test.csv has no visible target).

| Approach | RMSE | Note |
|---|---|---|
| Global mean predictor | 0.912 | = target std |
| Per-location, per-calendar-month climatology | 0.817 | Weaker than intuition suggests |
| Naive persistence (target = TWS_t) | 0.572 | Strong; TWS is highly autocorrelated month-to-month |
| Linear reconstruction of TWS_t from concurrent covariates | 0.806 (reconstruction), 0.821 (as target predictor) | Simple linear imputation does not rescue the masked-row case |
| Linear TWS_t + SPEI_12 | 0.540 | Modest ~5.6% gain over pure persistence when TWS_t is known |

**Verified findings:**

**The target is not a per-location standardized anomaly.** Per-location means vary meaningfully (std of per-location means = 0.393) and per-location standard deviations genuinely differ (std of per-location stds = 0.167, range ~0.6-1.0). Real cross-location structure exists in both level and spread, but combined with climatology's weak standalone performance (0.817), most of the error comes from **within-location month-to-month deviation, not from misjudging a location's baseline.**

**Masking in blackout months is not a fixed set of "reference stations."** The 4-65 unmasked rows per blackout month don't recur across months in any systematic way — genuinely sporadic, scattered partial recovery, not systematic calibration sites — and zero locations are unmasked in every blackout month (verified exhaustively, all 66 pairs of the 12 blackout months, `notebooks/02_forecastability.ipynb` §3, Project Phase 1 Experiment 1). **Correction from an earlier partial check:** the true exhaustive overlap range is 0-29 (mean 1.53), not "0-2" as an earlier sample-based check suggested — the outlier overlap of 29 is specifically between the same calendar month a year apart (Feb 2016 vs Feb 2017), a weak same-month echo for a couple of months rather than fixed stations, and not strong enough to change the "don't build a feature assuming any location is reliably always-available" conclusion.

**Nearest-neighbor spatial correlation of TWS_t within a month is 0.981** — very high, but close to unusable in the regime we care about: blackout months are ~99.7%+ masked across the *entire grid simultaneously*, so a masked cell's same-month neighbors are almost always masked too. Since the whole grid loses observation together (not scattered per-cell), a neighbor's last-known value is typically anchored to the same prior month as the cell's own last-known value, so it mostly adds noise-averaging, not new temporal information. This is the basis for a hard rule in §7-C below: build **historical** spatial features, not concurrent-month ones.

**Feature correlation with the delta (target − TWS_t) collapses relative to the level.** TWS_t goes from r=0.80 (level) to r=**−0.32** (delta, i.e. mostly mean-reversion); best SPEI feature on delta is only −0.065. This caps how much "persistence + correction" can achieve on the *unmasked* third of test (best simple model there: 0.540). This is directional evidence, not proof, of where leaderboard separation comes from — see the precise decomposition in §5.

**Persistence RMSE by year is stable (0.50-0.63) from 2002-2014, then jumps to 0.898 in 2015** — unexplained. Could be a partial-year artifact (train only has Jan-Aug 2015) or a genuine late-period data-quality shift (documented GRACE battery/sensor degradation worsened toward end-of-mission). **Treated as a hard gate in `PROJECT_PLAN.md` Project Phase 1** — we don't finalize trend/recency-weighting/climatology feature design until this is resolved, because if 2015 reflects a real regime change rather than noise, it changes how much we should trust 2002-2014 patterns for the Sep 2015-Dec 2018 test data — note the test period's first month (September 2015) is the month immediately following the last month of training data (August 2015), which sharpens why this gate matters.

## 4. Central hypothesis: state reconstruction, not one-step regression

The most useful conceptual shift across all rounds of analysis: stop framing this as "predict next month's TWS from this month's features," and reframe as **"reconstruct the hydrological state of each known location, then forecast its evolution."** Every test location already exists in train (same 15,715 cells) — temporal extrapolation at known locations, not spatial extrapolation to new ones. This explains nearly everything found so far: persistence is unusually strong, the delta is much harder to predict than the level, 66% of test rows lack current TWS, masking arrives in contiguous blocks (not scattered), location history matters, staleness matters, same-month spatial neighbors go dark together, and external forcing data should matter *more*, not less, exactly when TWS_t is missing (§7-F).

**Important scientific caveat:** "latent hydrological state" is a modeling hypothesis, not something we directly observe. What we actually observe is TWS, SPEI, soil moisture, and the mask indicator; "state" is our inferred construct built from those. Worth keeping this distinction explicit in any report language — it's more rigorous and it's also just accurate.

```
                    Raw observations (TWS, SPEI, soil moisture, mask)
                                     │
                                     ▼
                     Observation-state reconstruction layer
        (last-known value, observation age, state velocity/
         acceleration, ACF, historical location signature)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                             │
        ▼                            ▼                             ▼
   Temporal-history            Environmental forcing         Spatial-history
   reconstruction              (SPEI, soil moisture,         reconstruction
   (lags, trend, ACF)          external hydrology)           (historical neighbor
        │                            │                       state, not concurrent)
        └────────────────────────────┼─────────────────────────────┘
                                     ▼
                       Specialist predictors (persistence/
                       state-transition, environmental,
                       historical-location, spatial-history, analog)
                                     │
                                     ▼
                  Conditional gate / OOF-optimized blend
              (built only after specialists are measured — §7-G)
                                     │
                                     ▼
                     Residual / bias / physics-informed correction (OOF only)
                                     │
                                     ▼
                                TWS(t+1)
```

## 5. How much of the leaderboard gap is actually the masked regime? (precision fix)

An earlier draft stated "the leaderboard is very likely won almost entirely on the masked two-thirds." That's directionally right but imprecise — RMSE combines nonlinearly, not by simple weighted averaging of the raw values. The correct relationship:

RMSE_overall² = p_m · RMSE_m² + p_u · RMSE_u², where p_m ≈ 0.66, p_u ≈ 0.34.

Worked example: if masked RMSE = 0.57 and unmasked RMSE = 0.54, overall RMSE ≈ √(0.66×0.57² + 0.34×0.54²) ≈ **0.560** (verified by direct computation). Because of the squaring, a 0.03 improvement in the masked regime can move the overall score by a similar or larger amount than an equivalent improvement in the unmasked regime — but the unmasked regime isn't automatically irrelevant either. **Revised statement:** because the masked regime is ~66% of test rows and has a substantially higher baseline error, most leaderboard separation is *expected* to come from masked-regime performance — but this should be measured via the RMSE decomposition above for every model we build, not assumed. This decomposition becomes a standard row in the error-decomposition table (§10).

## 6. Internal performance targets

| Internal CV target | Meaning |
|---|---|
| < 0.572 | Beat naive persistence on a realistic (masked-aware) validation split |
| < 0.559 | Beat MOHAR's current public score, under our own honest validation |
| < 0.53 | Serious first-place contender |
| < 0.50 | Exceptional — investigate why before trusting it |

**These are not universal pass/fail thresholds on the overall number alone.** A model can hit a low overall RMSE while being dangerously dependent on the easy (unmasked) regime — e.g. overall 0.520 with masked 0.57 / unmasked 0.40 looks better in aggregate than overall 0.525 with masked 0.55 / unmasked 0.47, but the second is more robust to whatever the true private-set masked/unmasked ratio turns out to be. **Promotion rule: a model is only promoted up the champion ladder (§9) when it improves the relevant CV tier without materially degrading another important regime** — judged from the full error-decomposition table (§10), not the headline number.

## 7. Master hypothesis catalogue

### A. Temporal / persistence modeling — confidence: very high, impact: high
Persistence + learned delta correction; TWS lags/rolling stats/trend slope; per-location ACF at lag 1/3/6/12; optimal-lag search across SPEI timescales. Distributed-lag "memory kernel" features are lower priority specifically for GBMs, which approximate weighted lag combinations natively via splits.

### B. Missing-TWS (masked regime) handling, incl. the state-reconstruction layer — confidence: high, impact: very high, top-priority category
The masking process is not "missing values to route around" — it's a distinct forecasting regime requiring its own state representation. Concretely, distinguish four different things that could all loosely be called "a lag feature," because under real gap structure they are not interchangeable:
- **Calendar lag** — TWS at exactly t−k, which is itself frequently missing inside a blackout streak.
- **Last-observed lag** — TWS at the most recent month it was actually observed, however many months back that is.
- **Observation age** — t − t_last_observed (the staleness itself).
- **Observation trajectory** — the sequence of the last 2-3 *actually observed* values (last_known, previous_known, second_previous_known), from which state velocity (last_known − previous_known) and acceleration can be derived.
This full set — plus observation_count/density over trailing 12/24 months — is what we mean by the "state reconstruction layer" in §4's diagram, and it should be built as one coherent pipeline stage rather than scattered ad hoc features, so every downstream model consumes the same, consistently-defined state representation. Two-regime or gated modeling, and streak-shaped masking-augmentation during training, remain part of this category.

### C. Spatial information — confidence: medium-high, impact: high, hard rule below
Confirmed in §3: same-month neighbor correlation is very high (0.981) but functionally unusable during blackout months, since the whole grid goes dark together. **Hard rule: build historical spatial features, not concurrent-month ones.** Bad version: `neighbor_TWS_t` (almost always NaN exactly when it would matter). Better version: `neighbor_TWS_last_known`, `neighbor_TWS_lag_3/6`, `neighbor_historical_anomaly`, `neighbor_trend`, `neighbor_seasonal_signature`, `neighbor_ACF`. Basin-aware aggregation (vs. raw k-NN) is a reasonable upgrade if a basin dataset is sourced. Full/global kriging remains computationally impractical (O(n³) at n=15,715); a GBM-then-spatial-residual-model two-stage pipeline is the realistic version of that idea.

### D. Feature engineering on given variables — confidence: high, impact: medium-high
SPEI differencing and drought-persistence run-length features; soil-moisture trajectory; month × hemisphere interaction. Hand-crafted multiplicative interactions remain lower priority specifically for GBMs.

### E. Target transformation — confidence: high, impact: potentially large, cheap to test
Delta; anomaly (target − location/month climatology — a target-transform, distinct from using climatology as a *standalone predictor*, which we measured weak); trend-residual; volatility-normalized delta (ΔTWS / location's own historical ΔTWS std). Test all in one controlled comparison; per §5, expect the biggest gains on the masked regime, not the unmasked one.

### F. External data — confidence: medium, impact: potentially large, rules-sanctioned
**Sharper hypothesis than "add ERA5":** when TWS observations disappear, state uncertainty increases, and independent hydrological forcing variables (precipitation, ET, runoff, snow) become *disproportionately* valuable specifically because they're the best remaining source of information about state transition — i.e. external data's value should be concentrated in the masked regime, which is directly testable (compare masked-regime RMSE with/without external features). Water-balance composites (P, ET, P−ET at 1/3/6/12-month windows) align with why SPEI_12 already beats SPEI_01. **Provenance and vintage gate**, expanded with a revision-history column — reanalysis products are frequently revised after initial release, and using a later-reprocessed value that wasn't actually available in near-real-time form would be a real (if subtle) leakage risk:

| Feature | Source | Temporal resolution | Release latency | Revision history | Available at forecast time? | Leakage risk |
|---|---|---|---|---|---|---|
| e.g. precipitation | (to fill in) | (to fill in) | (to fill in) | (to fill in) | Yes/No | Low/Med/High |

This table is not bureaucracy for its own sake — it directly produces the documentation the Competition Phase 2 Trustworthiness/reproducibility rubric asks for.

### G. Modeling architecture and ensembling — confidence: high, impact: high
GBM (LightGBM/XGBoost/CatBoost) as primary; deep spatiotemporal models (GNN/ConvLSTM/Transformer) explicitly optional and deprioritized (§11). **Build named specialists before any gating network**, not a mixture-of-experts architecture from day one: a persistence/state-transition expert, an environmental-forcing expert, a historical-location expert, a spatial-history expert, and (later, lower priority) an analog-forecasting expert. Measure each specialist's error by regime *first*. Only build a learned gate once we can show "expert A wins under condition X, expert B wins under condition Y" — otherwise we risk building an architecture whose main achievement is looking sophisticated rather than lowering RMSE, and the competition rewards the latter only. Historical location "signature" meta-features (mean, std, trend, seasonality amplitude, ACF, SPEI/soil-moisture response) computed out-of-fold remain the correctly-regularized alternative to fitting 15,715 separate per-location models — and specifically need **shrinkage**, since some locations have noisier or sparser historical estimates than others: θ̂_location = w·θ_location + (1−w)·θ_global, with w increasing with the amount/quality of location-level evidence (e.g. w = n/(n+k) for some prior-strength constant k). This is a standard empirical-Bayes/hierarchical approach and should be implemented explicitly, not approximated by just handing a GBM a raw location ID.

### H. Post-hoc correction — confidence: medium, impact: medium, all conditional on strict OOF discipline
Residual modeling (second model on OOF residuals, using location/season/SPEI/months-since-TWS/prediction as inputs); systematic bias correction by location/region/season (doubles as direct Competition Phase 2 Trustworthiness evidence, since the rubric explicitly asks about bias); physics-informed residual blending (blend ML delta with a physically-estimated ΔTWS ≈ P−ET at an OOF-learned weight, if external water-balance data is sourced — a genuinely strong Innovation narrative). Temporal/spatial "prediction reconciliation" (smoothing implausible jumps) needs testing, not assuming — RMSE doesn't automatically reward smoothing and can punish it if real extremes get washed out.

### I. Uncertainty and confidence signals — confidence: medium, impact: mainly Competition Phase 2, some RMSE upside
Not currently a leaderboard requirement (point predictions only), but worth building regardless: ensemble variance across specialists, residual distribution conditioned on staleness/regime, and a **degradation slope** (ΔRMSE / Δmonths-since-observation, see the enriched blackout-curve diagnostic in `PROJECT_PLAN.md` Project Phase 1) as a fourth tracked metric alongside RMSE. A model whose confidence visibly narrows for fresh observations and widens for stale ones ("the model knows when its state estimate is degrading") is a compelling, concrete Trustworthiness finding, independent of whether uncertainty is ever submitted to Zindi.

### J. Validation and non-stationarity — confidence: very high, impact: decisive
Time-respecting expanding-window CV. Streak-shaped masking simulation (contiguous per-location blocks, not row-independent) matching the real structure confirmed in §3. **Three validation tiers, framed as three different questions, not just three sets of folds:** Tier 1 (forecastability) — can we predict a future month under normal observation conditions? Tier 2 (blackout) — can we forecast after losing the current TWS observation, and how does accuracy degrade with staleness? Tier 3 (test-regime) — can we reproduce the exact observation/masking/environmental conditions the actual test months represent? Each tier reports the full error-decomposition table (§10), not one aggregate number. **Two integrity safeguards, stated as explicit rules:** (1) Tier-3 (test-matched) validation may be used for diagnosis and robustness assessment, but final model selection must be based on predefined historical folds, not repeatedly tuned against test-specific analogs — the difference between "build validation episodes that reproduce test's known structural characteristics" (fine) and "find historical periods that happen to resemble the exact test rows and optimize against them" (a soft form of indirect test-set tuning). (2) Test row ordering is descriptive metadata only — no `test_row_index` or `relative_test_position` features, and no inference about the public/private split from row order.

### K. Ideas we'd recommend against or deprioritize
See the consolidated anti-goals list in §12 — kept there rather than duplicated here, since it now spans strategy, modeling, and process choices together.

## 8. Champion ladder

Each level must beat the previous one on the tier-appropriate CV (§7-J) before we add the next; if a level doesn't help, we drop it and move on. Revised to separate baseline types and feature families more precisely than earlier drafts, matching the state-reconstruction framing in §4:

0. Global mean → 1. Seasonal climatology → 2. Current-TWS (oracle) persistence → 3. Last-known-state persistence → 4. Persistence + learned delta → 5. Raw-feature GBM → 6. GBM + state-reconstruction layer (staleness, trajectory, velocity/acceleration) → 7. GBM + temporal-history features (lags, ACF, trend, recency weighting) → 8. GBM + historical-location signatures (shrunk) → 9. GBM + historical spatial-history features → 10. GBM + external hydrological data (provenance-gated) → 11. Named specialists → 12. Conditional/MoE blend (only if specialists show regime-conditional advantages) → 13. OOF-optimized ensemble → 14. OOF residual/bias correction → 15. Physics-informed blend (if external data supports it) → 16. Optional deep spatiotemporal model (last, explicitly optional — see §11).

Maps directly onto `PROJECT_PLAN.md` Project Phases 3-8; the ladder is the execution order within those phases, not a replacement for them.

## 9. Kill criteria

For every major research branch, define in advance what evidence makes us stop, so the project doesn't become an open-ended research rabbit hole:

- **Spatial-history branch:** stop if it doesn't improve blackout-tier CV by at least ~0.002 RMSE across multiple blackout folds.
- **External-data branch:** continue only if masked-regime RMSE improves, with no meaningful degradation elsewhere, and the feature clears the provenance/vintage gate (§7-F).
- **MoE/gating branch:** don't start until named specialists (§7-G) show measurable regime-conditional advantages; if they don't, a simple OOF-optimized blend is enough.
- **Deep-learning branch:** stop if the GBM-based champion remains better after one reasonable, reproducible experiment — don't iterate on architecture trying to catch up.
- **Analog-forecasting branch:** keep only as an ensemble member if it improves the blend; don't promote it to a primary model without clear evidence.

## 10. Error decomposition table (standing artifact for every champion)

Every model that reaches the champion ladder gets this table filled in — it's the project's central model-selection dashboard, not a one-off report:

| Model | Overall | Masked | Unmasked | 1-2mo stale | 3-4mo | 5+mo | N. Hemisphere | S. Hemisphere | Extreme TWS | Rapid-change |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle) | | | | | | | | | | |
| Last-known-state | | | | | | | | | | |
| GBM (raw) | | | | | | | | | | |
| GBM (state-aware) | | | | | | | | | | |
| Specialists / MoE | | | | | | | | | | |

## 11. Deep learning: explicitly optional, and that's a feature not a bug

A GRU/ConvLSTM/Transformer might look superficially more impressive in a portfolio, but if a carefully engineered GBM-plus-state-reconstruction pipeline beats it, the more senior-level story is **"we didn't reach for deep learning because the data-generating structure favored a well-engineered tabular state-space formulation, and we can show the comparison that proves it"** — that demonstrates judgment, not just familiarity with more tools. Deep learning stays on the champion ladder (level 16) as a genuine, honestly-tested comparison, not a rejected idea, but it's explicitly the last thing we try and the first thing we're willing to drop if it doesn't clear the bar.

## 12. Anti-goals — what we would deliberately not do

Reverse-engineering the public/private split via submission probing (distinct from using test rows' own covariates as features, which is normal and planned). Chasing public LB improvements without a matching internal validation improvement. Per-location climatology or linear imputation as standalone answers to the masked-row problem (directly contradicted by §3). Fully independent per-location or per-continent models as a first move (only fork into separate models if regime-as-a-feature proves insufficient, given ~150 observations per location). Building a Transformer or GNN before a GBM baseline exists. Training 15,715 individual per-location models. Spending days on GBM hyperparameter tuning before feature/state representation is settled — representation matters far more here than tuning. Random row-independent masking augmentation — it doesn't reproduce the real contiguous blackout process (§3, §7-B). Letting DVC/CI/CD/production engineering block or slow down the modeling track (see `PROJECT_PLAN.md` Project Phase 0's required-now vs. can-catch-up split).

## 13. Submission strategy

5/day, 200 total, and 2 submissions must be selected for private-leaderboard judging before close. Rough allocation: early submissions for validation sanity-checks, middle submissions for major architecture experiments (each champion-ladder level worth testing live), late submissions for fine-tuning. **The final two selected submissions should be chosen using error correlation, not just the two lowest CV RMSEs.** Two models with 0.520 and 0.525 CV RMSE but 0.99 error correlation offer almost no diversification benefit; two models with 0.520 and 0.528 but 0.75 error correlation could meaningfully hedge our risk on the private 70% we can't see. Compute corr(e_A, e_B) on OOF predictions for shortlisted candidates and factor it into the final choice alongside CV performance, regime performance (§10), and robustness to the historical training cutoff.

## 14. Three explicit objectives

(A) minimize public leaderboard RMSE, (B) minimize *expected* private leaderboard RMSE (the one that actually matters, estimated only via honest CV), (C) maximize the Competition Phase 2 rubric — trustworthiness, transparency, reproducibility, innovation, practicality, sustainability. Several items above (bias correction, the provenance table, physics-informed blending, uncertainty signals) score on both B and C simultaneously, worth favoring when choosing between otherwise-similar options.

## 15. Process notes

**Competition track vs. engineering track:** already how `PROJECT_PLAN.md` is sequenced — modeling phases come before production/deployment phases, and Project Phase 0 is split into required-now vs. can-catch-up so engineering polish doesn't compete with modeling insight for time.

**Why there's no separate "forensics phase":** folded into Project Phase 1/2 rather than added as a new phase number — it's exactly what those phases exist to do, now with a concrete, ordered experiment sequence (see `PROJECT_PLAN.md` Project Phase 1) rather than a general description.

**Rules and ethics boundary:** build a model that deserves to beat MOHAR, not a way to make the public leaderboard say we beat MOHAR. In bounds: rules-compliant external data, any level of spatial/temporal modeling sophistication, using test rows' own covariates as model inputs, ensembling, regime-specific models, moderate public-LB sanity-checking. Out of bounds: test targets/future values, undisclosed collaboration or multiple accounts, submission-probing to map the public/private split, anything that breaks the "must reproduce your own submitted code" requirement.

## 16. Top priorities, reconciled across all rounds

1. **Execute Project Phase 1's ordered experiment sequence** (see `PROJECT_PLAN.md`) — reproduce the masking process (done), resolve the 2015 anomaly as a hard gate, then the blackout-degradation curve stratified by staleness × latitude × season × location-ACF × drought regime (not just a single average curve), then last-known-state baseline, staleness×location-dynamics, covariate shift, and the real GRACE mission timeline.
2. **Missing-TWS regime handling via the formal state-reconstruction layer** (§7-B) — our own numbers show naive approaches fail here; likely where the leaderboard gap is decided, and now measurable precisely via §5's decomposition.
3. **Streak-aware, tier-based, integrity-safeguarded validation design** (§7-J) — determines whether we can trust any other number we produce.
4. **Historical location signature meta-features with explicit shrinkage** (§7-G) — the regularized way to exploit "same 15,715 locations in train and test."
5. **External Copernicus/reanalysis data**, gated by provenance *and* vintage (§7-F), tested specifically for masked-regime impact per the sharpened hypothesis in §7-F.
6. **Named specialists measured before any gating architecture** (§7-G) — avoid building MoE complexity the evidence hasn't earned yet.
