# Project Phase 1 — Findings Synthesis

**Scope:** everything established in `notebooks/01_eda.ipynb` (foundational EDA) and `notebooks/02_forecastability.ipynb` (the 7 ordered Project Phase 1 experiments), synthesized in one place with what each finding *means* for the rest of the project. Every number below is traceable to a specific notebook section and to `docs/ASSUMPTIONS.md` / `reports/experiments/experiment_log.csv`, not restated from memory.

**Why this document exists:** Phase 1's job was to understand exactly what we're being asked to forecast, and how hard each regime actually is, *before* writing any modeling code. That job is done. This is the handoff — the single place that answers "what do we now know, and what does it force us to do differently in Phase 2 onward."

---

## 1. Foundational EDA (`notebooks/01_eda.ipynb`)

Nine findings, all verified directly against the raw files (not assumed from the competition description):

1. **Target definition confirmed exactly**: `target[t] == TWS_t[t+1]` at the same location, max absolute difference 0.00e+00 across the whole file. Zero leakage ambiguity.
2. **Grid**: 15,715 unique locations, identical set in Train and Test, land-only, sharply Northern-Hemisphere-weighted (5,573 locations at 30–60°N vs. 570 at 30–60°S — a ~10:1 imbalance).
3. **Temporal coverage**: training spans 2002-05 to 2015-08 with 22 missing calendar months inside that span; test spans 2015-09 to 2018-12 but contains only 18 of the 40 calendar months in that window.
4. **Masking**: 66.5% of Test.csv rows have `TWS_t` withheld, with the `TWS_t_masked` indicator matching actual nullness with zero mismatches across all 280,961 rows.
5. **In-sample baselines**: global mean 0.912, seasonal climatology 0.817, naive persistence 0.572. Persistence is the number every model has to decisively beat.
6. **Non-stationarity**: mean target drifts from +0.226 (2002) to –0.136 (2012–2015 trough) — random K-fold CV would leak the future; this is why Phase 2 uses expanding-window splits.
7. **The 2015 persistence-RMSE anomaly**, first spotted here (0.898 vs. a 2002–2014 band of 0.506–0.627), was flagged as a hard gate and left unresolved for Experiment 2.
8. **Level-vs-delta correlation collapse**: `TWS_t` correlates r=0.803 with the *level* of next month's value but only r=–0.315 with the *delta* (mean-reversion); the best SPEI feature manages only r=0.065 on the delta. This is the single biggest reason the problem is hard — persistence captures the easy part (the level), and almost nothing in the given features explains the change.
9. **Nearest-neighbor spatial correlation** is very high same-month (r=0.982) — but flagged from the start as likely unusable during blackouts, since the whole grid goes dark together. Experiment 1 later confirmed this directly.

**What this established:** persistence is strong, the *change* is what's actually hard to predict, the grid is spatially and temporally uneven, and two specific loose ends (the 2015 anomaly, the real value of spatial correlation under blackout) needed dedicated experiments — which is exactly what Phase 1's 7 experiments were designed around.

---

## 2. The seven experiments — question, method, verdict

### Experiment 1 — Reproduce the masking process
**Question:** is masking a smooth gradient or a genuine on/off phenomenon, and does the pattern hint at a real external cause?
**Method:** per-month masking-rate breakdown; exhaustive pairwise overlap check across all 66 blackout-month pairs; per-month grid-completeness audit.
**Findings:** masking is strictly bimodal — 6 test months are 0% masked, 12 are 99.58–99.97% masked, nothing in between. Every single test month is short of the full 15,715-location grid *as rows*, not just masked (38–195 locations absent per month) — a finding that also turned out to hold for Train.csv (Experiment 3). Partial recovery during blackout months is scattered, not fixed reference stations: overlap across all 66 month-pairs ranges 0–29 (mean 1.53), with zero locations unmasked in every blackout month — this refined an earlier, looser "0–2 overlap" claim in `COMPETITIVE_ANALYSIS.md`. A single-search preview suggested the absent test months lined up with the real GRACE→GRACE-FO gap (later formally confirmed in Experiment 7).
**Verdict:** masking is a genuine two-state (fully-observed / blackout) regime switch, not a per-row probability — this is the direct justification for Phase 2's streak-aware masking simulator rather than a flat masking rate.

### Experiment 2 — Persistence ceiling and the 2015 anomaly (hard gate)
**Question:** is the 2015 persistence-RMSE spike (0.898) a data artifact or a real signal, and can 2002–2014 patterns be trusted for the test period that immediately follows 2015?
**Method:** ruled out partial-year/seasonal-composition artifacts directly (pooled Jan–Aug vs. Sep–Dec ratio only 1.03); checked whether the effect was outlier-driven (1%-trimmed RMSE barely moved) or regional (both hemispheres elevated similarly); sourced the 2015–16 El Niño event as a candidate physical cause.
**Findings:** the elevation is episodic within 2015 (Jan/Feb/Mar/Jun/Jul elevated 1.6–2.1×, Apr/May/Aug normal), broad-based (not a few outliers), global (not regional), and directional (systematic negative residual shift, i.e. real water-storage decline, not noise). The 2015–16 El Niño — comparable in strength to 1997–98, with documented drought/anomaly signatures in exactly the regions where TWS dropped — is a plausible physical explanation, though not yet verified month-by-month against an ENSO index.
**Verdict:** genuine regime characteristic, not an artifact. Don't discard or down-weight 2015 — but also don't assume 2002–2014's typical volatility is representative of the test period, since the test period's first month follows directly after this anomalous window.

### Experiment 3 — Blackout-degradation curve (highest-priority single experiment)
**Question:** how does last-known-value error grow with staleness, and does a single pooled curve hide meaningfully different subpopulations?
**Method:** simulated 15 overlapping contiguous blackout windows (K=9 months) on the verified gap-free 2004–2010 span, cross-checked against a strictly non-overlapping subset; stratified by per-location ACF(1), latitude band, season, and onset drought regime; compared against a theoretical AR(1) model.
**Findings:** pooled RMSE grows 0.537 (k=1) → 0.884 (k=9), a 1.65× increase. Per-location ACF(1) quartile is by far the dominant stratification factor (0.403 RMSE spread at k=9 — nearly 2× the next-largest factor), with a genuine nuance: low-ACF locations are worse in *absolute* terms at every horizon, while high-ACF locations degrade *proportionally* faster. Latitude band is next (0.191 spread, driven by a severe 0–30°S outlier), then season (0.079, weak) and onset drought regime (0.066, weak but directionally consistent with the EDA's mean-reversion finding). The AR(1) theoretical model gets the curve's *shape* right but systematically over-predicts absolute error — meaning real, exploitable structure exists beyond simple lag-1 persistence.
**Verdict:** the pooled curve does hide real subpopulations, exactly as suspected. This single experiment is the direct empirical justification for prioritizing ACF/historical-signature features above region/season/onset-state features in Phase 4.

### Experiment 4 — Last-known-state baseline (Baseline B)
**Question:** how would naive last-observation-carried-forward actually score on the *real* 18-month test structure, not a generic k-months-stale abstraction?
**Method:** first caught a subtle definitional bug — staleness must be measured to the *target* month (row+1), not the row's own month. Reconstructed the real test temporal structure (6 FULL + 12 BLACKOUT month offsets), then cross-validated two independent ways: Method A reweighted Experiment 3's curve by the real staleness distribution; Method B directly replayed the real 41-month FULL/BLACKOUT pattern onto 8 independent windows of clean 2004–2010 history, ground-truth scored.
**Findings:** the 12 real blackout months carry staleness-to-target k=2 through k=7 — not one fixed value (distribution: k=2×4, k=3×3, k=4×2, k=5/6/7×1 each). Methods A (RMSE≈0.7091) and B (RMSE=0.7145) agreed within 0.005 pooled and 0.028 mean-absolute per individual month. Full four-baseline picture on the real/replayed structure: **A** (oracle persistence, FULL months only) 0.5247; **B** (last-known, BLACKOUT months only) 0.7145; **C** (seasonal climatology) 0.8170; **D** (Hybrid, realistic naive floor) 0.6573.
**Verdict:** the in-sample persistence figure (0.572) that `COMPETITIVE_ANALYSIS.md`'s target ladder was originally calibrated against is *not* reachable on the real test set, because 12 of 18 test months have no current observation at all. **Baseline D (0.6573) is the real "do nothing clever" floor** — this single number reframes what "beating persistence" should mean for the rest of the project.

### Experiment 5 — Staleness × location-dynamics interaction
**Question:** does per-location ACF explain the *shape* of Experiment 3's curve — i.e. is staleness alone the right state variable, or does it need to interact with location persistence?
**Method:** four independent angles — a direct linear interaction regression (`error² ~ k + acf1 + k·acf1`), the AR(1) theoretical model's explanatory power, a finer 10-way ACF-decile stratification, and a confound check against TWS volatility (σ).
**Findings:** the linear interaction term added essentially zero R² (0.0645 → 0.0645) — which looks like a null result. But the AR(1) model (parameter-free, driven only by each quartile's ρ and σ) explains R²=0.448 of the curve's variance, and the decile stratification is monotonic at 9 of 9 tested values of k. The resolution: k and ACF genuinely interact, but through a nonlinear form (ρᵏ, not k·ρ) — a linear test was simply the wrong tool to detect it. Separately, TWS volatility (σ) turned out to be a largely independent signal from ACF (r=–0.141) that adds substantially *more* explanatory power (+0.0462 R², +71.6% relative) than the ACF interaction did.
**Verdict:** partially — real, but not linear. This directly justifies keeping `acf_1_3_6_12` and `location_signature` (mean/std) as distinct `StateSnapshot` fields rather than collapsing them, and recommends testing an explicit AR(1)-motivated composite feature in Phase 4, not just a raw multiplicative term.

### Experiment 6 — Covariate shift
**Question:** do train and test look meaningfully different on SPEI, soil moisture, calendar month, and masking rate — a generalization risk worth knowing about, and indirect evidence for or against A-003 (spatial-relationship stability)?
**Method:** KS statistics and histogram comparisons, train vs. all of test; a second three-way split (train vs. test-unmasked vs. test-masked) to check whether the blackout regime is *environmentally*, not just *observationally*, distinct; calendar-month coverage comparison.
**Findings:** masking rate (0% vs. 66.5%) is by far the largest covariate-regime difference, but that's the project's existing central design focus, not a new risk. SPEI/soil-moisture shift is minimal both train-vs-test (largest KS statistic 0.0352) and within test itself, masked-vs-unmasked (largest KS 0.0569) — the blackout regime is not environmentally distinct from the observed regime, consistent with masking being a hardware/mission-timeline cause rather than an environmental one. The real actionable finding: **test entirely omits October**, and calendar months that happen to recur twice within the 18 test months get roughly double the row-share of months appearing once.
**Verdict:** covariate shift on SPEI/soil moisture is not the dominant generalization risk here. The calendar-month coverage gap is a real, secondary risk that Phase 2's fold design and Phase 4's seasonal features both need to account for explicitly.

### Experiment 7 — Real GRACE/GRACE-FO mission timeline
**Question:** does the observed masking pattern actually match the real, documented satellite mission history, not just an internally-consistent pattern inferred from 18 sparse test months?
**Method:** dedicated sourced research (JPL's GRACE Tellus site, JPL's mission pages, and a peer-reviewed paper — Landerer et al. 2020, *Geophysical Research Letters*) — not a repeat of Experiment 1's single-search preview. Three cross-checks against the real data: the test set's absent-month structure, the missing training months' pattern, and the severely-masked (not absent) test-month cluster.
**Findings:** the single longest absent run in the test period (2017-07 to 2018-06, 12 months) matches the documented 11-month GRACE-to-GRACE-FO gap *exactly*, plus one extra month — GRACE-FO's own first, commissioning-adjacent data month, plausibly excluded by the competition's creators. Bonus resolution: 17 of the 22 missing *training* months (from 2011 onward, in 8 distinct events averaging 2/year) match the documented "battery management" outage cadence (~every 6 months since 2011) — a gap A-001 had flagged as unexplained. Honestly-flagged open item: 4 smaller absent runs also exist scattered through the test period; only one (coinciding with GRACE-2's October 2016 accelerometer shutdown) has a plausible sourced explanation, the other 3 don't and may simply be the competition's own month-selection choices.
**Verdict:** A-001 formally validated. The masking mechanism is the real satellite mission history, not an artificial construct — this grounds any "months since the gap began" feature in the actual mission calendar rather than an inferred pattern, and is directly citable evidence for the Competition Phase 2 Trustworthiness report.

---

## 3. Cross-cutting themes (what the seven experiments say together)

**The masking regime is the problem, and it's now fully characterized.** Not gradual, not random, not artificial — a real, two-state, mission-history-grounded phenomenon with a precisely measured staleness distribution (k=2–7, non-uniform) on the actual test set. Every subsequent phase can now build against real numbers instead of assumptions.

**Persistence's apparent strength was partly an illusion of measuring it in-sample.** 0.572 (Baseline A) is a ceiling reachable on only a third of the real test months. 0.6573 (Baseline D) is the real floor. This single correction changes what "beating the baseline" means everywhere downstream, including the target ladder in `COMPETITIVE_ANALYSIS.md` and the promotion rule in Phase 2.

**Staleness is necessary but not sufficient as a state variable.** It has to be interpreted through the lens of each location's own dynamics (ACF, volatility) — and that relationship is nonlinear, which is itself a finding worth remembering: a null result from one specific statistical test (the linear interaction) was nearly a false negative that a mechanistic model and a nonparametric check both overturned.

**2015 is a real, not-dismissible regime, and it sits right before the test period starts.** Plausibly El Niño-linked, but not fully pinned down. This is a standing caution against assuming the training data's typical behavior — not just its mean, but its volatility — carries over into the test window.

**The dominant generalization risk is structural (masking), not distributional (covariates).** SPEI and soil moisture look similar across train/test; the real risks are the masking-regime shift (by design, already the project's central architecture) and the calendar-month coverage gap (a real, previously-unquantified risk).

**Every finding that looked like it might be a coincidence was checked, not assumed.** The GRACE gap match, the battery-management cadence, the ACF/AR(1) relationship — each was cross-validated by at least two independent methods before being written into `docs/ASSUMPTIONS.md` as Validated. Two items were deliberately left open (pre-2011 missing months, 3 unexplained scattered test-period gaps) rather than force-fit into a tidy narrative.

---

## 4. Implications for Project Phase 2 (validation harness) — starting immediately next

This is documented in full detail in `docs/PROJECT_PLAN.md` Phase 2 and `docs/adr/0004-phase1-closure-phase2-validation-design.md`, but the short version:

- The **streak-aware masking simulator** must draw blackout lengths from the real k=2–7 distribution found in Experiment 4, not an invented geometric/uniform prior.
- **Tier 3 (test-regime) validation** should replay the real 6-FULL/12-BLACKOUT calendar structure from Experiments 1/4/7, respecting the October gap and 2×/1× recurrence pattern from Experiment 6.
- The **promotion-rule target ladder** must be recalibrated: clearing 0.6573 (Baseline D) is the real "beats naive" bar, not 0.572.
- The **error decomposition table** should bucket by the real k=2–7 staleness values, cross-cut by ACF quartile (Experiment 5's finding that staleness alone under-describes the regime).
- **Expanding-window CV folds** must not be constructed in a way that hides the October gap or averages away the 2015 anomaly (Experiment 6/Experiment 2's findings respectively).
- The harness's own sanity check: a trivial Hybrid-style model run through Tier 3 should reproduce something close to 0.6573 — if it doesn't, the harness isn't faithfully reproducing Phase 1's measured reality.

## 5. Implications for later phases

**Phase 3 (state-aware baselines):** Baselines A/B/C/D are already computed as a strong preview (0.5247 / 0.7145 / 0.8170 / 0.6573) — Phase 3's job is to recompute them formally inside the Phase 2 harness for the official record, not to discover them from scratch.

**Phase 4 (state reconstruction & features):** the priority order is now evidence-based, not intuited — per-location ACF/historical-signature features first (Experiment 3's dominant stratification factor, Experiment 5's confirmed nonlinear interaction with staleness), region/latitude-aware features second, calendar/seasonal features explicitly validated against the October gap and recurrence imbalance (Experiment 6), and an explicit AR(1)-motivated composite feature (`sigma*sqrt(2*(1-rho**k))`) worth testing alongside letting the GBM learn the interaction natively (Experiment 5).

**Phase 6 (external data):** Landerer et al.'s finding of no intermission bias between GRACE and GRACE-FO (Experiment 7) removes one concern for treating pre/post-gap TWS as one continuous series. The battery-management cadence is itself a plausible external-data-free feature — a "known mission data-availability risk" indicator derivable purely from the public mission calendar.

**Phase 9 (trustworthiness/explainability):** Experiment 7's sourced mission-history cross-check, and the deliberately-left-open items (A-012), are directly citable evidence for the Competition Phase 2 report's Trustworthiness section — this project can show its validation work, not just assert it.

## 6. Full assumptions register (post-Phase-1 status)

| ID | Statement (short) | Status |
|---|---|---|
| A-001 | Test masking = real GRACE→GRACE-FO gap | **Validated** (Experiment 7) |
| A-002 | External data can be sourced at correct vintage | Active (untested, Phase 6) |
| A-003 | Spatial relationships stable 2002–2015→test | Active (Experiment 6: indirect support only) |
| A-004 | 2015 anomaly doesn't invalidate 2002–2014 patterns | **Validated with caveat** (Experiment 2) |
| A-005 | Public/private split is random row-level | Active (not independently testable) |
| A-006 | GBM+state-reconstruction beats deep models at this scale | Active (untested, Phase 7) |
| A-007 | 2015–16 El Niño is the anomaly's physical cause | Active (plausible, not ENSO-index-verified) |
| A-008 | ACF/historical-signature features are Phase 4's top priority | **Validated** (Experiment 3) |
| A-009 | In-sample persistence (0.572) understates real test difficulty | **Validated** (Experiment 4) |
| A-010 | Staleness × location-dynamics interaction is real | **Validated, nonlinear form** (Experiment 5) |
| A-011 | Calendar-month coverage gap is an actionable risk | **Validated** (Experiment 6) |
| A-012 | Two narrow GRACE-timeline items remain unresolved | Active, low priority, open by design (Experiment 7) |

## 7. What's still genuinely open (carried forward, not swept away)

- The 2015–16 El Niño link (A-007) is plausible but not verified month-by-month against an ENSO/ONI index.
- A-003's core claim (spatial relationship *stability over time*) still needs its direct test — Experiment 6 gave supportive but indirect evidence only; the real test is a spatial-history feature ablation in Phase 6.
- Five pre-2011 missing training months and three scattered test-period absent runs have no confirmed cause (A-012) — narrow in scope, explicitly not blocking anything, but not silently assumed away either.

---

*All figures/RMSE values above are reproducible by re-running `notebooks/01_eda.ipynb` and `notebooks/02_forecastability.ipynb` top-to-bottom. Full per-experiment detail lives in `docs/PROJECT_PLAN.md` Project Phase 1 and `reports/experiments/experiment_log.csv` (EXP-001 through EXP-007).*
