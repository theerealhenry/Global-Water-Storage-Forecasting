# ADR-0004 — Close Project Phase 1; bind Phase 2's validation harness to Phase 1's measured distributions

**Status:** Accepted
**Date:** 2026-08-11
**Category:** Validation
**Deciders:** Steve, Claude

## Context

Project Phase 1 (`PROJECT_PLAN.md`) ran all 7 ordered experiments to completion: masking-process
reproduction, the 2015 persistence-anomaly hard gate, the blackout-degradation curve, the last-known-state
baseline, the staleness×ACF interaction, covariate shift, and the real GRACE/GRACE-FO mission timeline.
Full findings live in `notebooks/02_forecastability.ipynb` §1-18 and `docs/ASSUMPTIONS.md` A-001/A-004/
A-007/A-008/A-009/A-010/A-011/A-012. This ADR is the formal record that Phase 1 is closed, and — more
importantly — records the specific design decisions Phase 1's *empirical, measured* distributions force
onto Phase 2's validation harness, so Phase 2 doesn't quietly rebuild a generic/synthetic version of
something Phase 1 already characterized precisely.

## Evidence

- **Masking is bimodal, not gradual**: every test month is either 0% masked or 99.58-99.97% masked, never
  in between (Experiment 1).
- **The real staleness-to-target distribution on blackout months is k=2 (×4), k=3 (×3), k=4 (×2), k=5/6/7
  (×1 each)** — not uniform, not geometric, and specifically NOT the naive "row month minus last observed
  month" quantity (Experiment 4, the row_month+1 correction).
- **Test's 18 months are a specific, fixed, non-random-looking sample of the calendar** — October has zero
  representation; months that recur twice within the 18 (Jan/Feb/Mar/Jun/Jul/Sep/Dec) get ~2x the row-share
  of months appearing once (Apr/May/Aug/Nov) (Experiment 6, A-011).
- **Baseline A's in-sample persistence RMSE (0.572) is not the real naive floor** — Baseline D (Hybrid,
  0.6573) is, because 12/18 real test months have no current observation at all (Experiment 4, A-009).
- **Staleness (k) and per-location ACF/volatility interact nonlinearly** (ρ^k, not k·ρ) — a validation
  harness that buckets purely by k without also stratifying by ACF will average over a real, mechanistically
  understood source of heterogeneity (Experiment 5, A-010).
- **The real hard gap (Jul 2017-Jun 2018, 12 months) is grounded in the actual GRACE→GRACE-FO mission
  history**, not an artifact of this competition (Experiment 7, A-001 — now Validated).

## Current architecture

`PROJECT_PLAN.md` Project Phase 2 (pre-Phase-1-completion draft) specified a streak-aware masking
simulator and three validation tiers only in general terms — "nulls `TWS_t` in contiguous multi-month
blocks," "expanding-window splits" — without binding those mechanisms to any specific, measured
distribution. That was appropriate before Phase 1 existed; it is no longer appropriate now that the real
distributions are known.

## Decision

Phase 2's validation harness is built to reproduce Phase 1's measured distributions exactly, not
approximate them generically:

1. **The streak-aware masking simulator's blackout-length distribution is drawn from the real
   staleness-to-target histogram found in Experiment 4** (k=2×4, k=3×3, k=4×2, k=5/6/7×1 each — or the
   nearest well-justified generalization of it, e.g. resampling with replacement from this empirical
   distribution), not a synthetic geometric or uniform distribution.
2. **Tier 3 (test-regime-matched) validation replays the real 6-FULL/12-BLACKOUT month structure**
   established in Experiments 1/4/7, anchored to the real calendar-month identities (so it also respects
   A-011's coverage gap), not just "some fraction masked."
3. **The internal promotion-rule target ladder is recalibrated against Baseline D (0.6573), not Baseline
   A (0.572)**, per A-009 — `COMPETITIVE_ANALYSIS.md` §6 already carries this caveat; Phase 2's actual
   promotion-rule code must implement it, not just document it.
4. **The error-decomposition table's staleness-bucket breakdown uses the real k=2-7 buckets found in
   Experiment 4**, cross-cut by ACF quartile/decile per Experiment 5's finding that staleness alone
   under-describes the regime — not a generic "1-2mo/3-4mo/5+mo" bucketing invented independently of the
   data.
5. **Fold construction is checked against A-011's calendar-month coverage gap** (no fold may be
   constructed in a way that hides the October gap or the 2x/1x recurrence imbalance) before being
   accepted as a valid CV scheme.

## Reason

Phase 1 exists specifically so Phase 2 doesn't have to guess at these distributions. Building a generic
masking simulator now, after already measuring the real one precisely, would silently discard the
highest-leverage output of the entire Phase 1 investigation and reintroduce exactly the kind of
"internally consistent but not grounded in the real mechanism" risk Experiment 7 was built to rule out.

## Alternatives considered

**Build a general-purpose, distribution-agnostic masking simulator, parameterized for later tuning** —
rejected for the first version: Phase 1 already did the tuning work empirically; a parameterized version
is worth adding *later* if genuinely needed (e.g., for robustness/sensitivity testing in Phase 9), but the
default and primary validation scheme should be the measured real distribution, not an arbitrary prior.

**Use only Tier 1/Tier 2 (generic k-bucketed blackout simulation) and skip a dedicated Tier 3 real-replay
tier** — rejected: Experiment 4 already built and validated a real-replay methodology (Method B); not
reusing it in the actual validation harness would mean redoing equivalent work with less rigor.

## Consequences

Phase 2's `src/tws_forecast/validation/` module takes a direct dependency on Phase 1's measured constants
(the k-distribution, the FULL/BLACKOUT calendar-month identities, the Baseline A/B/C/D numbers) — these
should be defined once, in one place (a `phase1_constants.py` or equivalent, sourced with a comment
pointing back to the exact notebook section and experiment log entry), not copy-pasted into multiple
modules where they could drift out of sync if Phase 1 is ever revisited.

## Risks

If Phase 1's measured distributions are themselves an artifact of the specific 18-month sample (rather
than representative of the true underlying process), binding Phase 2 tightly to them could overfit the
validation scheme to this one sample. Mitigation: Tier 1/Tier 2 remain generic (not real-replay-bound) as
a robustness cross-check against Tier 3 specifically for this reason — if Tier 1/2 and Tier 3 disagree
sharply, that's itself a signal worth investigating, not something to average away.

## Validation

Confirmed correct if: running the Phase 2 harness against a trivial model reproduces Baseline D's 0.6573
figure (approximately) as the harness's own Tier 3 "hybrid naive" reference score — if it doesn't, the
harness isn't faithfully reproducing what Phase 1 already established, and that's a bug in Phase 2, not a
new finding.

## Affected components

- [x] validation
- [x] data
- [ ] modeling
- [ ] features
- [ ] deployment
- [x] documentation

## Related

- Experiments: EXP-001 through EXP-007 (`reports/experiments/experiment_log.csv`)
- MLflow runs: none yet (Phase 2 migrates the flat log toward MLflow per `PROJECT_PLAN.md` Phase 0/2)
- Submissions: none yet
- Supersedes: none
- Superseded by: none
- Related ADRs: ADR-0001 (state-reconstruction architecture — Phase 2's tiers are designed around the
  same masked/unmasked regime distinction that motivated ADR-0001)

## Follow-up actions

- [x] `PROJECT_PLAN.md` Project Phase 1 marked complete (all 7 experiment checkboxes, Definition of Done
  marked MET)
- [x] `PROJECT_PLAN.md` Project Phase 2 rewritten with the detailed, Phase-1-grounded execution plan this
  ADR's decisions require (see Phase 2 section)
- [ ] `src/tws_forecast/validation/` module implementing the above — tracked as Phase 2 implementation
  work, not part of this ADR itself
