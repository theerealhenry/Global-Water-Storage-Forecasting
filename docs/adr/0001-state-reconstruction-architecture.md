# ADR-0001 — Adopt state-reconstruction architecture over direct regression

**Status:** Accepted
**Date:** 2026-08-09
**Category:** Modeling
**Deciders:** Steve, Claude

## Context
The original framing was direct regression: `TWS_t + SPEI + soil moisture → TWS_t+1`. EDA surfaced findings this framing doesn't explain: naive persistence already achieves RMSE 0.572 (target std 0.912); 66% of test rows lack `TWS_t`; the delta (target − TWS_t) is very weakly explained by every given feature; masking arrives in contiguous month-level blocks affecting ~99.7%+ of the grid simultaneously; same-month spatial-neighbor correlation is very high (0.981) but functionally unusable during blackouts.

## Evidence
`COMPETITIVE_ANALYSIS.md` §3-4: persistence RMSE 0.572; delta-correlation collapse (TWS_t: r=0.80 for level vs. r=−0.32 for delta); linear reconstruction of TWS_t from concurrent covariates achieves only 0.806/0.821, barely better than the global mean (0.912); nearest-neighbor spatial correlation 0.981 measured directly on Train.csv.

## Current architecture
N/A — this ADR establishes the architecture; there was no prior system to describe.

## Decision
Reframe the problem as partially-observed state reconstruction and state-transition forecasting: `historical observations → reconstruct hydrological state → estimate state evolution → forecast TWS(t+1)`. A dedicated `state/` module sits upstream of all modeling; every model in the champion ladder consumes its output rather than raw columns.

## Reason
Direct regression and naive imputation both fail to explain the masked regime (imputation RMSE 0.821 vs. global-mean floor of 0.912 — essentially no better than guessing). An explicit state layer (last-known value, observation age, trajectory/velocity, historical signatures) can represent *how stale* and *how reliable* the available information is, which the evidence shows is a meaningful distinction, not just a modeling nicety.

## Alternatives considered
**Direct regression, missingness handled via LightGBM's native NaN support alone** — rejected: doesn't distinguish "missing this month" from "missing for 7 consecutive months," which the blackout-degradation-curve experiment (`PROJECT_PLAN.md` Phase 1) is specifically designed to show matters.
**Treat the masked regime as pure imputation** (impute TWS_t, then run persistence) — rejected on direct evidence: imputation RMSE (0.806-0.821) barely beats the global mean.

## Consequences
Touches nearly everything: `PROJECT_PLAN.md`'s phase ordering, the champion ladder in `COMPETITIVE_ANALYSIS.md` §8, and the repository structure (`src/tws_forecast/state/` exists specifically to implement this as shared, reusable code).

## Risks
The "latent state" framing could be mistaken for something we directly observe rather than infer — mitigated by stating the caveat explicitly wherever the term is used (`ARCHITECTURE.md` §1).

## Validation
Confirmed correct if state-aware GBM (champion ladder level 6+) beats raw-feature GBM (level 5) on the masked-regime CV tier specifically, not just overall.

## Affected components
- [x] modeling
- [x] features
- [x] documentation
- [ ] data
- [ ] validation
- [ ] deployment

## Related
- Experiments: the blackout-degradation-curve experiment (Project Phase 1, not yet run)
- MLflow runs: none yet — no code implemented at time of writing
- Submissions: none yet
- Supersedes: none (first ADR)

## Follow-up actions
`ARCHITECTURE.md`, `PROJECT_PLAN.md`, and `COMPETITIVE_ANALYSIS.md` already reflect this decision as of this ADR's date. This ADR documents it retroactively to establish the pattern for future entries.
