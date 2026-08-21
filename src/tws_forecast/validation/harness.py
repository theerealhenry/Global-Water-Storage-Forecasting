"""The harness orchestrator: ties splitters + scenarios + tiers +
decomposition together, and is the single legitimate way any candidate gets
called a "champion" — the promotion rule from ``docs/COMPETITIVE_ANALYSIS.md``
§6 is executable here, not aspirational.

Two integrity safeguards from ``docs/PROJECT_PLAN.md`` §2.4 / ``docs/
ARCHITECTURE.md`` §11 are enforced in code, not just documented:

1. ``promote()`` raises if called without both Tier 1 and Tier 2 results —
   a Tier-3-only score can never satisfy promotion, since Tier 3 is
   diagnostic/robustness-only by design (it may draw on the real test set's
   *structure* replayed onto historical analogs, but final model selection
   is never made from it alone).
2. A candidate that improves aggregate RMSE while regressing the hardest
   staleness buckets (k=5, 6, 7 — the ones A-010 found hide the most
   heterogeneity) relative to a baseline is not promoted on the aggregate
   number alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from tws_forecast.validation.decomposition import decompose, degradation_slope
from tws_forecast.validation.phase1_constants import PROMOTION_THRESHOLDS
from tws_forecast.validation.tiers import Predictor, TierResult, run_tier1, run_tier2, run_tier3

logger = logging.getLogger(__name__)

__all__ = ["CandidateReport", "PromotionDecision", "evaluate_candidate", "promote"]

# Ordered toughest-to-easiest, so promote() can report the *best* rung a
# candidate clears rather than merely whether it clears the floor. Values
# themselves live in phase1_constants.PROMOTION_THRESHOLDS — this is just
# the iteration order, kept here so a future re-ordering of the dict
# wouldn't silently change promotion behavior.
_RUNG_ORDER = ["exceptional", "serious_contender", "beat_mohar", "oracle_ceiling", "naive_floor"]

# The staleness buckets A-010 found hide the most real heterogeneity — a
# candidate is not promoted if it regresses on any of these relative to a
# baseline, even while improving in aggregate.
_HARD_STALENESS_BUCKETS = ("k=5", "k=6", "k=7")


@dataclass(frozen=True)
class CandidateReport:
    """Everything ``promote()`` and a human reviewer need for one candidate."""

    candidate_id: str
    tier1: TierResult | None = None
    tier2: TierResult | None = None
    tier3: TierResult | None = None
    tier1_decomposition: pd.DataFrame | None = None
    tier2_decomposition: pd.DataFrame | None = None
    tier3_decomposition: pd.DataFrame | None = None
    degradation_slope: pd.DataFrame | None = None


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    promoted: bool
    rung: str | None
    reason: str
    regressed_buckets: tuple[str, ...] = ()


def evaluate_candidate(
    model: Predictor,
    df: pd.DataFrame,
    candidate_id: str,
    acf_lookup: pd.Series | None = None,
    include_tier3: bool = True,
    n_anchors: int = 3,
    tier1_scenario: str = "expanding_window",
    tier2_scenario: str = "blackout_curve",
) -> CandidateReport:
    """Run a candidate through Tier 1 + Tier 2 (+ Tier 3 diagnostically),
    building the full decomposition table and degradation slope for each.

    Parameters
    ----------
    model:
        Anything matching ``validation.tiers.Predictor``.
    df:
        ``Train.csv``-shaped frame — same input every tier function takes.
    candidate_id:
        Free-text identifier carried through the report (and, from step
        2.10, the experiment log).
    acf_lookup:
        Optional per-location ACF, passed through to ``decompose()`` for the
        staleness x ACF-quartile cross-cut and the degradation slope.
        Without it, those two are simply absent from the report — Project
        Phase 4 supplies the real values.
    include_tier3:
        Tier 3 is diagnostic-only and slower (multiple anchor refits) — set
        False to skip it for a quick Tier 1/2-only check. ``promote()``
        never needs Tier 3 to be present.
    n_anchors:
        Forwarded to ``run_tier3``.
    tier1_scenario, tier2_scenario:
        The named ``configs/validation/*.yaml`` scenario each tier loads —
        default to the standard, full-rigor ``"expanding_window"``/
        ``"blackout_curve"`` scenarios every promotion decision is made
        against. Override to a cheaper scenario (fewer folds/windows, e.g.
        a project-defined ``"expanding_window_quick"``) for exploratory or
        comparative runs where full-rigor CV isn't the point — Project
        Phase 4 step 4.9's target-transformation comparison
        (``notebooks/05_state_features.ipynb``) is the first caller to do
        this, since it repeats the whole harness once per candidate
        target-transform and a full-cost fold count made an early proof
        run impractically slow at real ~15,715-location, ~2.15M-row scale.
        Never override for a report a `promote()` call will actually be
        based on.
    """
    tier1 = run_tier1(model, df, scenario=tier1_scenario)
    tier2 = run_tier2(model, df, scenario=tier2_scenario)
    tier3 = run_tier3(model, df, n_anchors=n_anchors) if include_tier3 else None

    tier1_decomposition = decompose(tier1, acf_lookup=acf_lookup)
    tier2_decomposition = decompose(tier2, acf_lookup=acf_lookup)
    tier3_decomposition = decompose(tier3, acf_lookup=acf_lookup) if tier3 is not None else None

    slope = None
    if acf_lookup is not None:
        try:
            slope = degradation_slope(tier2_decomposition)
        except ValueError:
            logger.info(
                "evaluate_candidate(%r): degradation_slope unavailable (no "
                "staleness x ACF cross-cut rows in Tier 2's decomposition)",
                candidate_id,
            )

    logger.info(
        "evaluate_candidate(%r): tier1_rmse=%.4f tier2_rmse=%.4f tier3_rmse=%s",
        candidate_id,
        tier1.overall_rmse,
        tier2.overall_rmse,
        f"{tier3.overall_rmse:.4f}" if tier3 is not None else "n/a",
    )

    return CandidateReport(
        candidate_id=candidate_id,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        tier1_decomposition=tier1_decomposition,
        tier2_decomposition=tier2_decomposition,
        tier3_decomposition=tier3_decomposition,
        degradation_slope=slope,
    )


def _bucket_rmse_map(decomp_df: pd.DataFrame) -> dict[str, float]:
    bucket_rows = decomp_df[decomp_df["slice_type"] == "staleness_bucket"]
    return dict(zip(bucket_rows["slice_value"], bucket_rows["rmse"], strict=True))


def promote(
    report: CandidateReport, baseline_report: CandidateReport | None = None
) -> PromotionDecision:
    """The only legitimate way a candidate becomes a champion.

    Hard rule: raises if ``report`` doesn't carry both Tier 1 and Tier 2
    results — Tier 3 alone is never sufficient, regardless of its score.

    The promotion ladder (``phase1_constants.PROMOTION_THRESHOLDS``) is
    evaluated against Tier 2's overall RMSE, not Tier 1's: Tier 2's
    validation window already mixes masked and unmasked rows within each
    fold (``apply_blackout_curve`` only masks a sampled subset of that
    fold's locations), making it the tier structurally analogous to
    Baseline D's realistic, mixed-regime measurement that the ladder's
    numbers were calibrated against (``docs/COMPETITIVE_ANALYSIS.md`` §6) —
    without needing Tier 3 (which ``docs/ARCHITECTURE.md`` §11 forbids using
    as a sole promotion basis) to supply that mix.

    If ``baseline_report`` is given, a candidate that regresses on any of
    the hardest staleness buckets (k=5, 6, 7) relative to it is not
    promoted, even if its aggregate Tier 2 RMSE improved — the improvement
    could be hiding fragility in exactly the regime A-010 found matters
    most.

    Parameters
    ----------
    report:
        Output of ``evaluate_candidate``.
    baseline_report:
        The current champion's (or any reference candidate's) report, for
        the regression check. If omitted, only the ladder threshold is
        checked.
    """
    if report.tier1 is None or report.tier2 is None:
        raise ValueError(
            f"promote() requires both Tier 1 and Tier 2 results for candidate "
            f"{report.candidate_id!r} — Tier 3 is diagnostic/robustness-only and "
            "can never satisfy promotion on its own (docs/ARCHITECTURE.md §11)."
        )

    regressed: list[str] = []
    if baseline_report is not None:
        if report.tier2_decomposition is None or baseline_report.tier2_decomposition is None:
            raise ValueError(
                "promote() with a baseline_report requires tier2_decomposition on "
                "both reports to run the hard-staleness-bucket regression check."
            )
        candidate_buckets = _bucket_rmse_map(report.tier2_decomposition)
        baseline_buckets = _bucket_rmse_map(baseline_report.tier2_decomposition)
        for bucket in _HARD_STALENESS_BUCKETS:
            if bucket in candidate_buckets and bucket in baseline_buckets:
                if candidate_buckets[bucket] > baseline_buckets[bucket]:
                    regressed.append(bucket)

    if regressed:
        reason = (
            f"regressed on hard staleness bucket(s) {sorted(regressed)} relative to "
            f"baseline {baseline_report.candidate_id!r}, despite any aggregate "
            "improvement — not promoted (docs/PHASE2_EXECUTION_PLAN.md §2.9)"
        )
        logger.info("promote(%r): NOT promoted — %s", report.candidate_id, reason)
        return PromotionDecision(
            candidate_id=report.candidate_id,
            promoted=False,
            rung=None,
            reason=reason,
            regressed_buckets=tuple(sorted(regressed)),
        )

    overall_rmse = report.tier2.overall_rmse
    rung = next(
        (name for name in _RUNG_ORDER if overall_rmse < PROMOTION_THRESHOLDS[name]),
        None,
    )

    if rung is not None:
        reason = f"cleared rung {rung!r} — Tier 2 overall RMSE {overall_rmse:.4f} < {PROMOTION_THRESHOLDS[rung]:.4f}"
    else:
        reason = (
            f"did not clear the naive floor — Tier 2 overall RMSE {overall_rmse:.4f} "
            f">= {PROMOTION_THRESHOLDS['naive_floor']:.4f} (Baseline D)"
        )

    logger.info("promote(%r): promoted=%s — %s", report.candidate_id, rung is not None, reason)

    return PromotionDecision(
        candidate_id=report.candidate_id,
        promoted=rung is not None,
        rung=rung,
        reason=reason,
        regressed_buckets=(),
    )
