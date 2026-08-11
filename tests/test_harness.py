"""Tests for tws_forecast.validation.harness."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.validation.harness import (
    CandidateReport,
    evaluate_candidate,
    promote,
)
from tws_forecast.validation.phase1_constants import PROMOTION_THRESHOLDS
from tws_forecast.validation.tiers import TierResult


class MeanPredictor:
    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


def _make_tier_result(overall_rmse: float, tier: int = 2) -> TierResult:
    return TierResult(
        tier=tier, scenario_name="x", predictions=pd.DataFrame(),
        fold_rmses=(overall_rmse,), overall_rmse=overall_rmse,
    )


def _make_bucket_decomp(bucket_rmse: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"slice_type": "staleness_bucket", "slice_value": k, "n": 10, "rmse": v}
        for k, v in bucket_rmse.items()
    ]
    return pd.DataFrame(rows)


# --- evaluate_candidate: real end-to-end run ---------------------------------


def test_evaluate_candidate_end_to_end(train_df: pd.DataFrame) -> None:
    report = evaluate_candidate(MeanPredictor(), train_df, candidate_id="mean_v1", n_anchors=1)
    assert isinstance(report, CandidateReport)
    assert report.candidate_id == "mean_v1"
    assert report.tier1 is not None
    assert report.tier2 is not None
    assert report.tier3 is not None
    assert report.tier1_decomposition is not None
    assert report.tier2_decomposition is not None
    assert report.tier3_decomposition is not None


def test_evaluate_candidate_skip_tier3(train_df: pd.DataFrame) -> None:
    report = evaluate_candidate(MeanPredictor(), train_df, candidate_id="mean_v1", include_tier3=False)
    assert report.tier3 is None
    assert report.tier3_decomposition is None


def test_evaluate_candidate_with_acf_lookup_produces_degradation_slope(train_df: pd.DataFrame) -> None:
    report = evaluate_candidate(MeanPredictor(), train_df, candidate_id="mean_v1", include_tier3=False)
    locations = sorted(report.tier2.predictions["location_id"].unique())
    acf_lookup = pd.Series(np.linspace(0.1, 0.95, len(locations)), index=locations)

    report_with_acf = evaluate_candidate(
        MeanPredictor(), train_df, candidate_id="mean_v1", acf_lookup=acf_lookup, include_tier3=False
    )
    assert report_with_acf.degradation_slope is not None
    assert len(report_with_acf.degradation_slope) > 0
    assert report.degradation_slope is None  # no acf_lookup -> no slope


# --- promote(): the hard Tier-3-only rule ------------------------------------


def test_promote_raises_on_tier3_only_report() -> None:
    report = CandidateReport(candidate_id="x", tier1=None, tier2=None, tier3=_make_tier_result(0.5, tier=3))
    with pytest.raises(ValueError, match="requires both Tier 1 and Tier 2"):
        promote(report)


def test_promote_raises_when_only_tier1_present() -> None:
    report = CandidateReport(candidate_id="x", tier1=_make_tier_result(0.5, tier=1), tier2=None)
    with pytest.raises(ValueError, match="requires both Tier 1 and Tier 2"):
        promote(report)


def test_promote_raises_when_only_tier2_present() -> None:
    report = CandidateReport(candidate_id="x", tier1=None, tier2=_make_tier_result(0.5, tier=2))
    with pytest.raises(ValueError, match="requires both Tier 1 and Tier 2"):
        promote(report)


# --- promote(): ladder rungs, no baseline -------------------------------------


@pytest.mark.parametrize(
    "overall_rmse,expected_rung",
    [
        (0.45, "exceptional"),
        (0.51, "serious_contender"),
        (0.54, "beat_mohar"),
        (0.56, "oracle_ceiling"),
        (0.60, "naive_floor"),
        (0.70, None),  # doesn't even clear the naive floor
    ],
)
def test_promote_ladder_rungs(overall_rmse: float, expected_rung: str | None) -> None:
    report = CandidateReport(
        candidate_id="x", tier1=_make_tier_result(overall_rmse, tier=1),
        tier2=_make_tier_result(overall_rmse, tier=2),
    )
    decision = promote(report)
    assert decision.rung == expected_rung
    assert decision.promoted == (expected_rung is not None)


def test_promote_clears_naive_floor_with_no_regression() -> None:
    # Directly exercises the plan's third required test case: a report
    # clearing 0.6573 with no regime regression is promoted at the correct
    # ladder rung.
    report = CandidateReport(
        candidate_id="beats_floor",
        tier1=_make_tier_result(0.60, tier=1),
        tier2=_make_tier_result(0.60, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.60, "k=6": 0.62, "k=7": 0.65}),
    )
    baseline = CandidateReport(
        candidate_id="baseline",
        tier1=_make_tier_result(0.6573, tier=1),
        tier2=_make_tier_result(0.6573, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.65, "k=6": 0.68, "k=7": 0.70}),
    )
    decision = promote(report, baseline_report=baseline)
    assert decision.promoted is True
    assert decision.rung == "naive_floor"
    assert decision.regressed_buckets == ()
    assert 0.60 < PROMOTION_THRESHOLDS["naive_floor"]


# --- promote(): hard-staleness-bucket regression check ------------------------


def test_promote_blocks_on_hard_bucket_regression_despite_aggregate_improvement() -> None:
    # Candidate's AGGREGATE Tier 2 RMSE improves over baseline, but its
    # k=6/k=7 buckets are worse -- must not be promoted.
    baseline = CandidateReport(
        candidate_id="baseline",
        tier1=_make_tier_result(0.62, tier=1),
        tier2=_make_tier_result(0.62, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.60, "k=6": 0.65, "k=7": 0.68}),
    )
    candidate = CandidateReport(
        candidate_id="regresses_on_hard_buckets",
        tier1=_make_tier_result(0.58, tier=1),
        tier2=_make_tier_result(0.58, tier=2),  # better aggregate than baseline
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.55, "k=6": 0.70, "k=7": 0.75}),  # worse on k=6, k=7
    )
    decision = promote(candidate, baseline_report=baseline)
    assert decision.promoted is False
    assert decision.rung is None
    assert set(decision.regressed_buckets) == {"k=6", "k=7"}


def test_promote_allows_improvement_on_all_hard_buckets() -> None:
    baseline = CandidateReport(
        candidate_id="baseline",
        tier1=_make_tier_result(0.62, tier=1),
        tier2=_make_tier_result(0.62, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.60, "k=6": 0.65, "k=7": 0.68}),
    )
    candidate = CandidateReport(
        candidate_id="genuinely_better",
        tier1=_make_tier_result(0.55, tier=1),
        tier2=_make_tier_result(0.55, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.50, "k=6": 0.60, "k=7": 0.62}),
    )
    decision = promote(candidate, baseline_report=baseline)
    assert decision.promoted is True
    assert decision.regressed_buckets == ()


def test_promote_with_baseline_requires_tier2_decomposition() -> None:
    report = CandidateReport(
        candidate_id="x", tier1=_make_tier_result(0.5, tier=1), tier2=_make_tier_result(0.5, tier=2),
        tier2_decomposition=None,
    )
    baseline = CandidateReport(
        candidate_id="baseline", tier1=_make_tier_result(0.6, tier=1), tier2=_make_tier_result(0.6, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.6}),
    )
    with pytest.raises(ValueError, match="tier2_decomposition"):
        promote(report, baseline_report=baseline)


def test_promote_missing_bucket_in_one_side_is_not_compared() -> None:
    # If a bucket only exists on one side (e.g. baseline never saw k=7),
    # it's simply not part of the regression check for that bucket --
    # rather than raising or assuming the worst.
    baseline = CandidateReport(
        candidate_id="baseline", tier1=_make_tier_result(0.6, tier=1), tier2=_make_tier_result(0.6, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.6, "k=6": 0.65}),  # no k=7
    )
    candidate = CandidateReport(
        candidate_id="x", tier1=_make_tier_result(0.55, tier=1), tier2=_make_tier_result(0.55, tier=2),
        tier2_decomposition=_make_bucket_decomp({"k=5": 0.55, "k=6": 0.60, "k=7": 0.90}),
    )
    decision = promote(candidate, baseline_report=baseline)
    assert decision.promoted is True  # k=7 wasn't comparable, k=5/k=6 both improved
