"""Tests for tws_forecast.validation.decomposition."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.validation.decomposition import (
    ACF_QUARTILE_ORDER,
    decompose,
    degradation_slope,
)
from tws_forecast.validation.phase1_constants import ACF_QUARTILE_AR1_PARAMS
from tws_forecast.validation.tiers import TierResult, run_tier1, run_tier2


class MeanPredictor:
    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


@pytest.fixture()
def tier2_result(train_df: pd.DataFrame) -> TierResult:
    return run_tier2(MeanPredictor(), train_df)


@pytest.fixture()
def acf_lookup(tier2_result: TierResult) -> pd.Series:
    # Synthetic but deterministic per-location ACF, covering every location
    # actually present in tier2_result's predictions — enough distinct
    # values to form real quartiles.
    locations = sorted(tier2_result.predictions["location_id"].unique())
    values = np.linspace(0.1, 0.95, len(locations))
    return pd.Series(values, index=locations)


# --- decompose() row-set completeness ---------------------------------------


def test_overall_row_matches_tier_result(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    overall = decomp[decomp["slice_type"] == "overall"]
    assert len(overall) == 1
    assert overall.iloc[0]["n"] == len(tier2_result.predictions)
    assert overall.iloc[0]["rmse"] == pytest.approx(tier2_result.overall_rmse, rel=1e-9)


def test_regime_rows_sum_to_overall(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    overall_n = decomp.loc[decomp["slice_type"] == "overall", "n"].iloc[0]
    regime_n = decomp.loc[decomp["slice_type"] == "regime", "n"].sum()
    assert regime_n == overall_n


def test_staleness_bucket_only_uses_real_k_values(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    staleness = decomp[decomp["slice_type"] == "staleness_bucket"]
    assert len(staleness) > 0
    for slice_value in staleness["slice_value"]:
        assert slice_value.startswith("k=")
        k = int(slice_value.removeprefix("k="))
        assert 2 <= k <= 7  # never an invented 1-2mo/3-4mo/5+mo scheme


def test_staleness_bucket_rows_sum_to_masked_regime_n(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    masked_n = decomp.loc[
        (decomp["slice_type"] == "regime") & (decomp["slice_value"] == "masked"), "n"
    ].iloc[0]
    staleness_n = decomp.loc[decomp["slice_type"] == "staleness_bucket", "n"].sum()
    assert staleness_n == masked_n


def test_staleness_x_acf_quartile_requires_acf_lookup(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result, acf_lookup=None)
    assert (decomp["slice_type"] == "staleness_x_acf_quartile").sum() == 0


def test_staleness_x_acf_quartile_no_rows_silently_dropped(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    cross_cut = decomp[decomp["slice_type"] == "staleness_x_acf_quartile"]
    assert len(cross_cut) > 0

    staleness = decomp[decomp["slice_type"] == "staleness_bucket"]
    for _, row in staleness.iterrows():
        k = row["slice_value"]  # e.g. "k=3"
        matching = cross_cut[cross_cut["slice_value"].str.startswith(f"{k}|")]
        assert matching["n"].sum() == row["n"], (
            f"cross-cut rows for {k} don't sum to the staleness bucket's own n "
            "— some masked rows were silently dropped"
        )


def test_staleness_x_acf_quartile_labels_are_valid_quartiles_or_unknown(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    cross_cut = decomp[decomp["slice_type"] == "staleness_x_acf_quartile"]
    valid_labels = set(ACF_QUARTILE_ORDER) | {"unknown_acf"}
    for slice_value in cross_cut["slice_value"]:
        quartile = slice_value.split("|", 1)[1]
        assert quartile in valid_labels


def test_hemisphere_rows_sum_to_overall(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    overall_n = decomp.loc[decomp["slice_type"] == "overall", "n"].iloc[0]
    hemi_n = decomp.loc[decomp["slice_type"] == "hemisphere", "n"].sum()
    assert hemi_n == overall_n
    assert set(decomp.loc[decomp["slice_type"] == "hemisphere", "slice_value"]) == {
        "Northern", "Southern",
    }


def test_extreme_target_rows_sum_to_overall(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    overall_n = decomp.loc[decomp["slice_type"] == "overall", "n"].iloc[0]
    extreme_n = decomp.loc[decomp["slice_type"] == "extreme_target", "n"].sum()
    assert extreme_n == overall_n


def test_rapid_change_rows_sum_to_overall(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)
    overall_n = decomp.loc[decomp["slice_type"] == "overall", "n"].iloc[0]
    rapid_n = decomp.loc[decomp["slice_type"] == "rapid_change", "n"].sum()
    assert rapid_n == overall_n


def test_tier1_result_skips_staleness_rows_gracefully(train_df: pd.DataFrame) -> None:
    tier1_result = run_tier1(MeanPredictor(), train_df)
    decomp = decompose(tier1_result)
    assert (decomp["slice_type"] == "staleness_bucket").sum() == 0
    assert (decomp["slice_type"] == "staleness_x_acf_quartile").sum() == 0
    # everything else still present
    assert (decomp["slice_type"] == "overall").sum() == 1
    assert (decomp["slice_type"] == "hemisphere").sum() == 2


def test_decompose_empty_predictions_raises() -> None:
    empty_result = TierResult(
        tier=1, scenario_name="x", predictions=pd.DataFrame(),
        fold_rmses=(), overall_rmse=float("nan"),
    )
    with pytest.raises(ValueError, match="empty predictions"):
        decompose(empty_result)


# --- degradation_slope() -----------------------------------------------------


def test_degradation_slope_theoretical_rmse_matches_ar1_formula(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    slope = degradation_slope(decomp)
    assert len(slope) > 0

    for _, row in slope.iterrows():
        params = ACF_QUARTILE_AR1_PARAMS[row["acf_quartile"]]
        expected = params["sigma"] * np.sqrt(2 * (1 - params["rho"] ** row["k"]))
        assert row["theoretical_rmse"] == pytest.approx(expected, rel=1e-9)


def test_degradation_slope_first_k_per_quartile_has_nan_delta(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    slope = degradation_slope(decomp)
    for quartile in slope["acf_quartile"].unique():
        sub = slope[slope["acf_quartile"] == quartile].sort_values("k")
        first = sub.iloc[0]
        assert np.isnan(first["empirical_delta_rmse"])
        assert np.isnan(first["theoretical_delta_rmse"])


def test_degradation_slope_delta_matches_consecutive_difference(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    slope = degradation_slope(decomp)
    for quartile in slope["acf_quartile"].unique():
        sub = slope[slope["acf_quartile"] == quartile].sort_values("k").reset_index(drop=True)
        for i in range(1, len(sub)):
            expected_delta = sub.loc[i, "theoretical_rmse"] - sub.loc[i - 1, "theoretical_rmse"]
            assert sub.loc[i, "theoretical_delta_rmse"] == pytest.approx(expected_delta, rel=1e-9)


def test_degradation_slope_without_cross_cut_rows_raises(tier2_result: TierResult) -> None:
    decomp = decompose(tier2_result)  # no acf_lookup -> no cross-cut rows
    with pytest.raises(ValueError, match="staleness_x_acf_quartile"):
        degradation_slope(decomp)


def test_degradation_slope_quartile_labels_are_canonical(
    tier2_result: TierResult, acf_lookup: pd.Series
) -> None:
    decomp = decompose(tier2_result, acf_lookup=acf_lookup)
    slope = degradation_slope(decomp)
    assert set(slope["acf_quartile"]).issubset(set(ACF_QUARTILE_ORDER))


# --- _compute_acf_quartiles error path (via decompose) -----------------------


def test_acf_lookup_with_too_few_distinct_values_raises(tier2_result: TierResult) -> None:
    locations = sorted(tier2_result.predictions["location_id"].unique())
    degenerate_lookup = pd.Series([0.5] * len(locations), index=locations)  # only 1 distinct value
    with pytest.raises(ValueError, match="distinct value"):
        decompose(tier2_result, acf_lookup=degenerate_lookup)
