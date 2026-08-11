"""Tests for tws_forecast.validation.masking_simulator.apply_blackout_curve."""

import numpy as np
import pandas as pd
import pytest

from tws_forecast.validation.masking_simulator import apply_blackout_curve


def _make_df_with_location_id(n_locations: int = 10, n_months: int = 8) -> pd.DataFrame:
    months = pd.date_range("2004-01-01", periods=n_months, freq="MS")
    rows = []
    for loc_idx in range(n_locations):
        for m in months:
            rows.append(
                {
                    "time": m,
                    "lat": float(loc_idx),
                    "lon": float(loc_idx) * 2,
                    "location_id": f"loc{loc_idx}",
                    "TWS_t": float(loc_idx) + m.month / 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def df() -> pd.DataFrame:
    return _make_df_with_location_id()


def test_requires_location_id_column() -> None:
    bad_df = _make_df_with_location_id().drop(columns=["location_id"])
    with pytest.raises(ValueError, match="location_id"):
        apply_blackout_curve(bad_df, k_distribution=[2, 3], n_windows=2)


def test_n_windows_must_be_positive(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_windows"):
        apply_blackout_curve(df, k_distribution=[2, 3], n_windows=0)


def test_k_distribution_must_not_be_empty(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="k_distribution"):
        apply_blackout_curve(df, k_distribution=[], n_windows=2)


def test_row_count_never_changes(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[2, 3, 4], n_windows=5, seed=1)
    assert len(out) == len(df)


def test_at_most_n_windows_locations_are_masked(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[2], n_windows=4, seed=1)
    masked_locations = out.loc[out["TWS_t_masked"], "location_id"].unique()
    assert len(masked_locations) <= 4


def test_n_windows_capped_at_available_locations() -> None:
    small_df = _make_df_with_location_id(n_locations=3, n_months=6)
    out = apply_blackout_curve(small_df, k_distribution=[2], n_windows=100, seed=1)
    masked_locations = out.loc[out["TWS_t_masked"], "location_id"].unique()
    assert len(masked_locations) <= 3


def test_each_masked_location_gets_exactly_min_k_available_rows_masked(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[3], n_windows=5, seed=2)
    for loc in out.loc[out["TWS_t_masked"], "location_id"].unique():
        loc_rows = out[out["location_id"] == loc]
        n_masked = loc_rows["TWS_t_masked"].sum()
        assert n_masked == 3  # only k=3 is in the distribution, and 8 months >= 3


def test_masked_run_is_the_most_recent_available_months(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[3], n_windows=10, seed=3)
    for loc in out.loc[out["TWS_t_masked"], "location_id"].unique():
        loc_rows = out[out["location_id"] == loc].sort_values("time")
        masked_flags = loc_rows["TWS_t_masked"].to_numpy()
        # the masked run must be a suffix of the sorted-by-time rows
        first_masked = masked_flags.argmax()
        assert masked_flags[first_masked:].all()
        assert not masked_flags[:first_masked].any()


def test_k_larger_than_available_rows_is_clipped() -> None:
    small_df = _make_df_with_location_id(n_locations=2, n_months=2)
    out = apply_blackout_curve(small_df, k_distribution=[7], n_windows=2, seed=4)
    for loc in out["location_id"].unique():
        loc_rows = out[out["location_id"] == loc]
        assert loc_rows["TWS_t_masked"].sum() <= 2  # clipped to available rows


def test_simulated_k_column_matches_masking(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[2, 5], n_windows=5, seed=5)
    masked = out[out["TWS_t_masked"]]
    unmasked = out[~out["TWS_t_masked"]]
    assert masked["simulated_k"].isin([2, 5]).all()
    assert unmasked["simulated_k"].isna().all()


def test_tws_t_masked_always_equals_isna(df: pd.DataFrame) -> None:
    out = apply_blackout_curve(df, k_distribution=[2, 3, 4], n_windows=6, seed=6)
    assert (out["TWS_t_masked"] == out["TWS_t"].isna()).all()


def test_determinism_given_fixed_seed(df: pd.DataFrame) -> None:
    out_a = apply_blackout_curve(df, k_distribution=[2, 3, 4, 5], n_windows=6, seed=42)
    out_b = apply_blackout_curve(df, k_distribution=[2, 3, 4, 5], n_windows=6, seed=42)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_different_seeds_can_diverge(df: pd.DataFrame) -> None:
    out_a = apply_blackout_curve(df, k_distribution=[2, 3, 4, 5], n_windows=6, seed=1)
    out_b = apply_blackout_curve(df, k_distribution=[2, 3, 4, 5], n_windows=6, seed=2)
    assert not out_a["TWS_t_masked"].equals(out_b["TWS_t_masked"])


def test_no_location_selected_twice(df: pd.DataFrame) -> None:
    # location draws are without replacement -- verify indirectly by
    # confirming no location has more masked rows than the largest k in
    # the distribution (a double-draw would risk masking more than one
    # k's worth via overlapping/compounding logic, though the current
    # implementation would just overwrite, not compound -- this test
    # guards the "at most one run per location" invariant regardless).
    out = apply_blackout_curve(df, k_distribution=[2, 3, 4], n_windows=10, seed=7)
    for loc in out["location_id"].unique():
        n_masked = out.loc[out["location_id"] == loc, "TWS_t_masked"].sum()
        assert n_masked <= 4
