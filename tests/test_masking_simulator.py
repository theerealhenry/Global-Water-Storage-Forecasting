"""Tests for tws_forecast.validation.masking_simulator — run against small,
hand-built synthetic fixtures (not the CSV golden fixtures), per
docs/PHASE2_EXECUTION_PLAN.md step 2.4."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tws_forecast.validation.masking_simulator import MaskingScenario, apply_masking


def _make_synthetic_df(n_locations: int = 3, n_months: int = 6) -> pd.DataFrame:
    months = pd.date_range("2004-01-01", periods=n_months, freq="MS")
    rows = []
    for loc_idx in range(n_locations):
        lat, lon = float(loc_idx), float(loc_idx) * 2
        for m in months:
            rows.append(
                {
                    "time": m,
                    "lat": lat,
                    "lon": lon,
                    "TWS_t": float(loc_idx) + m.month / 100,
                    "SPEI_01_t": 0.5,
                    "SOIL_MOISTURE_t": -0.5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_df()


def _abrupt_scenario(start: str, end: str, **kwargs) -> MaskingScenario:
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    streak = (end_d.year * 12 + end_d.month) - (start_d.year * 12 + start_d.month) + 1
    return MaskingScenario(
        blackout_start=start_d,
        blackout_end=end_d,
        streak_length=streak,
        source_rationale="unit test",
        **kwargs,
    )


def test_masked_rows_null_tws_t_other_columns_populated(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-03-01")
    out = apply_masking(synthetic_df, scenario)

    masked = out[out["TWS_t_masked"]]
    assert len(masked) > 0
    assert masked["TWS_t"].isna().all()
    # everything else stays populated
    for col in ["time", "lat", "lon", "SPEI_01_t", "SOIL_MOISTURE_t"]:
        assert masked[col].isna().sum() == 0

    unmasked = out[~out["TWS_t_masked"]]
    assert unmasked["TWS_t"].isna().sum() == 0


def test_row_count_never_changes(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-04-01")
    out = apply_masking(synthetic_df, scenario)
    assert len(out) == len(synthetic_df)


def test_only_rows_within_window_are_masked(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-03-01")
    out = apply_masking(synthetic_df, scenario)
    outside_window = out[(out["time"] < "2004-02-01") | (out["time"] > "2004-03-01")]
    assert not outside_window["TWS_t_masked"].any()


def test_exception_rate_zero_masks_every_candidate(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-03-01", exception_rate=0.0)
    out = apply_masking(synthetic_df, scenario)
    in_window = out[(out["time"] >= "2004-02-01") & (out["time"] <= "2004-03-01")]
    assert in_window["TWS_t_masked"].all()


def test_exception_rate_near_one_masks_almost_nothing() -> None:
    # Needs a larger sample to make the statistical claim meaningful.
    big_df = _make_synthetic_df(n_locations=200, n_months=3)
    scenario = _abrupt_scenario("2004-02-01", "2004-02-01", exception_rate=0.999)
    out = apply_masking(big_df, scenario, seed=123)
    in_window = out[out["time"] == "2004-02-01"]
    assert in_window["TWS_t_masked"].mean() < 0.05


def test_affected_locations_restricts_masking(synthetic_df: pd.DataFrame) -> None:
    # Only location (1.0, 2.0) should ever be masked.
    scenario = _abrupt_scenario(
        "2004-02-01", "2004-03-01", affected_locations=("1.0_2.0",)
    )
    out = apply_masking(synthetic_df, scenario)
    masked = out[out["TWS_t_masked"]]
    assert (masked["lat"] == 1.0).all() and (masked["lon"] == 2.0).all()
    # the other two locations, even within the window, are untouched
    other_in_window = out[
        (out["time"] >= "2004-02-01") & (out["time"] <= "2004-03-01") & (out["lat"] != 1.0)
    ]
    assert not other_in_window["TWS_t_masked"].any()


def test_derived_columns_also_nulled(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-02-01")
    out = apply_masking(synthetic_df, scenario, derived_columns=["SPEI_01_t"])
    masked = out[out["TWS_t_masked"]]
    assert masked["SPEI_01_t"].isna().all()
    # SOIL_MOISTURE_t was not listed as derived, so it must stay populated
    assert masked["SOIL_MOISTURE_t"].isna().sum() == 0


def test_tws_t_masked_always_equals_isna(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-04-01", exception_rate=0.2)
    out = apply_masking(synthetic_df, scenario, seed=1)
    assert (out["TWS_t_masked"] == out["TWS_t"].isna()).all()


def test_determinism_given_fixed_seed(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-05-01", exception_rate=0.3)
    out_a = apply_masking(synthetic_df, scenario, seed=7)
    out_b = apply_masking(synthetic_df, scenario, seed=7)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_different_seeds_can_diverge_exception_pattern() -> None:
    big_df = _make_synthetic_df(n_locations=200, n_months=1)
    scenario = _abrupt_scenario("2004-01-01", "2004-01-01", exception_rate=0.3)
    out_a = apply_masking(big_df, scenario, seed=1)
    out_b = apply_masking(big_df, scenario, seed=2)
    assert not out_a["TWS_t_masked"].equals(out_b["TWS_t_masked"])


def test_never_adds_or_masks_nonexistent_rows(synthetic_df: pd.DataFrame) -> None:
    # A row missing entirely from a given month (real grid irregularity)
    # must not appear after masking is applied.
    sparse = synthetic_df.drop(index=synthetic_df.index[3])  # drop one row
    scenario = _abrupt_scenario("2004-01-01", "2004-06-01")
    out = apply_masking(sparse, scenario)
    assert len(out) == len(sparse)
    assert set(zip(out["time"], out["lat"], out["lon"])) == set(
        zip(sparse["time"], sparse["lat"], sparse["lon"])
    )


# --- MaskingScenario validation -------------------------------------------


def test_scenario_requires_month_start_dates() -> None:
    with pytest.raises(ValidationError, match="first of the month"):
        MaskingScenario(
            blackout_start=date(2004, 2, 15),
            blackout_end=date(2004, 3, 1),
            streak_length=2,
            source_rationale="bad date",
        )


def test_scenario_blackout_end_before_start_raises() -> None:
    with pytest.raises(ValidationError, match="before blackout_start"):
        MaskingScenario(
            blackout_start=date(2004, 3, 1),
            blackout_end=date(2004, 2, 1),
            streak_length=1,
            source_rationale="bad order",
        )


def test_scenario_streak_length_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="streak_length"):
        MaskingScenario(
            blackout_start=date(2004, 2, 1),
            blackout_end=date(2004, 3, 1),
            streak_length=5,  # actual span is 2 months
            source_rationale="bad streak",
        )


def test_scenario_exception_rate_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="exception_rate"):
        MaskingScenario(
            blackout_start=date(2004, 2, 1),
            blackout_end=date(2004, 2, 1),
            streak_length=1,
            exception_rate=1.0,
            source_rationale="bad rate",
        )


def test_unimplemented_transition_pattern_raises(synthetic_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-02-01", transition_pattern="ramp_in")
    with pytest.raises(NotImplementedError, match="ramp_in"):
        apply_masking(synthetic_df, scenario)


def test_scenario_is_frozen() -> None:
    scenario = _abrupt_scenario("2004-02-01", "2004-02-01")
    with pytest.raises(ValidationError):
        scenario.exception_rate = 0.5  # type: ignore[misc]
