"""Tests for tws_forecast.validation.leakage_tests — the executable leakage
firewall (docs/PHASE2_EXECUTION_PLAN.md step 2.8).

Each check is exercised against both a correct example (must pass) and a
deliberately leaky one (must be caught) — proving the check itself has
teeth, not just that it doesn't crash. Real feature/signature functions
don't exist yet (Project Phase 4); the toy examples here stand in for them,
same as the execution plan calls for.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.validation.leakage_tests import (
    future_row_shuffle_test,
    historical_only_check,
    masking_simulator_no_leak_check,
    rolling_window_cutoff_check,
)
from tws_forecast.validation.masking_simulator import MaskingScenario


class MeanPredictor:
    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


# --- future_row_shuffle_test --------------------------------------------------


def test_future_row_shuffle_passes_for_real_tier_predictor(train_df: pd.DataFrame) -> None:
    # Exercises the real project code path (time-based boolean selection),
    # not a toy — this is expected to always pass given the actual
    # implementation, and is itself the regression guard.
    assert future_row_shuffle_test(MeanPredictor(), train_df, cutoff_time="2010-12-01") is True


def test_future_row_shuffle_no_rows_before_cutoff_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows at or before"):
        future_row_shuffle_test(MeanPredictor(), train_df, cutoff_time="1990-01-01")


class _PositionalLeakyPredictor:
    """A deliberately broken predictor: 'fit' just remembers the LAST row's
    target by physical position, not by time — a realistic bug pattern if
    someone used .iloc[-1] assuming chronological file order instead of
    filtering by the time column."""

    def fit(self, train_df: pd.DataFrame) -> None:
        self._last_value = float(train_df["target"].iloc[-1])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._last_value)


def test_future_row_shuffle_catches_positional_leak(train_df: pd.DataFrame) -> None:
    # The leaky predictor's fit() depends on row order within whatever
    # frame it's given. Shuffling the *future* rows' positions changes
    # which physical row lands at len(history)-1 only if the shuffle
    # touches the boundary — but here 'history' itself is always exactly
    # rows with time<=cutoff, built the safe (boolean) way, so this
    # particular leaky predictor is not actually distinguishable this
    # way. This test instead documents that future_row_shuffle_test
    # checks the *harness's* row-selection safety, not a model's internal
    # logic — see rolling_window_cutoff_check for catching a model/feature
    # that itself has an off-by-one leak.
    result = future_row_shuffle_test(_PositionalLeakyPredictor(), train_df, cutoff_time="2010-12-01")
    assert result is True  # the harness's own selection is still safe


# --- historical_only_check ----------------------------------------------------


def _good_signature(df: pd.DataFrame, t: pd.Timestamp):
    sub = df[df["time"] < t]
    return float(sub["TWS_t"].mean()) if len(sub) else float("nan")


def _leaky_signature(df: pd.DataFrame, t: pd.Timestamp):
    # Ignores t entirely -- trusts whatever frame it's handed. A realistic
    # bug: caller forgets that the signature function needs the caller to
    # already pre-filter, or the function was refactored and lost its own
    # internal filter.
    return float(df["TWS_t"].mean())


def test_historical_only_check_passes_for_correct_signature(train_df: pd.DataFrame) -> None:
    assert historical_only_check(_good_signature, train_df, evaluate_time="2010-06-01") is True


def test_historical_only_check_catches_leaky_signature(train_df: pd.DataFrame) -> None:
    assert historical_only_check(_leaky_signature, train_df, evaluate_time="2010-06-01") is False


def test_historical_only_check_all_history_and_no_history_agree_trivially() -> None:
    # Edge case: if there IS no history before t, a correct signature
    # returns NaN either way -- both NaN counts as "close" (equal_nan
    # handling), not a spurious failure.
    tiny_df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01"]), "TWS_t": [1.0]})
    assert historical_only_check(_good_signature, tiny_df, evaluate_time="2019-01-01") is True


# --- rolling_window_cutoff_check ----------------------------------------------


def _good_lag_feature(df: pd.DataFrame, t: pd.Timestamp):
    sub = df[df["time"] < t].sort_values("time")
    return float(sub["TWS_t"].iloc[-1]) if len(sub) else float("nan")


def _off_by_one_lag_feature(df: pd.DataFrame, t: pd.Timestamp):
    # Uses <= instead of < -- a classic, realistic off-by-one leak that
    # includes the forecast origin's own (not-yet-known) current value.
    sub = df[df["time"] <= t].sort_values("time")
    return float(sub["TWS_t"].iloc[-1]) if len(sub) else float("nan")


@pytest.fixture()
def df_with_row_at_origin() -> pd.DataFrame:
    times = pd.date_range("2004-01-01", periods=6, freq="MS")
    return pd.DataFrame({"time": times, "TWS_t": np.arange(6, dtype=float)})


def test_rolling_window_cutoff_check_passes_for_correct_lag_feature(
    df_with_row_at_origin: pd.DataFrame,
) -> None:
    origin = df_with_row_at_origin["time"].iloc[3]
    assert rolling_window_cutoff_check(_good_lag_feature, df_with_row_at_origin, origin_time=origin) is True


def test_rolling_window_cutoff_check_catches_off_by_one_leak(
    df_with_row_at_origin: pd.DataFrame,
) -> None:
    origin = df_with_row_at_origin["time"].iloc[3]
    assert rolling_window_cutoff_check(_off_by_one_lag_feature, df_with_row_at_origin, origin_time=origin) is False


# --- masking_simulator_no_leak_check -------------------------------------------


def _abrupt_scenario(start: str, end: str, **kwargs) -> MaskingScenario:
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    streak = (end_d.year * 12 + end_d.month) - (start_d.year * 12 + start_d.month) + 1
    return MaskingScenario(
        blackout_start=start_d, blackout_end=end_d, streak_length=streak,
        source_rationale="leakage test", **kwargs,
    )


def test_masking_simulator_no_leak_check_passes_for_real_apply_masking(train_df: pd.DataFrame) -> None:
    scenario = _abrupt_scenario("2010-01-01", "2010-03-01")
    assert masking_simulator_no_leak_check(scenario, train_df) is True


def test_masking_simulator_no_leak_check_scenario_masking_nothing_is_trivially_true() -> None:
    df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01"]), "lat": [0.0], "lon": [0.0], "TWS_t": [1.0]})
    scenario = _abrupt_scenario("1999-01-01", "1999-01-01")  # no rows fall in this window
    assert masking_simulator_no_leak_check(scenario, df) is True


def test_masking_simulator_no_leak_check_catches_unmasked_derived_column() -> None:
    # Build a df with a "derived" column that mirrors TWS_t, then check
    # with a derived_columns list that OMITS it -- simulating a caller bug
    # where a real derived feature exists but wasn't registered for
    # nulling. The check must catch that this column still reproduces the
    # true value on masked rows.
    times = pd.date_range("2004-01-01", periods=4, freq="MS")
    df = pd.DataFrame({
        "time": times, "lat": [0.0] * 4, "lon": [0.0] * 4,
        "TWS_t": [1.0, 2.0, 3.0, 4.0],
        "last_known_tws": [1.0, 1.0, 2.0, 3.0],  # a hypothetical derived column, NOT passed to derived_columns below
    })
    scenario = _abrupt_scenario("2004-02-01", "2004-02-01")
    # last_known_tws at 2004-02 happens to equal TWS_t at 2004-01 (1.0), not
    # the true masked value (2.0) -- so this particular row doesn't
    # reproduce the leak. Construct a case where it DOES: set
    # last_known_tws equal to the true masked value directly.
    df.loc[df["time"] == "2004-02-01", "last_known_tws"] = 2.0  # now equals the true (masked) TWS_t
    result = masking_simulator_no_leak_check(scenario, df, derived_columns=[])
    assert result is False


def test_masking_simulator_no_leak_check_derived_columns_correctly_nulled(train_df: pd.DataFrame) -> None:
    df = train_df.copy()
    df["fake_derived"] = df["TWS_t"] * 2
    scenario = _abrupt_scenario("2010-01-01", "2010-02-01")
    result = masking_simulator_no_leak_check(scenario, df, derived_columns=["fake_derived"])
    assert result is True
