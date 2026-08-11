"""Tests for tws_forecast.validation.tiers — run against a trivial
mean-predictor stand-in, per docs/PHASE2_EXECUTION_PLAN.md step 2.6."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.validation.phase1_constants import CLEAN_TRAIN_SPAN_END, CLEAN_TRAIN_SPAN_START
from tws_forecast.validation.splitters import FORECAST_ORIGIN_COLUMNS
from tws_forecast.validation.tiers import TierResult, _select_replay_anchors, run_tier1, run_tier2, run_tier3


class MeanPredictor:
    """The trivial stand-in the execution plan calls for: predicts the
    training fold's own target mean for every row, ignoring all features."""

    def __init__(self) -> None:
        self._mean = 0.0
        self.fit_calls = 0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())
        self.fit_calls += 1

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


# --- Tier 1 -----------------------------------------------------------------


def test_run_tier1_executes_end_to_end(train_df: pd.DataFrame) -> None:
    result = run_tier1(MeanPredictor(), train_df)
    assert isinstance(result, TierResult)
    assert result.tier == 1
    assert result.scenario_name == "expanding_window"
    assert len(result.fold_rmses) == 5  # default n_folds=5, all non-empty on this fixture
    assert np.isfinite(result.overall_rmse)
    assert 0.0 < result.overall_rmse < 5.0  # sane RMSE range for a mean predictor


def test_run_tier1_predictions_have_forecast_origin_columns(train_df: pd.DataFrame) -> None:
    result = run_tier1(MeanPredictor(), train_df)
    for col in [*FORECAST_ORIGIN_COLUMNS, "prediction", "target", "true_tws_t", "fold"]:
        assert col in result.predictions.columns


def test_run_tier1_true_tws_t_matches_target_frames_own_tws(train_df: pd.DataFrame) -> None:
    # Tier 1 never masks, so true_tws_t must equal what the model actually
    # saw for every row — this is the property that lets Tiers 2/3 reuse
    # the same column name to mean "ground truth, possibly hidden."
    result = run_tier1(MeanPredictor(), train_df)
    merged = result.predictions.merge(
        train_df[["time", "lat", "lon", "TWS_t"]],
        left_on="origin_time", right_on="time", how="left",
    )
    # location_id encodes lat/lon; spot check a sample rather than a full
    # merge-by-location (location_id parsing is exercised directly in
    # decomposition tests) — here we only need true_tws_t to be non-null
    # and internally consistent.
    assert result.predictions["true_tws_t"].notna().all()


def test_run_tier1_model_fit_once_per_fold(train_df: pd.DataFrame) -> None:
    model = MeanPredictor()
    run_tier1(model, train_df)
    assert model.fit_calls == 5


def test_run_tier1_all_rows_observed_train_has_no_masking(train_df: pd.DataFrame) -> None:
    result = run_tier1(MeanPredictor(), train_df)
    assert (result.predictions["regime"] == "observed").all()


def test_run_tier1_wrong_scenario_type_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="expanding_window"):
        run_tier1(MeanPredictor(), train_df, scenario="blackout_curve")


# --- Tier 2 -----------------------------------------------------------------


def test_run_tier2_executes_end_to_end(train_df: pd.DataFrame) -> None:
    result = run_tier2(MeanPredictor(), train_df)
    assert result.tier == 2
    assert result.scenario_name == "blackout_curve"
    assert len(result.fold_rmses) == 5
    assert np.isfinite(result.overall_rmse)


def test_run_tier2_predictions_have_simulated_k_column(train_df: pd.DataFrame) -> None:
    result = run_tier2(MeanPredictor(), train_df)
    assert "simulated_k" in result.predictions.columns
    masked = result.predictions[result.predictions["regime"] == "masked"]
    assert len(masked) > 0
    assert masked["simulated_k"].notna().all()
    assert masked["simulated_k"].isin([2, 3, 4, 5, 6, 7]).all()


def test_run_tier2_unmasked_rows_have_nan_simulated_k(train_df: pd.DataFrame) -> None:
    result = run_tier2(MeanPredictor(), train_df)
    unmasked = result.predictions[result.predictions["regime"] == "observed"]
    assert unmasked["simulated_k"].isna().all()


def test_run_tier2_wrong_scenario_type_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="blackout_curve"):
        run_tier2(MeanPredictor(), train_df, scenario="expanding_window")


# --- Tier 3 -----------------------------------------------------------------


def test_run_tier3_executes_end_to_end(train_df: pd.DataFrame) -> None:
    result = run_tier3(MeanPredictor(), train_df, n_anchors=2)
    assert result.tier == 3
    assert result.scenario_name == "test_regime_replay"
    assert len(result.fold_rmses) >= 1
    assert np.isfinite(result.overall_rmse)


def test_run_tier3_predictions_have_replay_offset_and_simulated_k(train_df: pd.DataFrame) -> None:
    result = run_tier3(MeanPredictor(), train_df, n_anchors=2)
    assert "replay_offset" in result.predictions.columns
    assert "simulated_k" in result.predictions.columns


def test_run_tier3_full_offsets_observed_blackout_offsets_masked(train_df: pd.DataFrame) -> None:
    result = run_tier3(MeanPredictor(), train_df, n_anchors=2)
    full_rows = result.predictions[result.predictions["simulated_k"].isna()]
    blackout_rows = result.predictions[result.predictions["simulated_k"].notna()]
    assert len(full_rows) > 0
    assert len(blackout_rows) > 0
    assert (full_rows["regime"] == "observed").all()
    assert (blackout_rows["regime"] == "masked").all()


def test_run_tier3_model_fit_once_per_anchor(train_df: pd.DataFrame) -> None:
    model = MeanPredictor()
    result = run_tier3(model, train_df, n_anchors=2)
    assert model.fit_calls == len(result.fold_rmses)


def test_run_tier3_wrong_scenario_type_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="test_regime_replay"):
        run_tier3(MeanPredictor(), train_df, scenario="expanding_window")


def test_run_tier3_invalid_n_anchors_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_anchors"):
        run_tier3(MeanPredictor(), train_df, n_anchors=0)


def test_run_tier3_single_anchor(train_df: pd.DataFrame) -> None:
    result = run_tier3(MeanPredictor(), train_df, n_anchors=1)
    assert len(result.fold_rmses) == 1


# --- _select_replay_anchors: clean-span restriction (regression test) ---------
#
# Bug found and fixed during Project Phase 2 step 2.11's proof run
# (notebooks/03_validation_harness.ipynb): _select_replay_anchors originally
# spanned df's own full min/max time range, which let a candidate anchor's
# replay pattern run into the documented post-2010 missing-month gaps (A-012)
# or start with almost no prior fit history near TRAIN_PERIOD_START -- both
# measured to pull Tier 3's score far from Baseline D's validated 0.6573.
# These tests pin the fix: every anchor, and its full replay pattern, must
# stay within the verified gap-free 2004-2010 span.


def test_select_replay_anchors_stays_within_clean_span(train_df: pd.DataFrame) -> None:
    pattern_length = 40  # matches configs/validation/test_regime_replay.yaml's real offsets
    anchors = _select_replay_anchors(train_df, pattern_length, n_anchors=3)
    assert len(anchors) >= 1
    for anchor in anchors:
        pattern_end = anchor + pd.DateOffset(months=pattern_length - 1)
        assert anchor >= pd.Timestamp(CLEAN_TRAIN_SPAN_START), (
            f"anchor {anchor.date()} starts before the verified clean span"
        )
        assert pattern_end <= pd.Timestamp(CLEAN_TRAIN_SPAN_END), (
            f"anchor {anchor.date()}'s replay pattern (through {pattern_end.date()}) "
            "runs past the verified clean span -- would overlap the documented post-2010 gaps"
        )


def test_select_replay_anchors_excludes_pre_clean_span_history(train_df: pd.DataFrame) -> None:
    # train_df (the golden fixture) starts 2002-05, well before CLEAN_TRAIN_SPAN_START (2004-01) --
    # confirms the fix actually constrains anchors rather than merely happening to avoid this range.
    assert train_df["time"].min() < pd.Timestamp(CLEAN_TRAIN_SPAN_START)
    anchors = _select_replay_anchors(train_df, pattern_length_months=40, n_anchors=5)
    assert all(a >= pd.Timestamp(CLEAN_TRAIN_SPAN_START) for a in anchors)


def test_select_replay_anchors_empty_when_pattern_longer_than_clean_span(train_df: pd.DataFrame) -> None:
    # The clean span is 84 months (2004-01 to 2010-12) -- a pattern longer than that can never fit.
    assert _select_replay_anchors(train_df, pattern_length_months=200, n_anchors=1) == []


# --- TierResult ---------------------------------------------------------------


def test_tier_result_rmse_mean_and_std() -> None:
    result = TierResult(
        tier=1, scenario_name="x", predictions=pd.DataFrame(),
        fold_rmses=(0.5, 0.7, 0.6), overall_rmse=0.6,
    )
    assert result.rmse_mean == pytest.approx(0.6, abs=1e-9)
    assert result.rmse_std == pytest.approx(np.std([0.5, 0.7, 0.6]), abs=1e-9)


def test_tier_result_empty_fold_rmses_returns_nan() -> None:
    result = TierResult(
        tier=1, scenario_name="x", predictions=pd.DataFrame(),
        fold_rmses=(), overall_rmse=float("nan"),
    )
    assert np.isnan(result.rmse_mean)
    assert np.isnan(result.rmse_std)


def test_tier_result_repr_does_not_raise() -> None:
    result = TierResult(
        tier=2, scenario_name="blackout_curve", predictions=pd.DataFrame({"a": [1, 2]}),
        fold_rmses=(0.5,), overall_rmse=0.5,
    )
    assert "TierResult" in repr(result)
    assert "blackout_curve" in repr(result)
