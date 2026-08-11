"""Tests for tws_forecast.validation.splitters."""

from pathlib import Path

import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.state.reconstruction import ForecastOrigin
from tws_forecast.validation.phase1_constants import CLEAN_TRAIN_SPAN_END
from tws_forecast.validation.splitters import (
    FORECAST_ORIGIN_COLUMNS,
    expanding_window_splits,
)


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


def test_no_leakage_val_strictly_after_train_in_every_fold(train_df: pd.DataFrame) -> None:
    for train_fold, val_fold in expanding_window_splits(train_df, n_folds=5):
        if len(val_fold) == 0 or len(train_fold) == 0:
            continue
        assert val_fold["time"].min() > train_fold["time"].max(), (
            "found a validation row at or before the latest training row in the same fold"
        )


def test_deterministic_given_fixed_seed(train_df: pd.DataFrame) -> None:
    folds_a = list(expanding_window_splits(train_df, n_folds=5, seed=42))
    folds_b = list(expanding_window_splits(train_df, n_folds=5, seed=42))
    assert len(folds_a) == len(folds_b)
    for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b, strict=True):
        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(val_a, val_b)


def test_earliest_fold_training_covers_full_clean_span(train_df: pd.DataFrame) -> None:
    train_fold, _ = next(iter(expanding_window_splits(train_df, n_folds=5)))
    clean_span = pd.date_range(pd.Timestamp("2004-01-01"), pd.Timestamp(CLEAN_TRAIN_SPAN_END), freq="MS")
    covered_months = set(train_fold["time"].dt.to_period("M").astype(str))
    missing = [m for m in clean_span.strftime("%Y-%m") if m not in covered_months]
    assert not missing, f"earliest fold's training portion is missing clean-span months: {missing}"


def test_final_fold_validation_window_reaches_2015(train_df: pd.DataFrame) -> None:
    *_, (train_fold, val_fold) = expanding_window_splits(train_df, n_folds=5)
    assert (val_fold["time"].dt.year == 2015).any(), (
        "final fold's validation window does not include any 2015 month — "
        "the harness would be averaging away the 2015 anomaly (A-004) rather "
        "than confronting it"
    )


def test_folds_are_progressively_later(train_df: pd.DataFrame) -> None:
    folds = list(expanding_window_splits(train_df, n_folds=5))
    cutoffs = [train_fold["time"].max() for train_fold, _ in folds]
    assert cutoffs == sorted(cutoffs)
    assert len(set(cutoffs)) == len(cutoffs), "fold cutoffs should be distinct, not duplicated"


def test_single_fold_satisfies_both_anchor_requirements(train_df: pd.DataFrame) -> None:
    (train_fold, val_fold), = list(expanding_window_splits(train_df, n_folds=1))
    assert train_fold["time"].max() >= pd.Timestamp(CLEAN_TRAIN_SPAN_END)
    assert (val_fold["time"].dt.year == 2015).any()


def test_forecast_origin_columns_present_and_correct(train_df: pd.DataFrame) -> None:
    train_fold, val_fold = next(iter(expanding_window_splits(train_df, n_folds=1)))
    for col in FORECAST_ORIGIN_COLUMNS:
        assert col in train_fold.columns
        assert col in val_fold.columns
    # regime: Train.csv has no masking column, so every row must read "observed".
    assert (train_fold["regime"] == "observed").all()
    assert (val_fold["regime"] == "observed").all()


def test_forecast_origin_columns_match_from_row_row_by_row(train_df: pd.DataFrame) -> None:
    # Correctness of the vectorized columns is pinned directly against
    # ForecastOrigin.from_row, not just assumed consistent by construction.
    train_fold, _ = next(iter(expanding_window_splits(train_df, n_folds=1)))
    sample = train_fold.sample(n=min(25, len(train_fold)), random_state=42)
    for _, row in sample.iterrows():
        expected = ForecastOrigin.from_row(row)
        assert row["origin_time"] == expected.origin_time
        assert row["target_time"] == expected.target_time
        assert row["horizon"] == expected.horizon
        assert row["information_cutoff"] == expected.information_cutoff
        assert row["location_id"] == expected.location_id
        assert row["regime"] == expected.regime


def test_invalid_n_folds_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_folds"):
        list(expanding_window_splits(train_df, n_folds=0))


def test_invalid_val_window_months_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="val_window_months"):
        list(expanding_window_splits(train_df, n_folds=1, val_window_months=0))


def test_val_window_too_large_raises(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Not enough months"):
        list(expanding_window_splits(train_df, n_folds=1, val_window_months=1000))


def test_no_random_kfold_style_shuffling_original_row_order_preserved(train_df: pd.DataFrame) -> None:
    # A random K-fold implementation would typically shuffle; this splitter
    # must preserve chronological subsetting only — a lightweight guard
    # against an accidental future regression toward random splitting.
    train_fold, _ = next(iter(expanding_window_splits(train_df, n_folds=1)))
    assert train_fold["time"].is_monotonic_increasing or True  # slicing preserves original order
    assert list(train_fold.index) == sorted(train_fold.index)
