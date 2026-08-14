"""Tests for tws_forecast.models.baselines — Project Phase 3's six
state-aware reference predictors.

Per the Project Phase 3 handoff §3.2, every class gets: fit-then-predict
returns an array of the right length with no NaNs (a predictor that returns
NaN silently breaks every downstream RMSE calculation); determinism;
the never-observed-location fallback path (LastKnownStatePredictor /
SeasonalClimatologyPredictor); OraclePersistencePredictor returns exactly
TWS_t on observed rows and the documented fallback on masked rows; and a
real integration check — each class passed to run_tier1/run_tier2/run_tier3
against the golden fixture, completing without error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.models.baselines import (
    GlobalMeanPredictor,
    HybridPersistencePredictor,
    LastKnownStatePredictor,
    OraclePersistencePredictor,
    RidgeBaselinePredictor,
    SeasonalClimatologyPredictor,
)
from tws_forecast.validation.tiers import run_tier1, run_tier2, run_tier3

ALL_PREDICTOR_CLASSES = [
    GlobalMeanPredictor,
    OraclePersistencePredictor,
    LastKnownStatePredictor,
    SeasonalClimatologyPredictor,
    HybridPersistencePredictor,
    RidgeBaselinePredictor,
]


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


@pytest.fixture()
def small_synthetic_frame() -> pd.DataFrame:
    """A tiny, hand-built two-location, two-month frame — enough to fit any
    of the six baselines without touching the golden fixture's scale."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2004-01-01", "2004-02-01", "2004-01-01", "2004-02-01"]
            ),
            "lat": [10.5, 10.5, 20.5, 20.5],
            "lon": [30.5, 30.5, 40.5, 40.5],
            "TWS_t": [1.0, 1.2, 2.0, 2.2],
            "SPEI_01_t": [0.1, 0.2, -0.1, -0.2],
            "SPEI_03_t": [0.1, 0.2, -0.1, -0.2],
            "SPEI_06_t": [0.1, 0.2, -0.1, -0.2],
            "SPEI_12_t": [0.1, 0.2, -0.1, -0.2],
            "SOIL_MOISTURE_t": [0.5, 0.6, -0.5, -0.6],
            "month_sin": [0.5, 0.87, 0.5, 0.87],
            "month_cos": [0.87, 0.5, 0.87, 0.5],
            "target": [1.2, 1.3, 2.2, 2.3],
        }
    )


@pytest.fixture()
def unseen_location_frame() -> pd.DataFrame:
    """One predict-time row at a location absent from ``small_synthetic_frame``."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2004-03-01"]),
            "lat": [99.5],
            "lon": [99.5],
            "TWS_t": [np.nan],
            "SPEI_01_t": [0.0],
            "SPEI_03_t": [0.0],
            "SPEI_06_t": [0.0],
            "SPEI_12_t": [0.0],
            "SOIL_MOISTURE_t": [0.0],
            "month_sin": [0.0],
            "month_cos": [1.0],
        }
    )


# --- Generic contract: every class, fit then predict, no NaN, right length --


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_fit_predict_returns_array_no_nans_right_length(
    predictor_cls, train_df: pd.DataFrame
) -> None:
    cut = train_df["time"].quantile(0.7)
    train_fold = train_df[train_df["time"] <= cut]
    val_fold = train_df[train_df["time"] > cut].copy()

    model = predictor_cls()
    model.fit(train_fold)
    preds = model.predict(val_fold)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(val_fold),)
    assert not np.isnan(preds).any()


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_predict_handles_mixed_observed_and_masked_rows(
    predictor_cls, train_df: pd.DataFrame
) -> None:
    cut = train_df["time"].quantile(0.7)
    train_fold = train_df[train_df["time"] <= cut]
    val_fold = train_df[train_df["time"] > cut].copy()

    rng = np.random.default_rng(0)
    mask_idx = val_fold.sample(frac=0.5, random_state=1).index
    val_fold.loc[mask_idx, "TWS_t"] = np.nan

    model = predictor_cls()
    model.fit(train_fold)
    preds = model.predict(val_fold)

    assert not np.isnan(preds).any()
    assert preds.shape == (len(val_fold),)


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_predict_does_not_mutate_input_frame(
    predictor_cls, train_df: pd.DataFrame
) -> None:
    cut = train_df["time"].quantile(0.7)
    train_fold = train_df[train_df["time"] <= cut]
    val_fold = train_df[train_df["time"] > cut].copy()
    val_fold_copy = val_fold.copy(deep=True)

    model = predictor_cls()
    model.fit(train_fold)
    model.predict(val_fold)

    pd.testing.assert_frame_equal(val_fold, val_fold_copy)


# --- Determinism ---------------------------------------------------------


def test_global_mean_predictor_deterministic(train_df: pd.DataFrame) -> None:
    m1, m2 = GlobalMeanPredictor(), GlobalMeanPredictor()
    m1.fit(train_df)
    m2.fit(train_df)
    np.testing.assert_array_equal(m1.predict(train_df), m2.predict(train_df))


def test_ridge_baseline_predictor_deterministic(train_df: pd.DataFrame) -> None:
    m1, m2 = RidgeBaselinePredictor(seed=42), RidgeBaselinePredictor(seed=42)
    m1.fit(train_df)
    m2.fit(train_df)
    np.testing.assert_allclose(m1.predict(train_df), m2.predict(train_df))


# --- Never-observed-location fallback -------------------------------------


def test_last_known_state_falls_back_to_global_mean_for_unseen_location(
    small_synthetic_frame: pd.DataFrame, unseen_location_frame: pd.DataFrame
) -> None:
    model = LastKnownStatePredictor()
    model.fit(small_synthetic_frame)
    preds = model.predict(unseen_location_frame)
    assert not np.isnan(preds).any()
    assert preds[0] == pytest.approx(small_synthetic_frame["target"].mean())


def test_seasonal_climatology_falls_back_to_global_mean_for_unseen_combo(
    small_synthetic_frame: pd.DataFrame, unseen_location_frame: pd.DataFrame
) -> None:
    model = SeasonalClimatologyPredictor()
    model.fit(small_synthetic_frame)
    preds = model.predict(unseen_location_frame)
    assert not np.isnan(preds).any()
    assert preds[0] == pytest.approx(small_synthetic_frame["target"].mean())


# --- LastKnownStatePredictor never reads the predict-time TWS_t ------------


def test_last_known_state_ignores_predict_time_tws_t(
    small_synthetic_frame: pd.DataFrame,
) -> None:
    """The defining distinction from HybridPersistencePredictor (Baseline D):
    even when a row in the predict-time frame carries a real, observed
    TWS_t, LastKnownStatePredictor must not use it — only its fit()-time
    history."""
    model = LastKnownStatePredictor()
    # Location (10.5, 30.5) only: Jan 2004 (TWS_t=1.0), Feb 2004 (TWS_t=1.2).
    # fit()-time last-known value for this location is therefore 1.2 (Feb).
    model.fit(small_synthetic_frame.iloc[:2])

    # A March row at the SAME location, with a real (but different) TWS_t.
    march_row = pd.DataFrame(
        {
            "time": pd.to_datetime(["2004-03-01"]),
            "lat": [10.5],
            "lon": [30.5],
            "TWS_t": [999.0],  # deliberately implausible, to catch any leak
        }
    )
    preds = model.predict(march_row)
    # Must equal the fit-time last-known value (1.2, Feb 2004), NOT 999.0.
    assert preds[0] == pytest.approx(1.2)


# --- OraclePersistencePredictor: exact TWS_t on observed, documented fallback on masked --


def test_oracle_persistence_returns_exact_tws_t_when_observed() -> None:
    train = pd.DataFrame({"TWS_t": [1.0, 2.0, 3.0], "target": [1.1, 2.1, 3.1]})
    predict_df = pd.DataFrame({"TWS_t": [5.0, 6.0]})

    model = OraclePersistencePredictor()
    model.fit(train)
    preds = model.predict(predict_df)

    np.testing.assert_array_equal(preds, [5.0, 6.0])


def test_oracle_persistence_falls_back_to_global_mean_on_masked_rows() -> None:
    train = pd.DataFrame({"TWS_t": [1.0, 2.0, 3.0], "target": [10.0, 20.0, 30.0]})
    predict_df = pd.DataFrame({"TWS_t": [5.0, np.nan, np.nan]})

    model = OraclePersistencePredictor()
    model.fit(train)
    preds = model.predict(predict_df)

    expected_fallback = train["target"].mean()  # 20.0
    np.testing.assert_array_equal(preds, [5.0, expected_fallback, expected_fallback])


def test_oracle_persistence_mixed_regime_fixture() -> None:
    """A hand-built fixture with both regimes present, verified precisely."""
    train = pd.DataFrame({"TWS_t": [0.0, 0.0], "target": [4.0, 6.0]})  # mean target = 5.0
    predict_df = pd.DataFrame({"TWS_t": [1.5, np.nan, -2.0, np.nan]})

    model = OraclePersistencePredictor()
    model.fit(train)
    preds = model.predict(predict_df)

    np.testing.assert_array_equal(preds, [1.5, 5.0, -2.0, 5.0])


# --- Predictor protocol / harness integration check ------------------------


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_predictor_runs_through_tier1(predictor_cls, train_df: pd.DataFrame) -> None:
    result = run_tier1(predictor_cls(), train_df)
    assert np.isfinite(result.overall_rmse)


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_predictor_runs_through_tier2(predictor_cls, train_df: pd.DataFrame) -> None:
    result = run_tier2(predictor_cls(), train_df)
    assert np.isfinite(result.overall_rmse)


@pytest.mark.parametrize("predictor_cls", ALL_PREDICTOR_CLASSES)
def test_predictor_runs_through_tier3(predictor_cls, train_df: pd.DataFrame) -> None:
    result = run_tier3(predictor_cls(), train_df, n_anchors=2)
    assert np.isfinite(result.overall_rmse)


# --- HybridPersistencePredictor reproduces pure persistence on always-observed data --


def test_hybrid_persistence_reproduces_persistence_when_never_masked(
    train_df: pd.DataFrame,
) -> None:
    """On Train.csv (TWS_t always observed), HybridPersistencePredictor's
    predict() must reduce to exactly the row's own TWS_t — the ffill should
    never need to reach past the row itself."""
    cut = train_df["time"].quantile(0.7)
    train_fold = train_df[train_df["time"] <= cut]
    val_fold = train_df[train_df["time"] > cut].copy()

    model = HybridPersistencePredictor()
    model.fit(train_fold)
    preds = model.predict(val_fold)

    np.testing.assert_allclose(preds, val_fold["TWS_t"].to_numpy(), rtol=1e-10)


# --- LastKnownStatePredictor vs. HybridPersistencePredictor: the documented divergence --


def test_last_known_state_diverges_from_hybrid_when_tws_t_observed(
    train_df: pd.DataFrame,
) -> None:
    """Confirms the module-docstring claim empirically: on a val fold where
    TWS_t is fully observed, HybridPersistencePredictor should track it
    (near-zero error against it), while LastKnownStatePredictor should not
    — it never reads TWS_t at predict time at all."""
    cut = train_df["time"].quantile(0.7)
    train_fold = train_df[train_df["time"] <= cut]
    val_fold = train_df[train_df["time"] > cut].copy()

    hybrid = HybridPersistencePredictor()
    hybrid.fit(train_fold)
    hybrid_preds = hybrid.predict(val_fold)

    last_known = LastKnownStatePredictor()
    last_known.fit(train_fold)
    last_known_preds = last_known.predict(val_fold)

    # Hybrid must match TWS_t exactly (see test above); last-known-state
    # predictions must NOT generally match TWS_t (they're fixed at fit time).
    assert np.allclose(hybrid_preds, val_fold["TWS_t"].to_numpy(), rtol=1e-10)
    assert not np.allclose(last_known_preds, val_fold["TWS_t"].to_numpy(), rtol=1e-10)
