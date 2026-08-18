"""Tests for tws_forecast.features.targets — Project Phase 4 step 4.7, per
docs/PHASE4_EXECUTION_PLAN.md §4.7.

The critical property this module's own docstring calls out: round-trip
invertibility (``inverse(forward(df), df) == df["target"]`` exactly) for
every one of the five transforms, on a fixture containing both observed and
masked ``TWS_t`` rows -- a bug here would silently corrupt every downstream
RMSE without necessarily looking wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.features.targets import (
    MIN_STD,
    TARGET_TRANSFORMS,
    AnomalyTargetTransform,
    DeltaTargetTransform,
    LevelTargetTransform,
    TargetTransform,
    TrendResidualTargetTransform,
    VolatilityNormalizedDeltaTargetTransform,
)

ALL_TRANSFORM_CLASSES = [
    LevelTargetTransform,
    DeltaTargetTransform,
    AnomalyTargetTransform,
    TrendResidualTargetTransform,
    VolatilityNormalizedDeltaTargetTransform,
]


def _mixed_observed_masked_frame(n_months: int = 30, start: str = "2003-01-01") -> pd.DataFrame:
    """A single-location fixture with a plausible trend + noise, some rows'
    own TWS_t masked (NaN) to mimic the real blackout structure, and a
    `target` column (the competition's TWS(t+1) label) present on every row
    regardless of whether TWS_t itself is masked."""
    rng = np.random.default_rng(7)
    times = [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(n_months)]
    trend = 1.0 + 0.05 * np.arange(n_months)
    noise = rng.normal(0.0, 0.3, size=n_months)
    tws = trend + noise

    # Mask a contiguous blackout block plus a couple of scattered months --
    # mirrors the real competition's masking structure closely enough for
    # this module's own logic (which never inspects TWS_t directly, only
    # build_state_snapshots/compute_location_signatures's derived fields).
    tws_masked = tws.copy()
    tws_masked[10:14] = np.nan
    tws_masked[20] = np.nan

    df = pd.DataFrame(
        {
            "time": times,
            "lat": 12.5,
            "lon": -3.5,
            "location_id": "12.5_-3.5",
            "TWS_t": tws_masked,
            # "target" = the following month's true TWS, the competition's
            # actual label -- defined for every row (observed or masked).
            "target": np.concatenate([trend[1:] + noise[1:], [trend[-1] + 0.05]]),
        }
    )
    return df


# --- protocol conformance ----------------------------------------------------


@pytest.mark.parametrize("transform_cls", ALL_TRANSFORM_CLASSES)
def test_satisfies_target_transform_protocol(transform_cls) -> None:
    assert isinstance(transform_cls(), TargetTransform)


def test_target_transforms_registry_has_exactly_five_expected_keys() -> None:
    assert set(TARGET_TRANSFORMS.keys()) == {
        "level",
        "delta",
        "anomaly",
        "trend_residual",
        "volatility_normalized_delta",
    }
    for transform in TARGET_TRANSFORMS.values():
        assert isinstance(transform, TargetTransform)


# --- round-trip invertibility: the property that must never silently break ---


@pytest.mark.parametrize("transform_cls", ALL_TRANSFORM_CLASSES)
def test_round_trip_invertibility_on_mixed_observed_and_masked_fixture(transform_cls) -> None:
    df = _mixed_observed_masked_frame()
    transform = transform_cls()

    forward = transform.forward(df)
    recovered = transform.inverse(forward, df)

    pd.testing.assert_series_equal(
        recovered.astype(float),
        df["target"].astype(float),
        check_names=False,
        atol=1e-8,
    )


@pytest.mark.parametrize("transform_cls", ALL_TRANSFORM_CLASSES)
def test_round_trip_invertibility_accepts_raw_ndarray_predictions(transform_cls) -> None:
    # inverse() must also accept a bare ndarray (not just a pd.Series),
    # per the TargetTransform protocol's own type union.
    df = _mixed_observed_masked_frame()
    transform = transform_cls()

    forward = transform.forward(df)
    recovered = transform.inverse(forward.to_numpy(), df)

    pd.testing.assert_series_equal(
        recovered.astype(float),
        df["target"].astype(float),
        check_names=False,
        atol=1e-8,
    )


@pytest.mark.parametrize("transform_cls", ALL_TRANSFORM_CLASSES)
def test_round_trip_invertibility_on_single_row_frame(transform_cls) -> None:
    # A degenerate single-row frame -- no history at all for
    # build_state_snapshots/compute_location_signatures to lean on -- must
    # still round-trip exactly (even if the intermediate baseline is a
    # fallback value like 0.0 or the row's own target).
    df = _mixed_observed_masked_frame().iloc[[0]].reset_index(drop=True)
    transform = transform_cls()

    forward = transform.forward(df)
    recovered = transform.inverse(forward, df)

    pd.testing.assert_series_equal(
        recovered.astype(float),
        df["target"].astype(float),
        check_names=False,
        atol=1e-8,
    )


# --- hand-computed arithmetic checks ------------------------------------------


def test_level_transform_is_identity() -> None:
    df = _mixed_observed_masked_frame()
    transform = LevelTargetTransform()
    forward = transform.forward(df)
    pd.testing.assert_series_equal(forward, df["target"].astype(float), check_names=False)


def test_delta_transform_matches_last_known_tws_by_hand() -> None:
    # A short, fully-observed fixture where last_known_tws for the final
    # row is trivially its own TWS_t (no blackout involved).
    times = pd.date_range("2003-01-01", periods=4, freq="MS")
    df = pd.DataFrame(
        {
            "time": times,
            "lat": 1.0,
            "lon": 1.0,
            "location_id": "1.0_1.0",
            "TWS_t": [10.0, 11.0, 12.0, 13.0],
            "target": [11.0, 12.0, 13.0, 14.0],
        }
    )
    transform = DeltaTargetTransform()
    forward = transform.forward(df)
    # last row: TWS_t=13.0 is observed, so effective_current == 13.0
    assert forward.iloc[-1] == pytest.approx(14.0 - 13.0)


def test_delta_transform_uses_last_known_when_current_row_masked() -> None:
    times = pd.date_range("2003-01-01", periods=5, freq="MS")
    df = pd.DataFrame(
        {
            "time": times,
            "lat": 1.0,
            "lon": 1.0,
            "location_id": "1.0_1.0",
            "TWS_t": [10.0, 11.0, 12.0, np.nan, np.nan],
            "target": [11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    transform = DeltaTargetTransform()
    forward = transform.forward(df)
    # Last row's own TWS_t is masked -> effective_current falls back to the
    # most recent observed value (12.0, from index 2).
    assert forward.iloc[-1] == pytest.approx(15.0 - 12.0)


def test_anomaly_transform_subtracts_shrinkage_regularized_mean() -> None:
    df = _mixed_observed_masked_frame()
    transform = AnomalyTargetTransform()
    forward = transform.forward(df)
    signature_mean = transform._signature_mean(df)
    pd.testing.assert_series_equal(
        forward, (df["target"].astype(float) - signature_mean), check_names=False
    )


# --- volatility-normalized delta: MIN_STD flooring ----------------------------


def test_volatility_normalized_delta_does_not_blow_up_when_std_near_zero() -> None:
    # A single-observation-per-period location has an ill-defined (or
    # exactly-zero, pre-shrinkage) raw std; MIN_STD must keep forward()
    # finite rather than dividing by (near) zero.
    times = pd.date_range("2003-01-01", periods=2, freq="MS")
    df = pd.DataFrame(
        {
            "time": times,
            "lat": 5.0,
            "lon": 5.0,
            "location_id": "5.0_5.0",
            "TWS_t": [7.0, 7.0],
            "target": [7.0, 7.0],
        }
    )
    transform = VolatilityNormalizedDeltaTargetTransform()
    forward = transform.forward(df)
    assert np.isfinite(forward).all()

    std = transform._std(df)
    assert (std >= MIN_STD - 1e-12).all()


def test_volatility_normalized_delta_round_trips_even_at_min_std_floor() -> None:
    times = pd.date_range("2003-01-01", periods=3, freq="MS")
    df = pd.DataFrame(
        {
            "time": times,
            "lat": 5.0,
            "lon": 5.0,
            "location_id": "5.0_5.0",
            "TWS_t": [7.0, 7.0, 7.0],
            "target": [7.0, 7.0, 7.0],
        }
    )
    transform = VolatilityNormalizedDeltaTargetTransform()
    forward = transform.forward(df)
    recovered = transform.inverse(forward, df)
    pd.testing.assert_series_equal(
        recovered.astype(float), df["target"].astype(float), check_names=False, atol=1e-8
    )
