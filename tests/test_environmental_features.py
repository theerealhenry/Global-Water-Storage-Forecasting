"""Tests for tws_forecast.features.environmental — Project Phase 4 step 4.6,
per docs/PHASE4_EXECUTION_PLAN.md §4.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.features.base import Transformer
from tws_forecast.features.environmental import (
    DroughtPersistenceTransformer,
    SoilMoistureTrajectoryTransformer,
    SpeiDifferencingTransformer,
)


def _monthly_frame(n_months: int, start: str = "2003-01-01", **columns) -> pd.DataFrame:
    times = [(pd.Timestamp(start) + pd.DateOffset(months=i)) for i in range(n_months)]
    base = {"time": times, "lat": 0.5, "lon": 0.5, "location_id": "0.5_0.5"}
    base.update(columns)
    return pd.DataFrame(base)


# --- Transformer protocol ---------------------------------------------------


def test_spei_differencing_satisfies_protocol() -> None:
    assert isinstance(SpeiDifferencingTransformer(), Transformer)


def test_drought_persistence_satisfies_protocol() -> None:
    assert isinstance(DroughtPersistenceTransformer(), Transformer)


def test_soil_moisture_trajectory_satisfies_protocol() -> None:
    assert isinstance(SoilMoistureTrajectoryTransformer(), Transformer)


@pytest.mark.parametrize(
    "transformer_cls",
    [SpeiDifferencingTransformer, DroughtPersistenceTransformer, SoilMoistureTrajectoryTransformer],
)
def test_raises_before_fit(transformer_cls) -> None:
    transformer = transformer_cls()
    df = _monthly_frame(3, SPEI_12_t=[0.0, 0.0, 0.0], SOIL_MOISTURE_t=[0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError):
        transformer.transform(df)


# --- SpeiDifferencingTransformer: hand-computed arithmetic -------------------


def test_spei_diff_arithmetic_on_hand_built_fixture() -> None:
    n_months = 15
    spei03 = [float(i) for i in range(n_months)]  # 0, 1, 2, ..., 14
    df = _monthly_frame(n_months, SPEI_03_t=spei03)

    train_df = df.iloc[:12]
    query_row = df.iloc[[12]]  # SPEI_03_t = 12.0

    transformer = SpeiDifferencingTransformer(spei_diff_lags=(3,))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    # value at t=12 is 12.0; value at t-3=9 is 9.0 -> diff = 3.0
    assert result["spei_03_diff_3"].iloc[0] == pytest.approx(3.0)


def test_spei_diff_nan_when_lagged_month_has_no_history() -> None:
    n_months = 4
    spei03 = [1.0, 2.0, 3.0, 4.0]
    df = _monthly_frame(n_months, SPEI_03_t=spei03)

    train_df = df.iloc[:2]
    query_row = df.iloc[[2]]  # only 2 months of prior history; lag-6 unavailable

    transformer = SpeiDifferencingTransformer(spei_diff_lags=(6,))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert np.isnan(result["spei_03_diff_6"].iloc[0])


def test_spei_diff_covers_every_present_timescale_and_lag() -> None:
    n_months = 20
    df = _monthly_frame(
        n_months,
        SPEI_01_t=list(range(n_months)),
        SPEI_03_t=list(range(n_months)),
        SPEI_06_t=list(range(n_months)),
        SPEI_12_t=list(range(n_months)),
    )
    train_df = df.iloc[:15]
    query_row = df.iloc[[15]]

    transformer = SpeiDifferencingTransformer(spei_diff_lags=(3, 6))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    expected_cols = {f"spei_{ts}_diff_{lag}" for ts in ("01", "03", "06", "12") for lag in (3, 6)}
    assert set(result.columns) == expected_cols


def test_spei_diff_no_leakage_from_future_rows() -> None:
    # A future outlier appended after the query row must not change the
    # diff computed for an earlier row.
    n_months = 10
    spei03 = [float(i) for i in range(n_months)]
    df = _monthly_frame(n_months, SPEI_03_t=spei03)
    query_row = df.iloc[[7]]
    train_without_future = df.iloc[:7]

    outlier = _monthly_frame(1, start="2005-01-01", SPEI_03_t=[9999.0])
    train_with_future = pd.concat([df.iloc[:8], outlier], ignore_index=True)

    t1 = SpeiDifferencingTransformer(spei_diff_lags=(3,))
    t1.fit(train_without_future)
    r1 = t1.transform(query_row)

    t2 = SpeiDifferencingTransformer(spei_diff_lags=(3,))
    t2.fit(train_with_future)
    r2 = t2.transform(query_row)

    assert r1["spei_03_diff_3"].iloc[0] == pytest.approx(r2["spei_03_diff_3"].iloc[0])


# --- DroughtPersistenceTransformer: hand-computed run-length -----------------


def test_drought_run_length_hand_computed() -> None:
    # threshold -1.5: months 3,4,5 (0-indexed) are in drought, then a break.
    spei12 = [0.0, 0.0, 0.0, -2.0, -1.8, -1.6, 0.5, -2.0]
    df = _monthly_frame(len(spei12), SPEI_12_t=spei12)

    transformer = DroughtPersistenceTransformer(drought_threshold=-1.5)
    transformer.fit(df.iloc[:6])
    result_at_index_5 = transformer.transform(df.iloc[[5]])
    assert result_at_index_5["drought_persistence_run_length"].iloc[0] == pytest.approx(3.0)

    transformer2 = DroughtPersistenceTransformer(drought_threshold=-1.5)
    transformer2.fit(df.iloc[:6])
    result_at_index_2 = transformer2.transform(df.iloc[[2]])
    assert result_at_index_2["drought_persistence_run_length"].iloc[0] == pytest.approx(0.0)

    transformer3 = DroughtPersistenceTransformer(drought_threshold=-1.5)
    transformer3.fit(df.iloc[:7])
    result_at_index_6 = transformer3.transform(df.iloc[[6]])
    assert result_at_index_6["drought_persistence_run_length"].iloc[0] == pytest.approx(0.0)

    transformer4 = DroughtPersistenceTransformer(drought_threshold=-1.5)
    transformer4.fit(df.iloc[:8])
    result_at_index_7 = transformer4.transform(df.iloc[[7]])
    assert result_at_index_7["drought_persistence_run_length"].iloc[0] == pytest.approx(1.0)


def test_drought_run_length_missing_column_falls_back_to_zero() -> None:
    df = _monthly_frame(3, lat=0.5)  # no SPEI_12_t column at all
    transformer = DroughtPersistenceTransformer()
    transformer.fit(df.iloc[:2])
    result = transformer.transform(df.iloc[[2]])
    assert result["drought_persistence_run_length"].iloc[0] == pytest.approx(0.0)


def test_drought_run_length_no_leakage_from_future_rows() -> None:
    spei12 = [0.0, -2.0, -2.0, -2.0, 0.0]
    df = _monthly_frame(len(spei12), SPEI_12_t=spei12)
    query_row = df.iloc[[3]]

    t1 = DroughtPersistenceTransformer(drought_threshold=-1.5)
    t1.fit(df.iloc[:3])
    r1 = t1.transform(query_row)

    # Now change the future (index 4) value -- must not affect the streak
    # computed for index 3.
    df_altered = df.copy()
    df_altered.loc[4, "SPEI_12_t"] = -99.0
    t2 = DroughtPersistenceTransformer(drought_threshold=-1.5)
    t2.fit(df_altered.iloc[:3])
    r2 = t2.transform(df_altered.iloc[[3]])

    assert r1["drought_persistence_run_length"].iloc[0] == pytest.approx(
        r2["drought_persistence_run_length"].iloc[0]
    )


# --- SoilMoistureTrajectoryTransformer ---------------------------------------


def test_soil_moisture_last_known_and_velocity() -> None:
    values = [1.0, 1.5, 2.7]
    df = _monthly_frame(3, SOIL_MOISTURE_t=values)

    transformer = SoilMoistureTrajectoryTransformer(trend_window_months=(3,))
    transformer.fit(df.iloc[:2])
    result = transformer.transform(df.iloc[[2]])

    assert result["soil_moisture_last_known"].iloc[0] == pytest.approx(2.7)
    assert result["soil_moisture_velocity"].iloc[0] == pytest.approx(1.2)  # 2.7 - 1.5


def test_soil_moisture_trend_recovers_known_slope() -> None:
    n_months = 12
    known_slope = 0.1
    values = [1.0 + known_slope * i for i in range(n_months)]
    df = _monthly_frame(n_months, SOIL_MOISTURE_t=values)

    transformer = SoilMoistureTrajectoryTransformer(trend_window_months=(12,))
    transformer.fit(df.iloc[:11])
    result = transformer.transform(df.iloc[[11]])

    assert result["soil_moisture_trend_12"].iloc[0] == pytest.approx(known_slope, abs=1e-6)


def test_soil_moisture_missing_column_falls_back_to_nan() -> None:
    df = _monthly_frame(3, lat=0.5)  # no SOIL_MOISTURE_t column
    transformer = SoilMoistureTrajectoryTransformer(trend_window_months=(3,))
    transformer.fit(df.iloc[:2])
    result = transformer.transform(df.iloc[[2]])
    assert np.isnan(result["soil_moisture_last_known"].iloc[0])


def test_soil_moisture_no_leakage_from_future_rows() -> None:
    values = [1.0, 1.0, 1.0, 1.0]
    df = _monthly_frame(4, SOIL_MOISTURE_t=values)
    query_row = df.iloc[[2]]

    t1 = SoilMoistureTrajectoryTransformer(trend_window_months=(3,))
    t1.fit(df.iloc[:2])
    r1 = t1.transform(query_row)

    df_altered = df.copy()
    df_altered.loc[3, "SOIL_MOISTURE_t"] = 999.0
    t2 = SoilMoistureTrajectoryTransformer(trend_window_months=(3,))
    t2.fit(df_altered.iloc[:2])
    r2 = t2.transform(df_altered.iloc[[2]])

    assert r1["soil_moisture_last_known"].iloc[0] == pytest.approx(
        r2["soil_moisture_last_known"].iloc[0]
    )
