"""Tests for tws_forecast.features.temporal — Project Phase 4 step 4.5, per
docs/PHASE4_EXECUTION_PLAN.md §4.5 and docs/ASSUMPTIONS.md A-011.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.features.base import Transformer
from tws_forecast.features.temporal import MonthHemisphereTransformer, TrailingTrendTransformer


def _location_frame(
    lat: float, lon: float, n_months: int, values, start: str = "2003-01-01"
) -> pd.DataFrame:
    times = [(pd.Timestamp(start) + pd.DateOffset(months=i)) for i in range(n_months)]
    return pd.DataFrame(
        {
            "time": times,
            "lat": lat,
            "lon": lon,
            "location_id": f"{lat}_{lon}",
            "TWS_t": values,
        }
    )


# --- Transformer protocol ---------------------------------------------------


def test_trailing_trend_transformer_satisfies_protocol() -> None:
    assert isinstance(TrailingTrendTransformer(), Transformer)


def test_month_hemisphere_transformer_satisfies_protocol() -> None:
    assert isinstance(MonthHemisphereTransformer(), Transformer)


def test_trailing_trend_raises_before_fit() -> None:
    transformer = TrailingTrendTransformer(trend_window_months=(12,))
    df = _location_frame(0.5, 0.5, 6, [1.0] * 6)
    with pytest.raises(RuntimeError):
        transformer.transform(df)


# --- TrailingTrendTransformer: recovers known slope --------------------------


def test_trend_slope_recovers_known_linear_slope() -> None:
    n_months = 30
    known_slope = 0.05
    values = [1.0 + known_slope * i for i in range(n_months)]
    df = _location_frame(0.5, 0.5, n_months, values)

    train_df = df.iloc[:24]
    query_row = df.iloc[[23]]

    transformer = TrailingTrendTransformer(trend_window_months=(12,))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert result["trend_slope_12"].iloc[0] == pytest.approx(known_slope, abs=1e-6)


def test_multiple_windows_produce_distinct_columns() -> None:
    n_months = 30
    values = [1.0 + 0.03 * i for i in range(n_months)]
    df = _location_frame(0.5, 0.5, n_months, values)
    train_df = df.iloc[:24]
    query_row = df.iloc[[23]]

    transformer = TrailingTrendTransformer(trend_window_months=(12, 24))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert set(result.columns) == {"trend_slope_12", "trend_slope_24"}
    assert result["trend_slope_12"].iloc[0] == pytest.approx(0.03, abs=1e-6)
    assert result["trend_slope_24"].iloc[0] == pytest.approx(0.03, abs=1e-6)


def test_default_windows_loaded_from_config() -> None:
    transformer = TrailingTrendTransformer()
    assert transformer._trend_window_months == (12, 24)


# --- A-011: behavior explicitly checked on October and under-represented ----
# --- months, not just an aggregate pass. ------------------------------------


def test_trend_slope_computes_correctly_for_october_row() -> None:
    # October is the month entirely absent from the real test set
    # (A-011) -- this pins that TrailingTrendTransformer still produces a
    # sensible, non-NaN slope when the *query* row itself happens to be an
    # October row (a plausible training/validation-fold row even though
    # the real test set never asks for October specifically).
    n_months = 24
    values = [1.0 + 0.04 * i for i in range(n_months)]
    df = _location_frame(0.5, 0.5, n_months, values, start="2003-03-01")
    october_rows = df[pd.to_datetime(df["time"]).dt.month == 10]
    assert not october_rows.empty

    query_row = october_rows.iloc[[-1]]
    train_df = df[pd.to_datetime(df["time"]) < pd.to_datetime(query_row["time"]).iloc[0]]

    transformer = TrailingTrendTransformer(trend_window_months=(12,))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert np.isfinite(result["trend_slope_12"].iloc[0])
    assert result["trend_slope_12"].iloc[0] == pytest.approx(0.04, abs=1e-6)


@pytest.mark.parametrize("month", [4, 5, 8, 11])
def test_trend_slope_computes_correctly_for_under_represented_months(month: int) -> None:
    # A-011: April/May/Aug/Nov are under-represented (roughly half the
    # row-share of the recurring months) in the real test calendar.
    n_months = 30
    values = [1.0 + 0.02 * i for i in range(n_months)]
    df = _location_frame(0.5, 0.5, n_months, values, start="2003-01-01")
    target_rows = df[pd.to_datetime(df["time"]).dt.month == month]
    assert not target_rows.empty

    query_row = target_rows.iloc[[-1]]
    train_df = df[pd.to_datetime(df["time"]) < pd.to_datetime(query_row["time"]).iloc[0]]

    transformer = TrailingTrendTransformer(trend_window_months=(12,))
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert np.isfinite(result["trend_slope_12"].iloc[0])


# --- MonthHemisphereTransformer ----------------------------------------------


def test_month_hemisphere_fit_is_a_no_op() -> None:
    transformer = MonthHemisphereTransformer()
    transformer.fit(pd.DataFrame())  # must not raise
    df = _location_frame(0.5, 0.5, 1, [1.0])
    result = transformer.transform(df)
    assert len(result) == 1


def test_southern_hemisphere_january_matches_northern_hemisphere_july() -> None:
    # Both are each hemisphere's own deep-summer month -- phase-shifting
    # should make their cyclical encodings coincide.
    north_july = pd.DataFrame({"time": [pd.Timestamp("2004-07-01")], "lat": [10.0], "lon": [0.0]})
    south_january = pd.DataFrame(
        {"time": [pd.Timestamp("2004-01-01")], "lat": [-10.0], "lon": [0.0]}
    )

    transformer = MonthHemisphereTransformer()
    north_result = transformer.transform(north_july)
    south_result = transformer.transform(south_january)

    assert north_result["month_hemisphere_sin"].iloc[0] == pytest.approx(
        south_result["month_hemisphere_sin"].iloc[0], abs=1e-9
    )
    assert north_result["month_hemisphere_cos"].iloc[0] == pytest.approx(
        south_result["month_hemisphere_cos"].iloc[0], abs=1e-9
    )


def test_northern_hemisphere_unshifted() -> None:
    df = pd.DataFrame({"time": [pd.Timestamp("2004-01-01")], "lat": [10.0], "lon": [0.0]})
    transformer = MonthHemisphereTransformer()
    result = transformer.transform(df)
    expected_angle = 2 * np.pi * 0 / 12.0
    assert result["month_hemisphere_sin"].iloc[0] == pytest.approx(np.sin(expected_angle), abs=1e-9)
    assert result["month_hemisphere_cos"].iloc[0] == pytest.approx(np.cos(expected_angle), abs=1e-9)


def test_month_hemisphere_output_bounded() -> None:
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "time": pd.date_range("2003-01-01", periods=n, freq="MS"),
            "lat": rng.uniform(-60, 60, size=n),
            "lon": rng.uniform(-180, 180, size=n),
        }
    )
    transformer = MonthHemisphereTransformer()
    result = transformer.transform(df)
    assert (result["month_hemisphere_sin"].abs() <= 1.0 + 1e-9).all()
    assert (result["month_hemisphere_cos"].abs() <= 1.0 + 1e-9).all()
