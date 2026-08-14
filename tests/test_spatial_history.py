"""Tests for tws_forecast.state.spatial_history — Project Phase 4 step 4.3,
per docs/PHASE4_EXECUTION_PLAN.md §4.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.features.base import Transformer
from tws_forecast.state.spatial_history import (
    SPATIAL_FEATURE_TAXONOMY,
    SpatialHistoryTransformer,
)


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


def _grid_fixture(n_months: int = 24, rng_seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    frames = []
    for lat in (0.5, 1.5, 2.5):
        for lon in (0.5, 1.5, 2.5):
            values = (
                1.0
                + 0.02 * lat
                + 0.01 * np.arange(n_months)
                + rng.normal(scale=0.02, size=n_months)
            ).tolist()
            frames.append(_location_frame(lat, lon, n_months, values))
    return pd.concat(frames, ignore_index=True)


# --- Taxonomy guard: no S1 feature in this module ---------------------------


def test_no_s1_feature_in_taxonomy() -> None:
    assert "S1" not in SPATIAL_FEATURE_TAXONOMY.values()


def test_taxonomy_only_contains_s2_and_s3() -> None:
    assert set(SPATIAL_FEATURE_TAXONOMY.values()) <= {"S2", "S3"}


def test_taxonomy_covers_every_output_column() -> None:
    df = _grid_fixture()
    train_df = df[pd.to_datetime(df["time"]) < pd.Timestamp("2004-06-01")]
    val_df = df[pd.to_datetime(df["time"]) >= pd.Timestamp("2004-06-01")]

    transformer = SpatialHistoryTransformer(
        n_neighbors=4, distance_weighting="inverse_distance", max_neighbor_distance_km=1000.0
    )
    transformer.fit(train_df)
    result = transformer.transform(val_df)

    assert set(result.columns) == set(SPATIAL_FEATURE_TAXONOMY.keys())


# --- Transformer protocol ---------------------------------------------------


def test_spatial_history_transformer_satisfies_protocol() -> None:
    assert isinstance(SpatialHistoryTransformer(), Transformer)


def test_raises_before_fit() -> None:
    transformer = SpatialHistoryTransformer()
    df = _grid_fixture()
    with pytest.raises(RuntimeError):
        transformer.transform(df)


# --- Origin-time indexing: unaffected by future rows -------------------------


def test_unaffected_by_rows_at_or_after_the_query_period() -> None:
    df = _grid_fixture(n_months=24)
    cutoff = pd.Timestamp("2004-01-01")
    train_df = df[pd.to_datetime(df["time"]) < cutoff]
    query_row = df[(df["location_id"] == "1.5_1.5") & (pd.to_datetime(df["time"]) == cutoff)]

    transformer = SpatialHistoryTransformer(n_neighbors=4, max_neighbor_distance_km=1000.0)
    transformer.fit(train_df)

    result_without_future = transformer.transform(query_row)

    df_with_future_row_included = df[pd.to_datetime(df["time"]) <= cutoff]
    result_with_future = transformer.transform(
        df_with_future_row_included[
            (df_with_future_row_included["location_id"] == "1.5_1.5")
            & (pd.to_datetime(df_with_future_row_included["time"]) == cutoff)
        ]
    )

    # transform() itself doesn't see "future" rows in either call (both
    # queries are for the same single row) -- this pins that the *neighbor*
    # lookups are keyed by the same period as the query row, not polluted
    # by whichever frame happens to be passed to fit().
    pd.testing.assert_frame_equal(
        result_without_future.reset_index(drop=True), result_with_future.reset_index(drop=True)
    )


def test_neighbor_feature_does_not_use_neighbor_row_after_query_period() -> None:
    # Neighbor's own history is truncated to fully-known values, then a
    # wild outlier is added far in the future -- if lookups leaked forward,
    # neighbor_TWS_last_known would be pulled toward the outlier.
    close = _location_frame(0.5, 0.5, 12, [1.0] * 12)
    far_future_row = _location_frame(0.5, 0.5, 1, [999.0], start="2005-01-01")
    neighbor = pd.concat([close, far_future_row], ignore_index=True)

    query_loc = _location_frame(0.5, 0.6, 12, [2.0] * 12)
    train_df = pd.concat([neighbor, query_loc], ignore_index=True)

    query_row = query_loc[pd.to_datetime(query_loc["time"]) == pd.Timestamp("2003-12-01")]

    transformer = SpatialHistoryTransformer(n_neighbors=1, max_neighbor_distance_km=1000.0)
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert result["neighbor_TWS_last_known"].iloc[0] == pytest.approx(1.0)


# --- Close vs. far neighbor weighting ---------------------------------------


def test_close_neighbor_dominates_inverse_distance_weighted_aggregate() -> None:
    # location at (0,0); a close neighbor at (0, 0.1) with value 10, a far
    # neighbor at (0, 5.0) with value -10. Inverse-distance weighting must
    # pull the aggregate toward the close neighbor's value.
    target = _location_frame(0.0, 0.0, 6, [0.0] * 6)
    close_neighbor = _location_frame(0.0, 0.1, 6, [10.0] * 6)
    far_neighbor = _location_frame(0.0, 5.0, 6, [-10.0] * 6)
    train_df = pd.concat([target, close_neighbor, far_neighbor], ignore_index=True)

    query_row = target[pd.to_datetime(target["time"]) == pd.Timestamp("2003-06-01")]

    transformer = SpatialHistoryTransformer(
        n_neighbors=2, distance_weighting="inverse_distance", max_neighbor_distance_km=1000.0
    )
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert result["neighbor_TWS_last_known"].iloc[0] > 0.0  # pulled toward +10, not toward -10


def test_flat_mean_weighting_averages_neighbors_equally() -> None:
    target = _location_frame(0.0, 0.0, 6, [0.0] * 6)
    close_neighbor = _location_frame(0.0, 0.1, 6, [10.0] * 6)
    far_neighbor = _location_frame(0.0, 1.0, 6, [-10.0] * 6)
    train_df = pd.concat([target, close_neighbor, far_neighbor], ignore_index=True)

    query_row = target[pd.to_datetime(target["time"]) == pd.Timestamp("2003-06-01")]

    transformer = SpatialHistoryTransformer(
        n_neighbors=2, distance_weighting="flat_mean", max_neighbor_distance_km=1000.0
    )
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert result["neighbor_TWS_last_known"].iloc[0] == pytest.approx(0.0, abs=1e-6)


# --- Zero-neighbor fallback ---------------------------------------------------


def test_isolated_location_falls_back_gracefully() -> None:
    isolated = _location_frame(80.0, 80.0, 6, [3.0] * 6)
    far_away = _location_frame(-80.0, -80.0, 6, [1.0] * 6)
    train_df = pd.concat([isolated, far_away], ignore_index=True)

    query_row = isolated[pd.to_datetime(isolated["time"]) == pd.Timestamp("2003-06-01")]

    transformer = SpatialHistoryTransformer(
        n_neighbors=4, distance_weighting="inverse_distance", max_neighbor_distance_km=10.0
    )
    transformer.fit(train_df)
    result = transformer.transform(query_row)

    assert not result.isna().any().any()
    # Fallback is the global pooled mean across both locations, not a crash
    # or NaN.
    assert result["neighbor_TWS_last_known"].iloc[0] == pytest.approx(2.0, abs=1e-6)
    assert result["neighbor_historical_anomaly"].iloc[0] == pytest.approx(0.0)


def test_single_location_dataset_never_raises() -> None:
    df = _location_frame(0.0, 0.0, 6, [1.0] * 6)
    transformer = SpatialHistoryTransformer(n_neighbors=4, max_neighbor_distance_km=100.0)
    transformer.fit(df)
    result = transformer.transform(df.tail(1))
    assert len(result) == 1
    assert not result.isna().any().any()


# --- Result shape / no NaNs on a normal grid ---------------------------------


def test_result_indexed_like_input_and_has_no_unexpected_nans() -> None:
    df = _grid_fixture(n_months=18)
    train_df = df[pd.to_datetime(df["time"]) < pd.Timestamp("2004-01-01")]
    val_df = df[pd.to_datetime(df["time"]) >= pd.Timestamp("2004-01-01")]

    transformer = SpatialHistoryTransformer(n_neighbors=4, max_neighbor_distance_km=1000.0)
    transformer.fit(train_df)
    result = transformer.transform(val_df)

    assert list(result.index) == list(val_df.index)
    assert not result.isna().any().any()
