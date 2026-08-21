"""Tests for tws_forecast.features.assemble — Project Phase 4 step 4.9's
feature-assembly pipeline, per docs/PHASE4_EXECUTION_PLAN.md §4.9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.features import assemble, temporal
from tws_forecast.features.assemble import RAW_PASSTHROUGH_COLUMNS, build_feature_matrix
from tws_forecast.validation.leakage_tests import future_row_shuffle_test


def _multi_location_frame(
    n_locations: int = 5, n_months: int = 20, start: str = "2003-01-01"
) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    times = [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(n_months)]
    frames = []
    for loc_idx in range(n_locations):
        lat, lon = 5.0 + loc_idx * 0.5, 15.0 + loc_idx * 0.5
        trend = 1.0 + 0.02 * np.arange(n_months) + loc_idx
        month = np.array([t.month for t in times])
        angle = 2 * np.pi * (month - 1) / 12.0
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "lat": lat,
                    "lon": lon,
                    "location_id": f"{lat}_{lon}",
                    "TWS_t": trend + rng.normal(0.0, 0.2, size=n_months),
                    "SPEI_01_t": rng.normal(0.0, 1.0, size=n_months),
                    "SPEI_03_t": rng.normal(0.0, 1.0, size=n_months),
                    "SPEI_06_t": rng.normal(0.0, 1.0, size=n_months),
                    "SPEI_12_t": rng.normal(0.0, 1.0, size=n_months),
                    "SOIL_MOISTURE_t": 5.0 + rng.normal(0.0, 0.1, size=n_months),
                    "month_sin": np.sin(angle),
                    "month_cos": np.cos(angle),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def real_train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


# --- basic shape/indexing -----------------------------------------------------


def test_output_indexed_identically_to_df() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df)
    pd.testing.assert_index_equal(matrix.index, df.index)


def test_output_has_same_row_count_for_a_query_subset() -> None:
    df = _multi_location_frame()
    train = df.iloc[:80]
    query = df.iloc[80:90]
    matrix = build_feature_matrix(query, train_df=train)
    assert len(matrix) == len(query)
    pd.testing.assert_index_equal(matrix.index, query.index)


# --- column composition: no collisions, every step's prefix present ----------


def test_every_step_prefix_present_with_all_steps_enabled() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df)
    columns = set(matrix.columns)

    assert set(RAW_PASSTHROUGH_COLUMNS) <= columns
    assert any(c.startswith("state_") for c in columns)
    assert any(c.startswith("signature_") for c in columns)
    assert any(c.startswith("neighbor_") for c in columns)
    assert "trend_slope_12" in columns
    assert {"month_hemisphere_sin", "month_hemisphere_cos"} <= columns
    assert any(c.startswith("spei_") and "diff" in c for c in columns)
    assert "drought_persistence_run_length" in columns
    assert any(c.startswith("soil_moisture_") for c in columns)


def test_no_column_name_collisions_across_steps() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df)
    assert (
        not matrix.columns.duplicated().any()
    ), f"duplicate columns across composed steps: {matrix.columns[matrix.columns.duplicated()].tolist()}"


def test_state_snapshot_join_keys_and_null_field_are_dropped() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df)
    for dropped in (
        "state_location_id",
        "state_as_of",
        "state_last_known_time",
        "state_location_signature",
    ):
        assert dropped not in matrix.columns, dropped
    for dropped in ("signature_location_id", "signature_as_of"):
        assert dropped not in matrix.columns, dropped


def test_raw_tws_t_excluded_but_state_last_known_tws_present() -> None:
    # TWS_t is masked on 2/3 of real test rows -- never a uniform raw
    # feature (see the module docstring / RidgeBaselinePredictor's
    # identical reasoning). state_last_known_tws is the leakage-safe,
    # observed/masked-resolved substitute.
    df = _multi_location_frame()
    matrix = build_feature_matrix(df)
    assert "TWS_t" not in matrix.columns
    assert "state_last_known_tws" in matrix.columns


# --- toggles -------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,expected_absent_prefix",
    [
        ("include_state_snapshot", "state_"),
        ("include_signatures", "signature_"),
        ("include_spatial_history", "neighbor_"),
    ],
)
def test_toggling_a_step_off_removes_its_columns(flag: str, expected_absent_prefix: str) -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df, **{flag: False})
    assert not any(c.startswith(expected_absent_prefix) for c in matrix.columns)


def test_toggling_temporal_off_removes_its_columns() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df, include_temporal=False)
    assert "trend_slope_12" not in matrix.columns
    assert "month_hemisphere_sin" not in matrix.columns


def test_toggling_environmental_off_removes_its_columns() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df, include_environmental=False)
    assert not any(c.startswith("spei_") and "diff" in c for c in matrix.columns)
    assert "drought_persistence_run_length" not in matrix.columns
    assert not any(c.startswith("soil_moisture_") for c in matrix.columns)


def test_all_steps_disabled_leaves_only_raw_passthrough() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(
        df,
        include_state_snapshot=False,
        include_signatures=False,
        include_spatial_history=False,
        include_temporal=False,
        include_environmental=False,
    )
    assert set(matrix.columns) == set(RAW_PASSTHROUGH_COLUMNS)


# --- train_df defaulting --------------------------------------------------------


def test_train_df_defaults_to_df_itself() -> None:
    df = _multi_location_frame()
    explicit = build_feature_matrix(df, train_df=df)
    defaulted = build_feature_matrix(df)
    pd.testing.assert_frame_equal(explicit, defaulted)


# --- forwarded kwargs ------------------------------------------------------------


def test_trend_window_months_kwarg_forwarded() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df, trend_window_months=(6,))
    assert "trend_slope_6" in matrix.columns
    assert "trend_slope_12" not in matrix.columns


def test_trailing_windows_kwarg_forwarded_to_state_snapshot_density() -> None:
    df = _multi_location_frame()
    matrix = build_feature_matrix(df, trailing_windows=(6,))
    assert "state_observation_density_6" in matrix.columns
    assert "state_observation_density_12" not in matrix.columns


# --- end-to-end leakage proof, at the composed level ----------------------------


class _FeatureMatrixSumPredictor:
    """A minimal Predictor whose 'prediction' is the row-sum of the fully
    assembled feature matrix -- future_row_shuffle_test only needs a
    Predictor-shaped stand-in to confirm the assembled pipeline's own row
    selection is time-based, not positional, end-to-end across every
    composed step at once (docs/PHASE4_EXECUTION_PLAN.md step 4.9 item 2 --
    step 4.8 already covers each step in isolation; this covers the
    composed whole, matching notebooks/05_state_features.ipynb section 2's
    real-data run)."""

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        matrix = build_feature_matrix(df, train_df=self._train_df)
        numeric = matrix.select_dtypes(include=[np.number]).fillna(0.0)
        return numeric.to_numpy(dtype=float).sum(axis=1)


def test_future_row_shuffle_no_leak_on_assembled_matrix() -> None:
    df = _multi_location_frame(n_locations=6, n_months=24)
    cutoff = df["time"].sort_values().unique()[len(df["time"].unique()) // 2]
    predictor = _FeatureMatrixSumPredictor()
    assert future_row_shuffle_test(predictor, df, cutoff_time=cutoff) is True


def test_future_row_shuffle_no_leak_on_real_golden_data(real_train_df: pd.DataFrame) -> None:
    predictor = _FeatureMatrixSumPredictor()
    assert future_row_shuffle_test(predictor, real_train_df, cutoff_time="2005-06-01") is True


# --- full composition against real data -----------------------------------------


# --- redundant-computation elimination (Project Phase 4 step 4.9's first ------
# --- proof run found build_state_snapshots/compute_location_signatures ---------
# --- each called far more times than necessary per build_feature_matrix() -----
# --- call -- see the module's own "Performance note" docstring) ----------------


def test_build_state_snapshots_and_signatures_are_not_recomputed_redundantly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _multi_location_frame(n_locations=6, n_months=20)

    state_calls = {"count": 0}
    signature_calls = {"count": 0}

    # build_state_snapshots is called from two module namespaces that each
    # bound their own reference via `from ... import build_state_snapshots`
    # (assemble.py directly, and temporal.py for whichever trend window
    # isn't already covered by assemble.py's shared panel) -- both must be
    # patched to observe the true total. compute_location_signatures is
    # only ever called directly from assemble.py: SpatialHistoryTransformer
    # never falls back to computing it internally here, since assemble.py
    # always supplies a signature_panel.
    real_build_state_snapshots_assemble = assemble.build_state_snapshots
    real_build_state_snapshots_temporal = temporal.build_state_snapshots
    real_compute_location_signatures = assemble.compute_location_signatures

    def counting_build_state_snapshots_assemble(*args, **kwargs):
        state_calls["count"] += 1
        return real_build_state_snapshots_assemble(*args, **kwargs)

    def counting_build_state_snapshots_temporal(*args, **kwargs):
        state_calls["count"] += 1
        return real_build_state_snapshots_temporal(*args, **kwargs)

    def counting_compute_location_signatures(*args, **kwargs):
        signature_calls["count"] += 1
        return real_compute_location_signatures(*args, **kwargs)

    monkeypatch.setattr(assemble, "build_state_snapshots", counting_build_state_snapshots_assemble)
    monkeypatch.setattr(temporal, "build_state_snapshots", counting_build_state_snapshots_temporal)
    monkeypatch.setattr(
        assemble, "compute_location_signatures", counting_compute_location_signatures
    )

    build_feature_matrix(df)

    # Before the panel-injection fix: build_state_snapshots was called 4x
    # (once for state_* columns, once per trend window inside
    # TrailingTrendTransformer, once more inside SpatialHistoryTransformer)
    # and compute_location_signatures 2x (once for signature_* columns,
    # once more inside SpatialHistoryTransformer). With the default
    # trailing_windows=(12, 24)/trend_window_months=(12, 24), window 24's
    # panel is now shared (assemble.py's own single call), leaving only
    # window 12's dedicated call (inside TrailingTrendTransformer) --
    # 2 build_state_snapshots calls total, down from 4.
    # SpatialHistoryTransformer receives both panels pre-built and never
    # calls either function itself -- 1 compute_location_signatures call
    # total, down from 2.
    assert state_calls["count"] == 2
    assert signature_calls["count"] == 1


def test_full_composition_against_real_golden_data(real_train_df: pd.DataFrame) -> None:
    train = real_train_df[real_train_df["time"] < "2005-01-01"]
    query = real_train_df[
        (real_train_df["time"] >= "2005-01-01") & (real_train_df["time"] < "2005-04-01")
    ]
    assert len(query) > 0
    matrix = build_feature_matrix(query, train_df=train)
    assert len(matrix) == len(query)
    assert not matrix.columns.duplicated().any()
    # Every column should be numeric except state_status, which carries
    # the observed/reconstructed/never-observed regime needed for step
    # 4.9's feature-importance-by-regime decomposition.
    non_numeric = [c for c in matrix.columns if not pd.api.types.is_numeric_dtype(matrix[c])]
    assert non_numeric == ["state_state_status"]
