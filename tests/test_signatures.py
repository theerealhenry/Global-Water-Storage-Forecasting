"""Tests for tws_forecast.state.signatures — Project Phase 4 step 4.2, per
docs/PHASE4_EXECUTION_PLAN.md §4.2 and docs/ASSUMPTIONS.md A-014.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.features.base import Transformer
from tws_forecast.state.signatures import (
    LocationSignature,
    LocationSignatureTransformer,
    compute_location_signature,
    compute_location_signatures,
)


def _rows(location_id: str, entries: list[tuple[str, float | None]]) -> pd.DataFrame:
    lat_str, lon_str = location_id.split("_")
    return pd.DataFrame(
        {
            "time": [pd.Timestamp(t) for t, _ in entries],
            "lat": [float(lat_str)] * len(entries),
            "lon": [float(lon_str)] * len(entries),
            "location_id": [location_id] * len(entries),
            "TWS_t": [v for _, v in entries],
        }
    )


def _monthly_entries(start: str, n_months: int, values) -> list[tuple[str, float | None]]:
    times = [
        (pd.Timestamp(start) + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
        for i in range(n_months)
    ]
    return list(zip(times, values, strict=True))


# --- LocationSignature basic shape / origin-time indexing -----------------


def test_location_signature_uses_strictly_before_as_of() -> None:
    # A row exactly at as_of must NOT be used -- opposite of StateSnapshot's
    # inclusive semantics, per the module docstring.
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 999.0)])
    sig = compute_location_signature(df, location_id="0.5_0.5", as_of="2004-02-01", shrinkage_k=5)
    # If Feb's 999.0 leaked in, mean would be pulled drastically upward.
    assert sig.n_observations == 1
    assert sig.mean < 10.0


def test_unchanged_when_rows_at_or_after_as_of_are_removed() -> None:
    df_with_future = _rows(
        "0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 1.5), ("2004-03-01", 500.0)]
    )
    df_without_future = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 1.5)])

    sig_with = compute_location_signature(
        df_with_future, location_id="0.5_0.5", as_of="2004-02-01", shrinkage_k=5
    )
    sig_without = compute_location_signature(
        df_without_future, location_id="0.5_0.5", as_of="2004-02-01", shrinkage_k=5
    )
    assert sig_with.mean == pytest.approx(sig_without.mean)
    assert sig_with.n_observations == sig_without.n_observations


def test_never_observed_location_returns_global_theta_with_zero_weight() -> None:
    df = pd.concat(
        [
            _rows("0.5_0.5", _monthly_entries("2003-01-01", 20, [1.0] * 20)),
            _rows("9.5_9.5", []),
        ],
        ignore_index=True,
    )
    sig = compute_location_signature(df, location_id="9.5_9.5", as_of="2004-06-01", shrinkage_k=5)
    assert sig.n_observations == 0
    assert sig.shrinkage_weight == pytest.approx(0.0)
    # w=0 -> signature should equal the global estimate exactly.
    assert sig.mean == pytest.approx(1.0, abs=1e-6)


# --- Shrinkage weight behavior ---------------------------------------------


def test_shrinkage_weight_increases_monotonically_with_evidence() -> None:
    weights = []
    for n in (0, 1, 5, 20, 50):
        entries = _monthly_entries("2003-01-01", n, [1.0 + 0.01 * i for i in range(n)])
        df = _rows("0.5_0.5", entries + [("2010-01-01", None)])
        sig = compute_location_signature(
            df, location_id="0.5_0.5", as_of="2010-01-01", shrinkage_k=10
        )
        weights.append(sig.shrinkage_weight)

    assert weights == sorted(weights)
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] < 1.0


def test_shrinkage_weight_asymptotes_toward_one() -> None:
    entries = _monthly_entries("1990-01-01", 400, [1.0] * 400)
    df = _rows("0.5_0.5", entries + [("2025-01-01", None)])
    sig = compute_location_signature(df, location_id="0.5_0.5", as_of="2025-01-01", shrinkage_k=10)
    assert sig.shrinkage_weight > 0.9


# --- A-014 regression test: shrinkage beats naive on a sparse fixture -----


def _sparse_multi_location_fixture(
    rng: np.random.Generator, n_locations: int = 25, n_months: int = 36
):
    """Many locations, few observations each -- the exact condition A-014
    measured (naive per-(location, calendar-month) climatology overfits
    when most cells have very little data)."""
    frames = []
    location_true_means = {}
    for i in range(n_locations):
        lat, lon = float(i), float(i) + 0.5
        loc_id = f"{lat}_{lon}"
        true_mean = rng.normal(loc=0.0, scale=0.3)
        location_true_means[loc_id] = true_mean
        values = true_mean + rng.normal(scale=1.0, size=n_months)
        entries = _monthly_entries("2004-01-01", n_months, values.tolist())
        frames.append(_rows(loc_id, entries))
    return pd.concat(frames, ignore_index=True), location_true_means


def test_a014_regression_shrinkage_beats_naive_out_of_fold() -> None:
    rng = np.random.default_rng(20260814)
    df, location_true_means = _sparse_multi_location_fixture(rng)

    # Split each location's series into "train" (first 24 months) and
    # "held-out" (last 12 months) -- naive per-location mean overfits the
    # train slice; the shrunk signature should generalize better.
    train_cutoff = pd.Timestamp("2004-01-01") + pd.DateOffset(months=24)
    train_df = df[pd.to_datetime(df["time"]) < train_cutoff]
    held_out_df = df[pd.to_datetime(df["time"]) >= train_cutoff]

    naive_means = train_df.groupby("location_id")["TWS_t"].mean()

    shrunk_errors = []
    naive_errors = []
    as_of = train_cutoff
    for location_id, held_out_rows in held_out_df.groupby("location_id"):
        sig = compute_location_signature(df, location_id=location_id, as_of=as_of, shrinkage_k=30)
        shrunk_pred = sig.mean
        naive_pred = naive_means.loc[location_id]

        actual = held_out_rows["TWS_t"].to_numpy()
        shrunk_errors.extend((actual - shrunk_pred) ** 2)
        naive_errors.extend((actual - naive_pred) ** 2)

    shrunk_rmse = float(np.sqrt(np.mean(shrunk_errors)))
    naive_rmse = float(np.sqrt(np.mean(naive_errors)))

    assert shrunk_rmse < naive_rmse


# --- Vectorized batch variant: consistency with the single-row function ---


def _multi_location_signature_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for loc_idx, (lat, lon) in enumerate([(1.5, 2.5), (10.5, -20.5), (-5.5, 100.5)]):
        n_months = 30
        values = (1.0 + 0.03 * np.arange(n_months) + rng.normal(scale=0.05, size=n_months)).tolist()
        mask_start = 10 + loc_idx * 2
        for j in range(mask_start, mask_start + 3):
            values[j] = None
        entries = _monthly_entries("2003-01-01", n_months, values)
        df = _rows(f"{lat}_{lon}", entries)
        df["SPEI_12_t"] = rng.normal(size=n_months)
        df["SOIL_MOISTURE_t"] = rng.normal(size=n_months)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_batch_variant_matches_single_row_function() -> None:
    df = _multi_location_signature_fixture()
    batch = compute_location_signatures(df, as_of_column="time", shrinkage_k=15)

    assert len(batch) == len(df)

    for location_id in df["location_id"].unique():
        loc_rows = df[df["location_id"] == location_id]
        for _, row in loc_rows.iterrows():
            expected = compute_location_signature(
                df, location_id=location_id, as_of=row["time"], shrinkage_k=15
            )
            actual = batch.loc[row.name]

            assert actual["location_id"] == expected.location_id
            assert actual["n_observations"] == expected.n_observations
            _assert_close(actual["shrinkage_weight"], expected.shrinkage_weight)
            _assert_close(actual["mean"], expected.mean)
            _assert_close(actual["std"], expected.std)
            _assert_close(actual["seasonality_amplitude"], expected.seasonality_amplitude)
            for lag in (1, 3, 6, 12):
                _assert_close(actual[f"acf_{lag}"], getattr(expected, f"acf_{lag}"))


def _assert_close(actual: object, expected: object) -> None:
    actual_missing = actual is None or (isinstance(actual, float) and np.isnan(actual))
    expected_missing = expected is None or (isinstance(expected, float) and np.isnan(expected))
    if actual_missing or expected_missing:
        assert actual_missing == expected_missing, (actual, expected)
        return
    assert float(actual) == pytest.approx(float(expected), rel=1e-4, abs=1e-6)


def test_batch_variant_empty_frame_returns_empty_dataframe_with_columns() -> None:
    df = pd.DataFrame(columns=["time", "lat", "lon", "location_id", "TWS_t"])
    result = compute_location_signatures(df)
    assert len(result) == 0
    assert "mean" in result.columns
    assert "shrinkage_weight" in result.columns


# --- Covariate response fallback -------------------------------------------


def test_covariate_response_none_when_column_absent() -> None:
    df = _rows("0.5_0.5", _monthly_entries("2004-01-01", 10, [1.0] * 10) + [("2004-11-01", None)])
    sig = compute_location_signature(df, location_id="0.5_0.5", as_of="2004-11-01", shrinkage_k=5)
    assert sig.spei_response is None
    assert sig.soil_moisture_response is None


def test_covariate_response_present_when_column_available() -> None:
    rng = np.random.default_rng(1)
    entries = _monthly_entries("2004-01-01", 20, (1.0 + rng.normal(scale=0.1, size=20)).tolist())
    df = _rows("0.5_0.5", entries + [("2005-09-01", None)])
    df["SPEI_12_t"] = rng.normal(size=len(df))
    df["SOIL_MOISTURE_t"] = rng.normal(size=len(df))
    sig = compute_location_signature(df, location_id="0.5_0.5", as_of="2005-09-01", shrinkage_k=5)
    assert sig.spei_response is not None
    assert -1.0 <= sig.spei_response <= 1.0
    assert sig.soil_moisture_response is not None


# --- LocationSignature is frozen -------------------------------------------


def test_location_signature_is_frozen() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", None)])
    sig = compute_location_signature(df, location_id="0.5_0.5", as_of="2004-02-01")
    assert isinstance(sig, LocationSignature)
    with pytest.raises(Exception):
        sig.mean = 99.0  # type: ignore[misc]


# --- LocationSignatureTransformer: Transformer protocol --------------------


def test_location_signature_transformer_satisfies_protocol() -> None:
    assert isinstance(LocationSignatureTransformer(), Transformer)


def test_location_signature_transformer_fit_transform() -> None:
    df = _multi_location_signature_fixture()
    train_cutoff = pd.Timestamp("2003-01-01") + pd.DateOffset(months=20)
    train_df = df[pd.to_datetime(df["time"]) < train_cutoff]
    val_df = df[pd.to_datetime(df["time"]) >= train_cutoff]

    transformer = LocationSignatureTransformer(shrinkage_k=15)
    transformer.fit(train_df)
    result = transformer.transform(val_df)

    assert len(result) == len(val_df)
    assert list(result.index) == list(val_df.index)
    assert "mean" in result.columns


def test_location_signature_transformer_raises_before_fit() -> None:
    transformer = LocationSignatureTransformer()
    df = _rows("0.5_0.5", [("2004-01-01", 1.0)])
    with pytest.raises(RuntimeError):
        transformer.transform(df)
