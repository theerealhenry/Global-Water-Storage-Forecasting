"""Tests for tws_forecast.state.reconstruction.StateSnapshot /
build_state_snapshot / build_state_snapshots — Project Phase 4 step 4.1,
per docs/PHASE4_EXECUTION_PLAN.md §4.1 and docs/adr/0006-statesnapshot-trajectory-fields.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tws_forecast.state.reconstruction import (
    DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS,
    DEFAULT_MIN_EVIDENCE_OBSERVATIONS,
    StateSnapshot,
    build_state_snapshot,
    build_state_snapshots,
)


def _rows(location_id: str, entries: list[tuple[str, float | None]]) -> pd.DataFrame:
    """Build a minimal fixture frame: one row per (time, TWS_t) entry, all
    at the same location. ``lat``/``lon`` are derived so location_id_from_lat_lon
    round-trips to ``location_id`` exactly."""
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


# --- Single-row build_state_snapshot: basic shape -----------------------


def test_never_observed_location_is_partially_reconstructed() -> None:
    df = _rows("1.5_2.5", [("2004-01-01", None), ("2004-02-01", None)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="1.5_2.5")

    assert snap.last_known_tws is None
    assert snap.last_known_time is None
    assert snap.months_since_observation is None
    assert snap.previous_known_tws is None
    assert snap.second_previous_known_tws is None
    assert snap.historical_delta is None
    assert snap.state_acceleration is None
    assert snap.state_status == "PARTIALLY_RECONSTRUCTED"


def test_observed_current_month_is_observed_status() -> None:
    df = _rows("1.5_2.5", [("2004-01-01", 1.0), ("2004-02-01", 1.2)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="1.5_2.5")

    assert snap.state_status == "OBSERVED"
    assert snap.last_known_tws == pytest.approx(1.2)
    assert snap.last_known_time == pd.Timestamp("2004-02-01")
    assert snap.months_since_observation == 0
    assert snap.previous_known_tws == pytest.approx(1.0)
    assert snap.historical_delta == pytest.approx(0.2)


def test_reconstructed_when_recent_and_enough_evidence() -> None:
    # 30 consecutive observed months (>= DEFAULT_MIN_EVIDENCE_OBSERVATIONS),
    # then the current (as_of) month is masked but recent (gap=1).
    entries = [
        (pd.Timestamp("2003-01-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
        for i in range(30)
    ]
    rows = [(t, 1.0 + 0.01 * i) for i, t in enumerate(entries)]
    rows.append(("2005-07-01", None))  # one month after the 30th observed month
    df = _rows("0.5_0.5", rows)

    snap = build_state_snapshot(df, as_of="2005-07-01", location_id="0.5_0.5")

    assert snap.state_status == "RECONSTRUCTED"
    assert snap.months_since_observation == 1
    assert snap.last_known_tws == pytest.approx(1.0 + 0.01 * 29)


def test_partially_reconstructed_when_gap_too_large() -> None:
    df = _rows(
        "0.5_0.5",
        [
            ("2003-01-01", 1.0),
            *[
                (
                    (pd.Timestamp("2003-02-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
                    None,
                )
                for i in range(DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS + 2)
            ],
        ],
    )
    as_of = (
        pd.Timestamp("2003-02-01") + pd.DateOffset(months=DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS + 1)
    ).strftime("%Y-%m-%d")
    snap = build_state_snapshot(df, as_of=as_of, location_id="0.5_0.5")

    assert snap.state_status == "PARTIALLY_RECONSTRUCTED"
    assert snap.months_since_observation is not None
    assert snap.months_since_observation > DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS


def test_partially_reconstructed_when_insufficient_evidence_despite_recency() -> None:
    # Observed once, one month before as_of — recent, but nowhere near
    # DEFAULT_MIN_EVIDENCE_OBSERVATIONS worth of history.
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", None)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="0.5_0.5")

    assert snap.months_since_observation == 1
    assert snap.months_since_observation <= DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS
    assert snap.state_status == "PARTIALLY_RECONSTRUCTED"


# --- Boundary condition: time == as_of is included, not excluded --------


def test_row_exactly_at_as_of_is_included_not_excluded() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 2.0)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="0.5_0.5")

    # If the row at as_of were excluded, last_known_tws would be 1.0 (Jan),
    # not 2.0 (Feb) -- this pins the documented <= semantics directly.
    assert snap.last_known_tws == pytest.approx(2.0)
    assert snap.state_status == "OBSERVED"


def test_rows_after_as_of_are_never_used() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 2.0), ("2004-03-01", 999.0)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="0.5_0.5")

    assert snap.last_known_tws == pytest.approx(2.0)
    assert snap.previous_known_tws == pytest.approx(1.0)


# --- Trajectory: velocity, acceleration (ADR-0006) -----------------------


def test_trajectory_velocity_and_acceleration_arithmetic() -> None:
    # Observed values 1.0, 1.5, 2.7 at Jan/Feb/Mar -> delta(prev)=0.5,
    # delta(last)=1.2, acceleration = 1.2 - 0.5 = 0.7.
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 1.5), ("2004-03-01", 2.7)])
    snap = build_state_snapshot(df, as_of="2004-03-01", location_id="0.5_0.5")

    assert snap.last_known_tws == pytest.approx(2.7)
    assert snap.previous_known_tws == pytest.approx(1.5)
    assert snap.second_previous_known_tws == pytest.approx(1.0)
    assert snap.historical_delta == pytest.approx(1.2)
    assert snap.state_acceleration == pytest.approx(0.7)


def test_state_acceleration_none_with_fewer_than_three_observations() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0), ("2004-02-01", 1.5)])
    snap = build_state_snapshot(df, as_of="2004-02-01", location_id="0.5_0.5")

    assert snap.second_previous_known_tws is None
    assert snap.state_acceleration is None
    assert snap.historical_delta == pytest.approx(0.5)


# --- Blackout streak, observation density --------------------------------


def test_blackout_streak_length_counts_consecutive_masked_months() -> None:
    df = _rows(
        "0.5_0.5",
        [
            ("2004-01-01", 1.0),
            ("2004-02-01", None),
            ("2004-03-01", None),
            ("2004-04-01", None),
        ],
    )
    snap = build_state_snapshot(df, as_of="2004-04-01", location_id="0.5_0.5")
    assert snap.blackout_streak_length == 3

    snap_observed = build_state_snapshot(df, as_of="2004-01-01", location_id="0.5_0.5")
    assert snap_observed.blackout_streak_length == 0


def test_observation_density_over_trailing_window() -> None:
    # 12-month window, 3 of 12 months observed.
    entries = []
    for i in range(12):
        t = (pd.Timestamp("2004-01-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
        value = 1.0 if i in (0, 5, 11) else None
        entries.append((t, value))
    df = _rows("0.5_0.5", entries)
    snap = build_state_snapshot(
        df, as_of="2004-12-01", location_id="0.5_0.5", trailing_windows=(12,)
    )
    assert snap.observation_density[12] == pytest.approx(3 / 12)


# --- ACF -------------------------------------------------------------------


def test_acf_lag1_near_one_for_perfectly_linear_series() -> None:
    entries = [
        (
            (pd.Timestamp("2004-01-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
            float(i),
        )
        for i in range(20)
    ]
    df = _rows("0.5_0.5", entries)
    snap = build_state_snapshot(df, as_of=entries[-1][0], location_id="0.5_0.5")
    assert snap.acf_1_3_6_12[1] is not None
    assert snap.acf_1_3_6_12[1] == pytest.approx(1.0, abs=1e-6)


def test_acf_none_with_insufficient_history() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0)])
    snap = build_state_snapshot(df, as_of="2004-01-01", location_id="0.5_0.5")
    for lag in (1, 3, 6, 12):
        assert snap.acf_1_3_6_12[lag] is None


# --- location_signature is always None from this module -------------------


def test_location_signature_always_none() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0)])
    snap = build_state_snapshot(df, as_of="2004-01-01", location_id="0.5_0.5")
    assert snap.location_signature is None


# --- Vectorized batch variant: consistency with the single-row function ---


def _multi_location_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = []
    for loc_idx, (lat, lon) in enumerate([(1.5, 2.5), (10.5, -20.5), (-5.5, 100.5)]):
        n_months = 40
        times = [(pd.Timestamp("2003-01-01") + pd.DateOffset(months=i)) for i in range(n_months)]
        values = 1.0 + 0.05 * np.arange(n_months) + rng.normal(scale=0.05, size=n_months)
        # Mask out a contiguous block to exercise blackout/reconstruction paths.
        mask_start, mask_len = 15 + loc_idx * 3, 4
        values = values.astype(object)
        for j in range(mask_start, mask_start + mask_len):
            values[j] = None
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "lat": lat,
                    "lon": lon,
                    "location_id": f"{lat}_{lon}",
                    "TWS_t": [None if v is None else float(v) for v in values],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_batch_variant_matches_single_row_function() -> None:
    df = _multi_location_fixture()
    batch = build_state_snapshots(df, as_of_column="time")

    assert len(batch) == len(df)

    # Spot-check every row for a couple of locations to keep this fast.
    for location_id in df["location_id"].unique():
        loc_rows = df[df["location_id"] == location_id]
        for _, row in loc_rows.iterrows():
            expected = build_state_snapshot(df, as_of=row["time"], location_id=location_id)
            actual = batch.loc[row.name]

            assert actual["location_id"] == expected.location_id
            _assert_scalar_close(actual["last_known_tws"], expected.last_known_tws)
            _assert_scalar_close(
                actual["months_since_observation"], expected.months_since_observation
            )
            _assert_scalar_close(actual["previous_known_tws"], expected.previous_known_tws)
            _assert_scalar_close(
                actual["second_previous_known_tws"], expected.second_previous_known_tws
            )
            _assert_scalar_close(actual["historical_delta"], expected.historical_delta)
            _assert_scalar_close(actual["state_acceleration"], expected.state_acceleration)
            _assert_scalar_close(actual["blackout_streak_length"], expected.blackout_streak_length)
            assert actual["state_status"] == expected.state_status
            for lag in (1, 3, 6, 12):
                _assert_scalar_close(actual[f"acf_lag{lag}"], expected.acf_1_3_6_12[lag])
            for w in (12, 24):
                _assert_scalar_close(
                    actual[f"observation_density_{w}"], expected.observation_density[w]
                )


def _assert_scalar_close(actual: object, expected: object) -> None:
    actual_is_missing = actual is None or (isinstance(actual, float) and np.isnan(actual))
    expected_is_missing = expected is None or (isinstance(expected, float) and np.isnan(expected))
    if actual_is_missing or expected_is_missing:
        assert actual_is_missing == expected_is_missing, (actual, expected)
        return
    assert float(actual) == pytest.approx(float(expected), rel=1e-6, abs=1e-9)


def test_batch_variant_empty_frame_returns_empty_dataframe_with_columns() -> None:
    df = pd.DataFrame(columns=["time", "lat", "lon", "location_id", "TWS_t"])
    result = build_state_snapshots(df)
    assert len(result) == 0
    assert "last_known_tws" in result.columns
    assert "acf_lag1" in result.columns
    assert "observation_density_12" in result.columns


def test_batch_variant_never_observed_location() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", None), ("2004-02-01", None)])
    result = build_state_snapshots(df)
    assert result["state_status"].eq("PARTIALLY_RECONSTRUCTED").all()
    assert result["last_known_tws"].isna().all()


# --- state_status classification via three hand-built fixtures ------------


def test_state_status_three_way_classification() -> None:
    observed_df = _rows("0.5_0.5", [("2004-01-01", 1.0)])
    observed_snap = build_state_snapshot(observed_df, as_of="2004-01-01", location_id="0.5_0.5")
    assert observed_snap.state_status == "OBSERVED"

    entries = [
        (pd.Timestamp("2003-01-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
        for i in range(25)
    ]
    reconstructed_rows = [(t, 1.0) for t in entries]
    reconstructed_rows.append(("2005-02-01", None))
    reconstructed_df = _rows("0.5_0.5", reconstructed_rows)
    reconstructed_snap = build_state_snapshot(
        reconstructed_df, as_of="2005-02-01", location_id="0.5_0.5"
    )
    assert reconstructed_snap.state_status == "RECONSTRUCTED"

    partial_df = _rows("0.5_0.5", [("2004-01-01", None)])
    partial_snap = build_state_snapshot(partial_df, as_of="2004-01-01", location_id="0.5_0.5")
    assert partial_snap.state_status == "PARTIALLY_RECONSTRUCTED"


def test_state_snapshot_is_frozen() -> None:
    df = _rows("0.5_0.5", [("2004-01-01", 1.0)])
    snap = build_state_snapshot(df, as_of="2004-01-01", location_id="0.5_0.5")
    assert isinstance(snap, StateSnapshot)
    with pytest.raises(Exception):
        snap.last_known_tws = 99.0  # type: ignore[misc]


def test_default_thresholds_are_module_constants() -> None:
    assert DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS > 0
    assert DEFAULT_MIN_EVIDENCE_OBSERVATIONS > 0
