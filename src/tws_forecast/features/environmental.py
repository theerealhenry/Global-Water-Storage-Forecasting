"""Environmental features — Project Phase 4 step 4.6.

Three ``features.base.Transformer`` implementations, per
``docs/PHASE4_EXECUTION_PLAN.md`` §4.6, all config-driven from
``configs/features/environmental.yaml``:

- :class:`SpeiDifferencingTransformer` — ``SPEI_XX_t - SPEI_XX_{t-lag}``
  style deltas, across every SPEI timescale present in the input frame
  (``SPEI_01_t``/``SPEI_03_t``/``SPEI_06_t``/``SPEI_12_t``) and every
  configured lag (``spei_diff_lags``, default ``(3, 6, 12)``).
- :class:`DroughtPersistenceTransformer` — consecutive months, ending at
  each row's own origin time inclusive, where ``SPEI_12_t`` (the covariate
  Project Phase 1 established as by far the strongest single predictor,
  reused from ``state.signatures.SPEI_COVARIATE_COLUMN``) is at or below
  ``drought_threshold``. Uses the same "reset-on-break, cumcount-within-
  group" vectorization ``state.reconstruction.build_state_snapshots`` uses
  for ``blackout_streak_length`` — the same mechanism, applied to a value
  threshold instead of a nullness check.
- :class:`SoilMoistureTrajectoryTransformer` — the same lag/trend treatment
  ``StateSnapshot`` gives ``TWS_t`` (last-known value, velocity, rolling
  trend), applied to ``SOIL_MOISTURE_t``. Implemented as its own compact
  per-location pass rather than generalizing
  ``state.reconstruction.build_state_snapshots`` to an arbitrary column —
  that module's schema and tests are already shipped for Project Phase 4
  step 4.1 specifically as TWS's state representation, and widening it to
  a second column is a larger, separate change not warranted by this one
  feature. ``rolling_slope_raw`` (the trend-fitting primitive, already
  promoted to public in ``state.reconstruction`` for step 4.5's
  ``TrailingTrendTransformer``) is reused directly, so the actual slope
  arithmetic is still not duplicated a third time.

SPEI/soil-moisture covariates are not subject to this project's masking
structure (only ``TWS_t`` is masked in ``Test.csv`` — Project Phase 1's
verified data dictionary) — a calendar-lag lookup that finds no row at all
for a given ``(location, period)`` returns ``NaN`` (insufficient history),
never a silent leak of a later value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tws_forecast.features.registry import load_feature_config
from tws_forecast.state.reconstruction import ensure_location_id, rolling_slope_raw
from tws_forecast.state.signatures import SPEI_COVARIATE_COLUMN

__all__ = [
    "SPEI_TIMESCALE_COLUMNS",
    "SoilMoistureColumn",
    "SpeiDifferencingTransformer",
    "DroughtPersistenceTransformer",
    "SoilMoistureTrajectoryTransformer",
]

#: Every SPEI timescale this project's data provides
#: (``docs/DATA_DICTIONARY.md``). ``SpeiDifferencingTransformer`` uses
#: whichever of these are actually present in its input frame, so a
#: fixture missing some timescales still works.
SPEI_TIMESCALE_COLUMNS: tuple[str, ...] = ("SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t")

SoilMoistureColumn = "SOIL_MOISTURE_t"


def _resolve_environmental_config(
    spei_diff_lags: tuple[int, ...] | None, drought_threshold: float | None
) -> tuple[tuple[int, ...], float]:
    if spei_diff_lags is not None and drought_threshold is not None:
        return spei_diff_lags, drought_threshold
    config = load_feature_config("environmental")
    return (
        spei_diff_lags if spei_diff_lags is not None else config.spei_diff_lags,
        drought_threshold if drought_threshold is not None else config.drought_threshold,
    )


def _dedup_combined(train_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df, df], ignore_index=False)
    combined = ensure_location_id(combined)
    # Deduplicate by (location_id, time) content, never by raw pandas
    # index -- see state.signatures.LocationSignatureTransformer's
    # identical fix for why index-based deduplication is unsafe.
    return combined.loc[~combined.duplicated(subset=["location_id", "time"], keep="last")]


class SpeiDifferencingTransformer:
    """``spei_{timescale}_diff_{lag}`` columns: each SPEI timescale's value
    at the row's own origin time minus its value ``lag`` months earlier
    (calendar lag, not last-observed lag -- SPEI is not blackout-masked, so
    there is no "last known" distinction to make here).

    ``NaN`` when the lagged month has no row for that location at all
    (insufficient history), never a leak of a later value.
    """

    def __init__(self, spei_diff_lags: tuple[int, ...] | None = None) -> None:
        self._spei_diff_lags, _ = _resolve_environmental_config(spei_diff_lags, -1.5)
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_df is None:
            raise RuntimeError("SpeiDifferencingTransformer.transform called before fit()")

        frame = ensure_location_id(df).copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frame["period"] = pd.PeriodIndex(frame["time"], freq="M")

        combined = _dedup_combined(self._train_df, df)
        combined["period"] = pd.PeriodIndex(combined["time"], freq="M")

        present_columns = [c for c in SPEI_TIMESCALE_COLUMNS if c in combined.columns]

        result = pd.DataFrame(index=df.index)
        for column in present_columns:
            lookup = combined[["location_id", "period", column]]
            for lag in self._spei_diff_lags:
                lagged_period = frame["period"] - lag
                lagged_lookup = lookup.rename(
                    columns={"period": "_lagged_period", column: "_lagged_value"}
                )
                merged = pd.DataFrame(
                    {
                        "location_id": frame["location_id"].values,
                        "_lagged_period": lagged_period.values,
                        "_row_order": np.arange(len(frame)),
                    }
                ).merge(lagged_lookup, on=["location_id", "_lagged_period"], how="left")
                merged = merged.sort_values("_row_order")

                timescale = column.replace("SPEI_", "").replace("_t", "").lower()
                result[f"spei_{timescale}_diff_{lag}"] = (
                    frame[column].to_numpy() - merged["_lagged_value"].to_numpy()
                )

        return result


class DroughtPersistenceTransformer:
    """``drought_persistence_run_length``: consecutive calendar months,
    ending at each row's own origin time inclusive, where
    ``state.signatures.SPEI_COVARIATE_COLUMN`` (SPEI_12) is at or below
    ``drought_threshold``. ``0`` when the current month is not itself in
    drought.
    """

    def __init__(self, drought_threshold: float | None = None) -> None:
        _, self._drought_threshold = _resolve_environmental_config((3, 6, 12), drought_threshold)
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_df is None:
            raise RuntimeError("DroughtPersistenceTransformer.transform called before fit()")

        frame = ensure_location_id(df).copy()
        frame["time"] = pd.to_datetime(frame["time"])

        combined = _dedup_combined(self._train_df, df)
        if SPEI_COVARIATE_COLUMN not in combined.columns:
            return pd.DataFrame(
                {"drought_persistence_run_length": np.zeros(len(df))}, index=df.index
            )

        pieces = []
        for location_id, group in combined.groupby("location_id", sort=False):
            pieces.append(self._location_run_length(group, location_id))
        panel = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame(columns=["location_id", "period", "drought_persistence_run_length"])
        )

        frame_periods = pd.PeriodIndex(frame["time"], freq="M")
        lookup = panel.set_index(["location_id", "period"])["drought_persistence_run_length"]
        keys = list(zip(frame["location_id"], frame_periods, strict=True))
        values = [lookup.get(key, 0.0) for key in keys]

        return pd.DataFrame({"drought_persistence_run_length": values}, index=df.index)

    def _location_run_length(self, group: pd.DataFrame, location_id: str) -> pd.DataFrame:
        group = group.sort_values("time")
        times = pd.to_datetime(group["time"])
        full_range = pd.period_range(start=times.min(), end=times.max(), freq="M")

        raw = pd.Series(
            group[SPEI_COVARIATE_COLUMN].astype(float).values,
            index=pd.PeriodIndex(times, freq="M"),
        )
        raw = raw.groupby(level=0).first().reindex(full_range)

        in_drought = raw <= self._drought_threshold
        in_drought = in_drought.fillna(False)  # missing covariate is not drought evidence

        not_in_drought = ~in_drought
        grp_id = not_in_drought.astype(int).cumsum()
        run_length = grp_id.groupby(grp_id).cumcount()

        return pd.DataFrame(
            {
                "location_id": location_id,
                "period": full_range,
                "drought_persistence_run_length": run_length.to_numpy(dtype=float),
            }
        )


class SoilMoistureTrajectoryTransformer:
    """``soil_moisture_last_known``, ``soil_moisture_velocity``,
    ``soil_moisture_trend_{window}`` -- the same last-known/velocity/rolling
    -trend treatment ``StateSnapshot`` gives ``TWS_t``, applied to
    ``SOIL_MOISTURE_t``.
    """

    def __init__(self, trend_window_months: tuple[int, ...] = (12,)) -> None:
        self._trend_window_months = trend_window_months
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_df is None:
            raise RuntimeError("SoilMoistureTrajectoryTransformer.transform called before fit()")

        frame = ensure_location_id(df).copy()
        frame["time"] = pd.to_datetime(frame["time"])

        combined = _dedup_combined(self._train_df, df)
        columns = ["last_known", "velocity"] + [f"trend_{w}" for w in self._trend_window_months]

        if SoilMoistureColumn not in combined.columns:
            return pd.DataFrame(
                {f"soil_moisture_{c}": np.full(len(df), np.nan) for c in columns}, index=df.index
            )

        pieces = []
        for location_id, group in combined.groupby("location_id", sort=False):
            pieces.append(self._location_trajectory(group, location_id))
        panel = pd.concat(pieces, ignore_index=True)

        frame_periods = pd.PeriodIndex(frame["time"], freq="M")
        panel_lookup = panel.set_index(["location_id", "period"])
        keys = list(zip(frame["location_id"], frame_periods, strict=True))

        result = {}
        for c in columns:
            col_lookup = panel_lookup[c]
            result[f"soil_moisture_{c}"] = [col_lookup.get(key, np.nan) for key in keys]

        return pd.DataFrame(result, index=df.index)

    def _location_trajectory(self, group: pd.DataFrame, location_id: str) -> pd.DataFrame:
        group = group.sort_values("time")
        times = pd.to_datetime(group["time"])
        full_range = pd.period_range(start=times.min(), end=times.max(), freq="M")

        raw = pd.Series(
            group[SoilMoistureColumn].astype(float).values,
            index=pd.PeriodIndex(times, freq="M"),
        )
        raw = raw.groupby(level=0).first().reindex(full_range)

        last_known = raw.ffill()
        obs = raw.dropna()
        prev_at_obs = obs.shift(1)
        velocity_at_obs = obs - prev_at_obs
        velocity_full = pd.Series(np.nan, index=full_range)
        velocity_full.loc[obs.index] = velocity_at_obs.values
        velocity_full = velocity_full.ffill()

        out = pd.DataFrame(
            {
                "location_id": location_id,
                "period": full_range,
                "last_known": last_known.to_numpy(),
                "velocity": velocity_full.to_numpy(),
            }
        )
        for window in self._trend_window_months:
            trend = raw.rolling(window=window, min_periods=1).apply(rolling_slope_raw, raw=True)
            out[f"trend_{window}"] = trend.to_numpy()

        return out
