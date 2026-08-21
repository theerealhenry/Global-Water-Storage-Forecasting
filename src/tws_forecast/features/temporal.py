"""Seasonal/trend features — Project Phase 4 step 4.5.

Two independent ``features.base.Transformer`` implementations:

- :class:`TrailingTrendTransformer` — a trailing linear-trend slope per
  location, at each of ``configs/features/temporal.yaml``'s
  ``trend_window_months`` (default ``(12, 24)``). Deliberately reuses
  ``state.reconstruction.build_state_snapshots``'s own ``local_trend``
  computation rather than re-deriving the rolling-OLS-slope math a second
  time — per ``docs/PHASE4_EXECUTION_PLAN.md`` §4.5's explicit instruction
  not to duplicate it. ``build_state_snapshots(..., trailing_windows=(w,))``
  computes ``local_trend`` over exactly one window (``w``, the max of
  whatever tuple is passed), so this transformer calls it once per
  configured window and renames the result — a *feature* column, distinct
  from (but computed identically to) ``StateSnapshot.local_trend`` itself.
- :class:`MonthHemisphereTransformer` — a calendar x hemisphere interaction.
  Stateless (``fit`` is a no-op): a location's hemisphere never changes, and
  the calendar month of a row is already known from its own ``time`` column,
  so nothing needs to be learned from a training fold.

Both address A-011 directly, per ``docs/ASSUMPTIONS.md``: the real test set
omits October entirely and has a 2x row-share imbalance across the other
eleven months, so ``tests/test_temporal_features.py`` checks behavior on
exactly those under/zero-represented months rather than only an aggregate
pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tws_forecast.features.registry import load_feature_config
from tws_forecast.state.reconstruction import build_state_snapshots, ensure_location_id

__all__ = ["TrailingTrendTransformer", "MonthHemisphereTransformer"]


def _resolve_trend_windows(trend_window_months: tuple[int, ...] | None) -> tuple[int, ...]:
    if trend_window_months is not None:
        return trend_window_months
    return load_feature_config("temporal").trend_window_months


class TrailingTrendTransformer:
    """One ``trend_slope_{window}`` column per configured trailing window
    (months) — the rolling OLS slope of observed ``TWS_t``, reusing
    ``state.reconstruction.build_state_snapshots``'s ``local_trend`` field.

    ``fit(train_df)`` stores the training frame as historical context, the
    same pattern ``state.signatures.LocationSignatureTransformer`` and
    ``state.spatial_history.SpatialHistoryTransformer`` use.
    ``transform(df)`` computes each window's trend over the union of the
    training history and ``df`` itself, then returns only ``df``'s own rows
    — origin-time-indexed by construction, since it delegates entirely to
    ``build_state_snapshots``.
    """

    def __init__(self, trend_window_months: tuple[int, ...] | None = None) -> None:
        self._trend_window_months = _resolve_trend_windows(trend_window_months)
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def transform(
        self, df: pd.DataFrame, precomputed_state_panels: dict[int, pd.DataFrame] | None = None
    ) -> pd.DataFrame:
        """Compute ``trend_slope_{window}`` for every configured window.

        ``precomputed_state_panels`` is an optional performance escape
        hatch: ``{window: panel}`` where ``panel`` is already
        ``build_state_snapshots(combined, as_of_column="time",
        trailing_windows=(window,))``'s own output for this call's
        ``train_df ∪ df`` -- lets a composing caller
        (``features.assemble.build_feature_matrix``) skip a redundant call
        for whichever window it already had to compute anyway (Project
        Phase 4 step 4.9's proof run found this a real, multi-minute-
        per-call cost at ~15,715-location scale, paid once per window,
        every ``fit``/``predict`` cycle). Any window *not* present in this
        dict is computed exactly as before. Omitting the parameter entirely
        (the default) reproduces the original, fully-self-contained
        behavior -- every existing caller, including every test in
        ``tests/test_temporal_features.py``, is unaffected.
        """
        if self._train_df is None:
            raise RuntimeError("TrailingTrendTransformer.transform called before fit()")

        combined = pd.concat([self._train_df, df], ignore_index=False)
        combined = ensure_location_id(combined)
        # Deduplicate by (location_id, time) content, never by raw pandas
        # index -- see state.signatures.LocationSignatureTransformer's
        # identical fix for why index-based deduplication is unsafe.
        combined = combined.loc[~combined.duplicated(subset=["location_id", "time"], keep="last")]

        precomputed_state_panels = precomputed_state_panels or {}

        result = pd.DataFrame(index=df.index)
        for window in self._trend_window_months:
            snapshots = precomputed_state_panels.get(window)
            if snapshots is not None:
                if len(snapshots) != len(combined) or set(snapshots.index) != set(combined.index):
                    raise ValueError(
                        f"TrailingTrendTransformer.transform: precomputed_state_panels[{window}] "
                        "does not cover this call's own train_df ∪ df -- it was not built over "
                        "the same combined frame this call constructs."
                    )
            else:
                # build_state_snapshots returns a frame indexed identically to
                # its input (`combined`, whose index is the preserved union of
                # train_df's and df's own original indices) -- no re-indexing
                # needed before selecting df's own rows back out.
                snapshots = build_state_snapshots(
                    combined, as_of_column="time", trailing_windows=(window,)
                )
            result[f"trend_slope_{window}"] = snapshots.loc[df.index, "local_trend"]

        return result


#: Southern-hemisphere locations are phase-shifted by six months before
#: cyclical encoding, so a Northern-Hemisphere January (deep winter) and a
#: Southern-Hemisphere July (also deep winter) map to the same encoded
#: position -- directly targeting the risk ARCHITECTURE.md Section 7 flags:
#: a single global seasonal signal misrepresenting the Southern Hemisphere's
#: opposite phase.
_HEMISPHERE_PHASE_SHIFT_MONTHS = 6


class MonthHemisphereTransformer:
    """Stateless month x hemisphere cyclical interaction.

    Produces ``month_hemisphere_sin``/``month_hemisphere_cos``: the row's
    own calendar month, shifted by six months for Southern-Hemisphere
    locations (``lat < 0``) before the usual sin/cos cyclical encoding, so
    both hemispheres' summers (and winters) land at the same encoded
    position rather than six months apart.
    """

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # stateless: hemisphere and calendar month need no training-fold statistics.

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        times = pd.to_datetime(df["time"])
        calendar_month = times.dt.month.to_numpy()
        is_southern = df["lat"].to_numpy() < 0

        effective_month = np.where(
            is_southern,
            ((calendar_month - 1 + _HEMISPHERE_PHASE_SHIFT_MONTHS) % 12) + 1,
            calendar_month,
        )
        angle = 2 * np.pi * (effective_month - 1) / 12.0

        return pd.DataFrame(
            {
                "month_hemisphere_sin": np.sin(angle),
                "month_hemisphere_cos": np.cos(angle),
            },
            index=df.index,
        )
