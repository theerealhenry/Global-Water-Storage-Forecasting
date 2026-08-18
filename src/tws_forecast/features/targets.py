"""Target transformation comparison — Project Phase 4 step 4.7.

Five interchangeable target framings, each implementing the
:class:`TargetTransform` contract (``forward(df) -> pd.Series``,
``inverse(predictions, df) -> pd.Series``) rather than
``features.base.Transformer``'s ``fit``/``transform`` shape — per
``docs/PHASE4_EXECUTION_PLAN.md`` §4.7, a target transform has no
training-fold state of its own to learn (every baseline it needs --
last-known TWS, a shrinkage-regularized signature mean/std, a trend
extrapolation -- is itself already origin-time-indexed and recomputed
directly from whatever frame it's given, via ``state.reconstruction`` and
``state.signatures``). ``inverse`` must always undo ``forward`` exactly, so
a model trained in a transformed space can be scored on the real
``TWS(t+1)`` level the competition actually evaluates against, never
whatever space it happened to train in.

The five transforms, per the execution plan:

- :class:`LevelTargetTransform` -- ``target`` as-is (the control condition).
- :class:`DeltaTargetTransform` -- ``target - effective_current``, where
  ``effective_current`` is ``state.reconstruction.build_state_snapshots``'s
  own ``last_known_tws`` (which already equals the row's own ``TWS_t`` when
  observed, by ``StateSnapshot``'s construction -- so this one field
  correctly covers both the observed and masked case, the same fallback
  ``models.baselines.HybridPersistencePredictor`` uses).
- :class:`AnomalyTargetTransform` -- ``target - location_signature.mean``,
  the *shrinkage-regularized* mean (``state.signatures``), never the naive
  per-location mean A-014 showed overfits badly out-of-fold.
- :class:`TrendResidualTargetTransform` -- ``target - (last_known_tws +
  local_trend)``, i.e. residual against a naive one-month trend
  extrapolation.
- :class:`VolatilityNormalizedDeltaTargetTransform` -- ``(target -
  effective_current) / location_signature.std`` (shrinkage-regularized
  std, floored away from zero).

Every transform's round-trip property (``inverse(forward(df), df) ==
df["target"]``, exactly) holds by construction as long as the same
baseline is recomputed identically in both directions -- which it is,
since both calls are pure functions of the same ``df``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from tws_forecast.state.reconstruction import build_state_snapshots
from tws_forecast.state.signatures import compute_location_signatures

__all__ = [
    "TargetTransform",
    "LevelTargetTransform",
    "DeltaTargetTransform",
    "AnomalyTargetTransform",
    "TrendResidualTargetTransform",
    "VolatilityNormalizedDeltaTargetTransform",
    "TARGET_TRANSFORMS",
]

#: Floor applied to a shrinkage-regularized std before dividing by it --
#: a location with a near-zero (or, pre-shrinkage, exactly zero for a
#: single-observation location) std would otherwise blow up the
#: volatility-normalized delta.
MIN_STD = 1e-6


@runtime_checkable
class TargetTransform(Protocol):
    """The two-method contract every target framing implements -- see the
    module docstring for why this is a distinct, smaller protocol than
    ``features.base.Transformer``."""

    def forward(self, df: pd.DataFrame) -> pd.Series: ...

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series: ...


def _as_series(predictions: pd.Series | np.ndarray, index: pd.Index) -> pd.Series:
    if isinstance(predictions, pd.Series):
        return predictions
    return pd.Series(np.asarray(predictions, dtype=float), index=index)


class LevelTargetTransform:
    """``target`` as-is -- the current default / control condition."""

    def forward(self, df: pd.DataFrame) -> pd.Series:
        return df["target"].astype(float).copy()

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series:
        return _as_series(predictions, df.index)


class DeltaTargetTransform:
    """``target - effective_current``, ``effective_current`` = last-known
    TWS (equals the row's own ``TWS_t`` when observed, by construction)."""

    def _effective_current(self, df: pd.DataFrame) -> pd.Series:
        snapshots = build_state_snapshots(df)
        return snapshots.loc[df.index, "last_known_tws"].astype(float)

    def forward(self, df: pd.DataFrame) -> pd.Series:
        return df["target"].astype(float) - self._effective_current(df)

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series:
        return _as_series(predictions, df.index) + self._effective_current(df)


class AnomalyTargetTransform:
    """``target - location_signature.mean`` (shrinkage-regularized)."""

    def __init__(self, shrinkage_k: int | None = None) -> None:
        self._shrinkage_k = shrinkage_k

    def _signature_mean(self, df: pd.DataFrame) -> pd.Series:
        signatures = compute_location_signatures(df, shrinkage_k=self._shrinkage_k)
        return signatures.loc[df.index, "mean"].astype(float)

    def forward(self, df: pd.DataFrame) -> pd.Series:
        return df["target"].astype(float) - self._signature_mean(df)

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series:
        return _as_series(predictions, df.index) + self._signature_mean(df)


class TrendResidualTargetTransform:
    """``target - (last_known_tws + local_trend)`` -- residual against a
    naive one-month trend extrapolation. Falls back to pure persistence
    (``local_trend`` treated as ``0``) when the trend is undefined (fewer
    than two observed points in the trailing window)."""

    def _baseline(self, df: pd.DataFrame) -> pd.Series:
        snapshots = build_state_snapshots(df)
        aligned = snapshots.loc[df.index]
        return aligned["last_known_tws"].astype(float) + aligned["local_trend"].fillna(0.0)

    def forward(self, df: pd.DataFrame) -> pd.Series:
        return df["target"].astype(float) - self._baseline(df)

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series:
        return _as_series(predictions, df.index) + self._baseline(df)


class VolatilityNormalizedDeltaTargetTransform:
    """``(target - effective_current) / location_signature.std``
    (shrinkage-regularized std, floored at :data:`MIN_STD`)."""

    def __init__(self, shrinkage_k: int | None = None) -> None:
        self._shrinkage_k = shrinkage_k

    def _effective_current(self, df: pd.DataFrame) -> pd.Series:
        snapshots = build_state_snapshots(df)
        return snapshots.loc[df.index, "last_known_tws"].astype(float)

    def _std(self, df: pd.DataFrame) -> pd.Series:
        signatures = compute_location_signatures(df, shrinkage_k=self._shrinkage_k)
        return signatures.loc[df.index, "std"].astype(float).clip(lower=MIN_STD)

    def forward(self, df: pd.DataFrame) -> pd.Series:
        return (df["target"].astype(float) - self._effective_current(df)) / self._std(df)

    def inverse(self, predictions: pd.Series | np.ndarray, df: pd.DataFrame) -> pd.Series:
        return _as_series(predictions, df.index) * self._std(df) + self._effective_current(df)


#: Registry for step 4.9's controlled head-to-head comparison -- iterate
#: over this rather than hardcoding the five class names at each call site.
TARGET_TRANSFORMS: dict[str, TargetTransform] = {
    "level": LevelTargetTransform(),
    "delta": DeltaTargetTransform(),
    "anomaly": AnomalyTargetTransform(),
    "trend_residual": TrendResidualTargetTransform(),
    "volatility_normalized_delta": VolatilityNormalizedDeltaTargetTransform(),
}
