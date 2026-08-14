"""Shrinkage-regularized historical location signatures — Project Phase 4
step 4.2.

Every location-level aggregate this project computes is shrunk toward a
global estimate, never used as a naive per-location statistic —
``theta_hat_location = w * theta_location + (1 - w) * theta_global``,
``w = n / (n + shrinkage_k)`` — per ``docs/ARCHITECTURE.md`` §10/§17 and
this project's own held-out evidence for *why*: naive per-``(location,
calendar-month)`` climatology (Baseline C, Project Phase 3) scored 1.0796
RMSE out-of-fold, *worse* than a plain global mean (0.8740), because most
of the ~15,715 x 12 cells it estimates have very little data per fold
(``docs/ASSUMPTIONS.md`` A-014). ``test_a014_regression_shrinkage_beats_naive``
in this module's test file is the direct confirmation A-014's own
"validation experiment" column asked for.

A different boundary rule than ``state.reconstruction.StateSnapshot``,
**deliberately**: ``docs/ARCHITECTURE.md`` §4's signature invariant requires
history *strictly before* ``as_of`` (``time < as_of``), not
``time <= as_of``. A ``StateSnapshot`` legitimately includes the origin
month's own observation (if present) as part of "what we currently know."
A location *signature*, by contrast, is meant to be a climatological
baseline computed independently of the very month it may later be used to
describe an anomaly against (e.g. Project Phase 4 step 4.7's
``anomaly = target - location_signature.mean``) — including the current
month in its own baseline would make that comparison circular.

``mean``/``std``/``trend``/``seasonality_amplitude``/``acf_1_3_6_12`` reuse
the same per-location autocorrelation logic already promoted into
``state.reconstruction`` (``compute_acf_at_lags``) rather than a third
notebook copy-paste, per this project's own established pattern
(``run_tier3_sequential_state`` was the last thing promoted this way).

Two build entry points, mirroring ``state.reconstruction``'s
``build_state_snapshot``/``build_state_snapshots`` pair:

- :func:`compute_location_signature` — one location, one ``as_of``. Slow but
  simple; the correctness reference the vectorized batch variant is tested
  against.
- :func:`compute_location_signatures` — every row of a frame at once,
  vectorized per-location (never a Python loop calling the single-location
  function once per row). The real path Phase 4's feature-assembly pipeline
  (step 4.9) uses.

:class:`LocationSignatureTransformer` wraps the batch function in the
``features.base.Transformer`` protocol (``fit``/``transform``) so it
composes uniformly with every other Project Phase 4 feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tws_forecast.features.registry import load_feature_config
from tws_forecast.state.reconstruction import (
    ACF_LAGS,
    compute_acf_at_lags,
    ensure_location_id,
    month_diff,
)

__all__ = [
    "LocationSignature",
    "SPEI_COVARIATE_COLUMN",
    "SOIL_MOISTURE_COVARIATE_COLUMN",
    "compute_location_signature",
    "compute_location_signatures",
    "LocationSignatureTransformer",
]

#: SPEI_12 is used for ``spei_response`` rather than every SPEI timescale --
#: Project Phase 1 established it as by far the strongest single covariate
#: (``docs/PROJECT_PLAN.md`` "Key findings"), and ``LocationSignature`` keeps
#: exactly one ``spei_response`` field per ``docs/ARCHITECTURE.md`` §4's
#: named list, not one per timescale.
SPEI_COVARIATE_COLUMN = "SPEI_12_t"
SOIL_MOISTURE_COVARIATE_COLUMN = "SOIL_MOISTURE_t"


@dataclass(frozen=True)
class LocationSignature:
    """A shrinkage-regularized summary of one location's historical water
    state, as of one origin time — every field already shrunk toward the
    matching global (all-location) estimate; there is no "raw" version of
    this object.

    Attributes
    ----------
    location_id, as_of:
        The key this signature answers "what is this location's climatology"
        for. ``as_of`` here means "computed from history strictly before
        this time" (see the module docstring for why this differs from
        ``StateSnapshot.as_of``'s inclusive semantics).
    mean, std:
        Shrunk mean/population-std of observed ``TWS_t`` history.
    trend:
        Shrunk long-run OLS slope of observed ``TWS_t`` against time (TWS
        units per month) over *all* available history — distinct from
        ``StateSnapshot.local_trend``'s bounded trailing window, which
        answers a different question ("how is this location moving
        *right now*") from this field's ("what is this location's
        long-run climatological trend").
    seasonality_amplitude:
        Shrunk ``max - min`` of the location's per-calendar-month mean TWS.
    acf_1, acf_3, acf_6, acf_12:
        Shrunk autocorrelation at each lag (months), reusing
        ``state.reconstruction.compute_acf_at_lags``.
    spei_response, soil_moisture_response:
        Shrunk Pearson correlation between observed ``TWS_t`` and
        ``SPEI_12_t``/``SOIL_MOISTURE_t`` respectively, over the same
        historical window. ``None`` only when the relevant column is
        entirely absent from the input frame (e.g. a fixture without
        environmental covariates) -- shrinkage cannot be applied to a
        response that was never computable at all, at the global level
        either.
    n_observations:
        The location's own observed-row count in the strictly-before
        history — the raw evidence count, unshrunk (this is what feeds
        ``shrinkage_weight``, not something to be shrunk itself).
    shrinkage_weight:
        ``w = n_observations / (n_observations + shrinkage_k)`` — how much
        of every shrunk field above is this location's own signal (``w``)
        vs. the global prior (``1 - w``).
    """

    location_id: str
    as_of: pd.Timestamp
    mean: float
    std: float
    trend: float
    seasonality_amplitude: float
    acf_1: float | None
    acf_3: float | None
    acf_6: float | None
    acf_12: float | None
    spei_response: float | None
    soil_moisture_response: float | None
    n_observations: int
    shrinkage_weight: float


def _resolve_shrinkage_k(shrinkage_k: int | None) -> int:
    if shrinkage_k is not None:
        return shrinkage_k
    return load_feature_config("signatures").shrinkage_k


def _ols_slope_unbounded(observed: pd.DataFrame) -> float:
    """Long-run OLS slope over *all* rows of ``observed`` (no trailing
    window) -- distinct from ``state.reconstruction``'s windowed
    ``local_trend`` helper, which this deliberately does not reuse."""
    if len(observed) < 2:
        return 0.0
    times = pd.to_datetime(observed["time"])
    t0 = times.min()
    x = np.array([month_diff(t, t0) for t in times], dtype=float)
    y = observed["TWS_t"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        # Every observed row falls in the same calendar month (e.g. only
        # one month of pooled global history exists so far) -- the design
        # matrix is degenerate (zero x-variance), so the slope is
        # undefined; 0.0 is the same "insufficient evidence" fallback used
        # elsewhere in this module (n<2 also returns 0.0).
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _seasonality_amplitude(observed: pd.DataFrame) -> float:
    if observed.empty:
        return 0.0
    calendar_months = pd.to_datetime(observed["time"]).dt.month
    monthly_means = observed.groupby(calendar_months)["TWS_t"].mean()
    if len(monthly_means) < 2:
        return 0.0
    return float(monthly_means.max() - monthly_means.min())


def _covariate_response(observed: pd.DataFrame, column: str) -> float | None:
    if column not in observed.columns:
        return None
    paired = observed.dropna(subset=[column, "TWS_t"])
    if len(paired) < 2:
        return None
    x = paired[column].to_numpy(dtype=float)
    y = paired["TWS_t"].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _location_lag_pairs(
    observed_location: pd.DataFrame, lag: int
) -> tuple[list[float], list[float]]:
    """The same period-dict lag-pairing logic ``compute_acf_at_lags`` uses
    internally, exposed here so pooled (cross-location) correlations can be
    built from the same pairs a single location's own ACF would use --
    the "promote, don't re-copy" pattern applied to the *pooling* step,
    not just the per-location computation itself."""
    if observed_location.empty:
        return [], []
    periods = pd.PeriodIndex(pd.to_datetime(observed_location["time"]), freq="M")
    value_by_period = dict(zip(periods, observed_location["TWS_t"].astype(float), strict=True))
    xs, ys = [], []
    for period, value in value_by_period.items():
        lagged = period - lag
        if lagged in value_by_period:
            xs.append(value_by_period[lagged])
            ys.append(value)
    return xs, ys


def _pooled_acf(history_all: pd.DataFrame, lags: tuple[int, ...]) -> dict[int, float | None]:
    """The global (all-location) fallback for ``acf_1_3_6_12``: pool every
    location's own ``(value_at_t_minus_lag, value_at_t)`` pairs together
    and compute one correlation across the pooled set, rather than
    averaging per-location correlations (which would need every location's
    own ACF computed first, at real dataset scale) -- this is the
    ``theta_global`` this project's shrinkage formula shrinks toward."""
    observed_all = history_all.dropna(subset=["TWS_t"])
    result: dict[int, float | None] = {}
    for lag in lags:
        xs_all: list[float] = []
        ys_all: list[float] = []
        for _, loc_group in observed_all.groupby("location_id", sort=False):
            xs, ys = _location_lag_pairs(loc_group, lag)
            xs_all.extend(xs)
            ys_all.extend(ys)
        if len(xs_all) >= 2 and np.std(xs_all) > 0 and np.std(ys_all) > 0:
            result[lag] = float(np.corrcoef(xs_all, ys_all)[0, 1])
        else:
            result[lag] = None
    return result


def _pooled_covariate_response(history_all: pd.DataFrame, column: str) -> float | None:
    """The global fallback for ``spei_response``/``soil_moisture_response``:
    pool ``(covariate, TWS_t)`` pairs across every location, computed once
    (not per-location)."""
    if column not in history_all.columns:
        return None
    paired = history_all.dropna(subset=[column, "TWS_t"])
    if len(paired) < 2:
        return None
    x = paired[column].to_numpy(dtype=float)
    y = paired["TWS_t"].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _shrink(raw_location: float | None, raw_global: float | None, w: float) -> float | None:
    if raw_location is None and raw_global is None:
        return None
    loc_val = 0.0 if raw_location is None else raw_location
    global_val = 0.0 if raw_global is None else raw_global
    return w * loc_val + (1 - w) * global_val


def compute_location_signature(
    df: pd.DataFrame,
    location_id: str,
    as_of: pd.Timestamp | str,
    shrinkage_k: int | None = None,
) -> LocationSignature:
    """Build one shrinkage-regularized :class:`LocationSignature`.

    Uses only rows with ``time < as_of`` (strictly before -- see the module
    docstring for why this differs from ``build_state_snapshot``'s ``<=``).
    ``df`` must contain ``time``, ``TWS_t``, and either ``location_id`` or
    both ``lat``/``lon``; ``SPEI_12_t``/``SOIL_MOISTURE_t`` are used if
    present, otherwise those two response fields fall back to ``None``.

    Parameters
    ----------
    shrinkage_k:
        ``w = n / (n + shrinkage_k)``. Defaults to
        ``configs/features/signatures.yaml``'s ``shrinkage_k`` when omitted
        (config-driven, per ``docs/PHASE4_EXECUTION_PLAN.md`` §1).
    """
    resolved_k = _resolve_shrinkage_k(shrinkage_k)
    as_of_ts = pd.Timestamp(as_of)
    frame = ensure_location_id(df)
    times = pd.to_datetime(frame["time"])
    history_all = frame.loc[times < as_of_ts]

    loc_history = history_all.loc[history_all["location_id"] == location_id].sort_values("time")
    loc_observed = loc_history.dropna(subset=["TWS_t"])
    n_observations = len(loc_observed)

    loc_mean = float(loc_observed["TWS_t"].mean()) if n_observations >= 1 else 0.0
    loc_std = float(loc_observed["TWS_t"].std(ddof=0)) if n_observations >= 2 else 0.0
    loc_trend = _ols_slope_unbounded(loc_observed)
    loc_seasonality = _seasonality_amplitude(loc_observed)
    loc_acf = compute_acf_at_lags(loc_observed, lags=ACF_LAGS)
    loc_spei = _covariate_response(loc_observed, SPEI_COVARIATE_COLUMN)
    loc_soil = _covariate_response(loc_observed, SOIL_MOISTURE_COVARIATE_COLUMN)

    global_observed = history_all.dropna(subset=["TWS_t"])
    global_mean = float(global_observed["TWS_t"].mean()) if len(global_observed) >= 1 else 0.0
    global_std = float(global_observed["TWS_t"].std(ddof=0)) if len(global_observed) >= 2 else 0.0
    global_trend = _ols_slope_unbounded(global_observed)
    global_seasonality = _seasonality_amplitude(global_observed)
    global_acf = _pooled_acf(history_all, lags=ACF_LAGS)
    global_spei = _pooled_covariate_response(history_all, SPEI_COVARIATE_COLUMN)
    global_soil = _pooled_covariate_response(history_all, SOIL_MOISTURE_COVARIATE_COLUMN)

    w = n_observations / (n_observations + resolved_k)

    return LocationSignature(
        location_id=location_id,
        as_of=as_of_ts,
        mean=_shrink(loc_mean, global_mean, w),
        std=_shrink(loc_std, global_std, w),
        trend=_shrink(loc_trend, global_trend, w),
        seasonality_amplitude=_shrink(loc_seasonality, global_seasonality, w),
        acf_1=_shrink(loc_acf[1], global_acf[1], w),
        acf_3=_shrink(loc_acf[3], global_acf[3], w),
        acf_6=_shrink(loc_acf[6], global_acf[6], w),
        acf_12=_shrink(loc_acf[12], global_acf[12], w),
        spei_response=_shrink(loc_spei, global_spei, w),
        soil_moisture_response=_shrink(loc_soil, global_soil, w),
        n_observations=n_observations,
        shrinkage_weight=w,
    )


def _expanding_slope_raw(values: np.ndarray) -> float:
    y = values
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _location_expanding_raw_stats(group: pd.DataFrame, location_id: str) -> pd.DataFrame:
    """Per-location, strictly-before-each-period raw (unshrunk) signature
    components -- the location-level half of :func:`compute_location_signatures`.

    Every quantity is computed as pandas' ordinary *inclusive* expanding
    statistic (``.expanding().mean()``, etc. -- up to and including each
    position), then shifted forward by one period. This is the general
    trick that turns any inclusive expanding computation into the
    strictly-before-this-period one ``docs/ARCHITECTURE.md`` §4's signature
    invariant requires, without a second, separately-derived formula for
    each statistic.
    """
    group = group.sort_values("time")
    times = pd.to_datetime(group["time"])
    full_range = pd.period_range(start=times.min(), end=times.max(), freq="M")

    raw = pd.Series(group["TWS_t"].astype(float).values, index=pd.PeriodIndex(times, freq="M"))
    raw = raw.groupby(level=0).first()
    s = raw.reindex(full_range)
    notna = s.notna()

    n_obs_excl = notna.astype(int).cumsum().shift(1).fillna(0).astype(int)
    mean_excl = s.expanding(min_periods=1).mean().shift(1)
    std_excl = s.expanding(min_periods=1).std(ddof=0).shift(1)
    trend_excl = s.expanding(min_periods=2).apply(_expanding_slope_raw, raw=True).shift(1)

    months = full_range.month
    monthly_running = {}
    for m in range(1, 13):
        shadow = s.where(months == m)
        monthly_running[m] = shadow.expanding(min_periods=1).mean().ffill().shift(1)
    monthly_frame = pd.DataFrame(monthly_running)
    seasonality_excl = monthly_frame.max(axis=1) - monthly_frame.min(axis=1)
    seasonality_populated = monthly_frame.notna().sum(axis=1)
    seasonality_excl = seasonality_excl.where(seasonality_populated.values >= 2, other=0.0)

    out = pd.DataFrame(
        {
            "location_id": location_id,
            "period": full_range,
            "n_observations": n_obs_excl.values,
            "raw_mean": mean_excl.fillna(0.0).values,
            "raw_std": std_excl.fillna(0.0).values,
            "raw_trend": trend_excl.fillna(0.0).values,
            "raw_seasonality_amplitude": seasonality_excl.fillna(0.0).values,
            "value": s.values,
        }
    )
    for lag in ACF_LAGS:
        corr_excl = s.expanding(min_periods=2).corr(s.shift(lag)).shift(1)
        out[f"lag{lag}_x"] = s.shift(lag).values
        out[f"raw_acf_{lag}"] = corr_excl.values

    for column, key in (
        (SPEI_COVARIATE_COLUMN, "spei"),
        (SOIL_MOISTURE_COVARIATE_COLUMN, "soil_moisture"),
    ):
        if column in group.columns:
            cov_raw = pd.Series(
                group[column].astype(float).values, index=pd.PeriodIndex(times, freq="M")
            )
            cov_raw = cov_raw.groupby(level=0).first().reindex(full_range)
            out[f"{key}_covariate"] = cov_raw.values
            out[f"raw_{key}_response"] = s.expanding(min_periods=2).corr(cov_raw).shift(1).values
        else:
            out[f"{key}_covariate"] = np.nan
            out[f"raw_{key}_response"] = np.nan

    return out


def _period_pooled_correlation(
    panel: pd.DataFrame, x_col: str, y_col: str, period_order: pd.PeriodIndex
) -> pd.Series:
    """Strictly-before-each-period pooled Pearson correlation of
    ``(x_col, y_col)`` across every location present in ``panel``, indexed
    by ``period_order``. The global (all-location) fallback for a pooled
    per-location quantity (ACF at a lag, a covariate response) --
    aggregated once per period (bounded by the number of *periods*, not
    rows), never recomputed per row.
    """
    valid = panel.dropna(subset=[x_col, y_col])
    if valid.empty:
        return pd.Series(np.nan, index=period_order)

    grouped = valid.groupby("period")[[x_col, y_col]].apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "sx": g[x_col].sum(),
                "sy": g[y_col].sum(),
                "sxx": (g[x_col] ** 2).sum(),
                "syy": (g[y_col] ** 2).sum(),
                "sxy": (g[x_col] * g[y_col]).sum(),
            }
        )
    )
    ledger = grouped.reindex(period_order).fillna(0.0)

    cum_n = ledger["n"].cumsum().shift(1).fillna(0.0)
    cum_sx = ledger["sx"].cumsum().shift(1).fillna(0.0)
    cum_sy = ledger["sy"].cumsum().shift(1).fillna(0.0)
    cum_sxx = ledger["sxx"].cumsum().shift(1).fillna(0.0)
    cum_syy = ledger["syy"].cumsum().shift(1).fillna(0.0)
    cum_sxy = ledger["sxy"].cumsum().shift(1).fillna(0.0)

    numerator = cum_n * cum_sxy - cum_sx * cum_sy
    denom_x = cum_n * cum_sxx - cum_sx**2
    denom_y = cum_n * cum_syy - cum_sy**2
    denom = np.sqrt(denom_x * denom_y)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = numerator / denom
    corr = corr.where((cum_n >= 2) & (denom > 0))
    return corr


def compute_location_signatures(
    df: pd.DataFrame,
    as_of_column: str = "time",
    shrinkage_k: int | None = None,
) -> pd.DataFrame:
    """Vectorized batch equivalent of calling
    :func:`compute_location_signature` for every row of ``df`` (at that
    row's own ``as_of_column`` value) — the function Phase 4's
    feature-assembly pipeline (step 4.9) actually calls.

    Computed in two passes, both bounded by the number of *locations* or
    *periods*, never by the number of rows: a per-location expanding pass
    (raw, unshrunk location-level statistics) and a per-period pooled pass
    (raw, unshrunk global statistics, aggregated once per calendar month
    across every location). The two are merged by period and shrunk
    elementwise — the same ``w = n / (n + shrinkage_k)`` formula
    :func:`compute_location_signature` uses.

    Returns a ``pd.DataFrame`` indexed identically to ``df``, with one
    column per scalar :class:`LocationSignature` field (``mean``, ``std``,
    ``trend``, ``seasonality_amplitude``, ``acf_1``/``acf_3``/``acf_6``/
    ``acf_12``, ``spei_response``, ``soil_moisture_response``,
    ``n_observations``, ``shrinkage_weight``).
    """
    resolved_k = _resolve_shrinkage_k(shrinkage_k)
    frame = ensure_location_id(df).copy()
    frame[as_of_column] = pd.to_datetime(frame[as_of_column])

    location_groups = list(frame.groupby("location_id", sort=False))
    if not location_groups:
        columns = [
            "location_id",
            "as_of",
            "mean",
            "std",
            "trend",
            "seasonality_amplitude",
            "acf_1",
            "acf_3",
            "acf_6",
            "acf_12",
            "spei_response",
            "soil_moisture_response",
            "n_observations",
            "shrinkage_weight",
        ]
        return pd.DataFrame(columns=columns)

    per_location = [
        _location_expanding_raw_stats(group, location_id) for location_id, group in location_groups
    ]
    panel = pd.concat(per_location, ignore_index=True)

    period_order = pd.period_range(
        start=pd.PeriodIndex(panel["period"]).min(),
        end=pd.PeriodIndex(panel["period"]).max(),
        freq="M",
    )

    # Global (pooled-across-locations) raw stats, strictly before each
    # period -- aggregated once per period, not once per row.
    observed_panel = panel.dropna(subset=["value"])
    period_ledger = (
        observed_panel.groupby("period")["value"]
        .agg(n="count", sum="sum", sumsq=lambda s: (s**2).sum())
        .reindex(period_order)
        .fillna(0.0)
    )
    cum_n = period_ledger["n"].cumsum().shift(1).fillna(0.0)
    cum_sum = period_ledger["sum"].cumsum().shift(1).fillna(0.0)
    cum_sumsq = period_ledger["sumsq"].cumsum().shift(1).fillna(0.0)
    global_mean_by_period = (cum_sum / cum_n.replace(0, np.nan)).fillna(0.0)
    global_var_by_period = (cum_sumsq / cum_n.replace(0, np.nan) - global_mean_by_period**2).clip(
        lower=0
    )
    global_std_by_period = np.sqrt(global_var_by_period).fillna(0.0)

    # Global trend: OLS slope of (period-ordinal, value) pooled across all
    # locations, via the same expanding-cumulative-sums technique, using
    # each period's integer position in period_order as its shared x.
    period_to_x = {p: i for i, p in enumerate(period_order)}
    observed_panel = observed_panel.assign(x=observed_panel["period"].map(period_to_x))
    trend_ledger = (
        observed_panel.groupby("period")
        .apply(
            lambda g: pd.Series(
                {
                    "sx": g["x"].sum(),
                    "sxx": (g["x"] ** 2).sum(),
                    "sxy": (g["x"] * g["value"]).sum(),
                }
            )
        )
        .reindex(period_order)
        .fillna(0.0)
    )
    cum_sx = trend_ledger["sx"].cumsum().shift(1).fillna(0.0)
    cum_sxx = trend_ledger["sxx"].cumsum().shift(1).fillna(0.0)
    cum_sxy = trend_ledger["sxy"].cumsum().shift(1).fillna(0.0)
    trend_num = cum_n * cum_sxy - cum_sx * cum_sum
    trend_den = cum_n * cum_sxx - cum_sx**2
    with np.errstate(invalid="ignore", divide="ignore"):
        global_trend_by_period = (
            (trend_num / trend_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    # Global seasonality amplitude: 12 running per-calendar-month pooled
    # means, strictly before each period.
    observed_panel = observed_panel.assign(month=pd.PeriodIndex(observed_panel["period"]).month)
    monthly_running_global = {}
    for m in range(1, 13):
        subset = observed_panel.loc[observed_panel["month"] == m]
        month_ledger = (
            subset.groupby("period")["value"]
            .agg(n="count", sum="sum")
            .reindex(period_order)
            .fillna(0.0)
        )
        cum_n_m = month_ledger["n"].cumsum().shift(1).fillna(0.0)
        cum_sum_m = month_ledger["sum"].cumsum().shift(1).fillna(0.0)
        monthly_running_global[m] = (cum_sum_m / cum_n_m.replace(0, np.nan)).ffill()
    monthly_global_frame = pd.DataFrame(monthly_running_global)
    global_seasonality_by_period = (
        monthly_global_frame.max(axis=1) - monthly_global_frame.min(axis=1)
    ).fillna(0.0)

    global_acf_by_period = {
        lag: _period_pooled_correlation(panel, f"lag{lag}_x", "value", period_order)
        for lag in ACF_LAGS
    }
    global_spei_by_period = _period_pooled_correlation(
        panel, "spei_covariate", "value", period_order
    )
    global_soil_by_period = _period_pooled_correlation(
        panel, "soil_moisture_covariate", "value", period_order
    )

    # --- Merge location-level raw stats with the period-indexed globals,
    # then shrink elementwise. ---
    panel = panel.set_index("period")
    w = panel["n_observations"] / (panel["n_observations"] + resolved_k)

    def _shrink_series(loc_col: str, global_by_period: pd.Series) -> pd.Series:
        global_aligned = global_by_period.reindex(panel.index).to_numpy()
        loc_vals = panel[loc_col].to_numpy()
        loc_has_value = (
            ~pd.isna(loc_vals) if pd.isna(loc_vals).any() else np.ones(len(loc_vals), dtype=bool)
        )
        loc_filled = np.nan_to_num(loc_vals, nan=0.0)
        global_filled = np.nan_to_num(global_aligned, nan=0.0)
        shrunk = w.to_numpy() * loc_filled + (1 - w.to_numpy()) * global_filled
        both_missing = (~loc_has_value) & pd.isna(global_aligned)
        return pd.Series(np.where(both_missing, np.nan, shrunk), index=panel.index)

    result = pd.DataFrame(
        {
            "location_id": panel["location_id"].values,
            "as_of": panel.index.to_timestamp().values,
            "mean": _shrink_series("raw_mean", global_mean_by_period).values,
            "std": _shrink_series("raw_std", global_std_by_period).values,
            "trend": _shrink_series("raw_trend", global_trend_by_period).values,
            "seasonality_amplitude": _shrink_series(
                "raw_seasonality_amplitude", global_seasonality_by_period
            ).values,
            "n_observations": panel["n_observations"].values,
            "shrinkage_weight": w.values,
        }
    )
    for lag in ACF_LAGS:
        result[f"acf_{lag}"] = _shrink_series(f"raw_acf_{lag}", global_acf_by_period[lag]).values
    result["spei_response"] = _shrink_series("raw_spei_response", global_spei_by_period).values
    result["soil_moisture_response"] = _shrink_series(
        "raw_soil_moisture_response", global_soil_by_period
    ).values

    result.index = panel.index  # temporary, for the row-selection join below
    result = result.reset_index(drop=True)
    panel_periods = panel.index

    # Select exactly the rows corresponding to each original df row's own
    # (location_id, as_of_column) — panel/result currently hold one row per
    # (location, period) in each location's full continuous range, which can
    # be a superset of df's own rows if df has gaps.
    lookup = result.copy()
    lookup["period"] = panel_periods.values
    lookup_key = list(zip(lookup["location_id"], lookup["period"], strict=True))
    lookup_map = {key: idx for idx, key in enumerate(lookup_key)}

    frame_periods = pd.PeriodIndex(frame[as_of_column], freq="M")
    row_keys = list(zip(frame["location_id"], frame_periods, strict=True))
    row_positions = [lookup_map[key] for key in row_keys]

    return lookup.iloc[row_positions].drop(columns=["period"]).set_axis(frame.index)


class LocationSignatureTransformer:
    """``features.base.Transformer`` wrapper around
    :func:`compute_location_signatures`.

    ``fit(train_df)`` stores the training frame as this transformer's
    historical source. ``transform(df)`` computes signatures for every row
    of ``df`` using the *union* of the stored training frame and ``df``
    itself as history — never ``df`` alone — so that a signature at any
    row of ``df`` is still built only from rows strictly before that row's
    own time, drawn from whatever genuinely would have been known by then
    (the training fold, plus any earlier-in-time rows of ``df`` that have
    themselves already "elapsed" relative to a later row in the same
    frame). This mirrors how a real deployed system accumulates history
    over time, and is the same reasoning
    ``docs/PHASE4_EXECUTION_PLAN.md`` §1's leakage-safe-transformer
    invariant requires of every Phase 4 feature.

    Returns only the new signature columns (see
    :func:`compute_location_signatures`), indexed like its input --
    callers join with ``pd.concat([df, transformer.transform(df)], axis=1)``.
    """

    def __init__(self, shrinkage_k: int | None = None) -> None:
        self._shrinkage_k = shrinkage_k
        self._train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_df is None:
            raise RuntimeError("LocationSignatureTransformer.transform called before fit()")

        combined = pd.concat([self._train_df, df], ignore_index=False)
        combined = ensure_location_id(combined)
        # Deduplicate by (location_id, time) content, never by raw pandas
        # index -- train_df and df may have colliding positional indices
        # despite representing entirely different rows (e.g. each was
        # built independently), and index-based deduplication would
        # silently drop real rows in that case. keep="last" prefers df's
        # own row over train_df's when both genuinely describe the same
        # (location, time).
        combined = combined.loc[~combined.duplicated(subset=["location_id", "time"], keep="last")]
        signatures = compute_location_signatures(combined, shrinkage_k=self._shrinkage_k)
        return signatures.loc[df.index]
