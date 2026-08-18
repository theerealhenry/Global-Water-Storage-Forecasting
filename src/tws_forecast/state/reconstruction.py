"""State-reconstruction schemas.

This module holds the two canonical schemas ``docs/ARCHITECTURE.md`` §4
assigns to it: ``ForecastOrigin`` (Project Phase 2) and ``StateSnapshot``
(Project Phase 4, step 4.1).

``StateSnapshot`` is the single, canonical definition of what is known about
a location's water state at a given forecast origin — no other module is
permitted to compute its own competing notion of "months since observation"
or "last known value" (``docs/ARCHITECTURE.md`` §4). Its field list extends
``ARCHITECTURE.md`` §4's original twelve fields with two more
(``second_previous_known_tws``, ``state_acceleration``) per
``docs/adr/0006-statesnapshot-trajectory-fields.md`` — a minimal, additive
reconciliation between that document and ``docs/PROJECT_PLAN.md``'s Phase 4
bullet asking for observation-trajectory *acceleration*, which needs a third
trajectory point to be defined at all.

``location_signature`` is deliberately typed ``object | None`` and always
``None`` from this module alone: the shrinkage-regularized
``state.signatures.LocationSignature`` it will hold is Phase 4 step 4.2's
job, built on top of this module rather than inside it (``docs/
PHASE4_EXECUTION_PLAN.md`` §4.1/§4.2). Every downstream consumer must treat
a ``None`` signature as "not yet computed," not "no evidence exists."

Two build entry points, mirroring ``ForecastOrigin.from_row`` vs.
``validation.splitters.attach_forecast_origin_columns``:

- :func:`build_state_snapshot` — one location, one ``as_of`` timestamp.
  Used for single-row diagnostics and as the correctness reference the
  vectorized batch variant is tested against.
- :func:`build_state_snapshots` — every row of a frame at once, computed via
  per-location vectorized pandas operations (expanding/rolling/ffill), never
  a Python loop over millions of individual rows. This is the function the
  Phase 4 feature-assembly pipeline (step 4.9) actually calls.

Origin-time indexing (``docs/ARCHITECTURE.md`` §4's checked invariant): both
functions build a location's state strictly from rows with
``time <= as_of`` — which, since ``as_of`` is always this project's
``information_cutoff`` and every horizon here is 1 month
(``ForecastOrigin.information_cutoff == origin_time``), is exactly
"everything strictly before the *next* month being forecast." A row exactly
at ``as_of`` is included (not excluded) — if that row's own ``TWS_t`` is
observed, the snapshot's ``state_status`` is ``OBSERVED`` and its trajectory
fields trivially equal that row's own value, by construction, not as a
special case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "ForecastOrigin",
    "location_id_from_lat_lon",
    "ensure_location_id",
    "compute_acf_at_lags",
    "month_diff",
    "rolling_slope_raw",
    "StateSnapshot",
    "StateStatus",
    "DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS",
    "DEFAULT_MIN_EVIDENCE_OBSERVATIONS",
    "ACF_LAGS",
    "build_state_snapshot",
    "build_state_snapshots",
]

Regime = Literal["observed", "masked"]


def location_id_from_lat_lon(lat: float, lon: float) -> str:
    """The single, canonical way to turn a grid cell's coordinates into a
    stable ``location_id`` string — ``"{lat}_{lon}"``.

    Exported (not a private helper) specifically so bulk/vectorized code
    (e.g. ``validation/splitters.py``, which computes this column for
    millions of rows at once rather than building one ``ForecastOrigin``
    per row) can produce values guaranteed identical to
    ``ForecastOrigin.from_row``'s, rather than each maintaining its own
    copy of the same formatting rule.
    """
    return f"{float(lat)}_{float(lon)}"


@dataclass(frozen=True)
class ForecastOrigin:
    """What a single forecast is made from and targets, precisely.

    Every fold, masked example, and decomposition-table row from Project
    Phase 2 onward is keyed by one of these — never by raw row index — so
    that "did this computation use anything at or after the forecast
    origin" is a mechanically checkable question (``docs/ARCHITECTURE.md``
    §4, the leakage-firewall discipline built out in step 2.8).

    Attributes
    ----------
    origin_time:
        The month the forecast is made *from* — i.e. the row's own ``time``.
    target_time:
        The month being forecast — ``origin_time`` plus ``horizon`` months.
        For this competition, always exactly one calendar month ahead
        (verified target definition, ``docs/DATA_DICTIONARY.md``).
    horizon:
        Number of months from origin to target. Always ``1`` for this
        competition; kept as an explicit field rather than a hardcoded
        assumption so the schema doesn't silently break if a multi-step
        experiment is ever tried.
    information_cutoff:
        The latest time whose data may be used to produce this forecast.
        Always equal to ``origin_time`` here (the row's own current
        observation is, at most, as recent as the row itself) — kept as a
        distinct field, rather than reused as `origin_time` everywhere,
        because a future feature (e.g. one deliberately using only data
        through month t-1) could set it earlier without needing a new
        schema.
    location_id:
        A stable identifier for the fixed grid cell, ``"{lat}_{lon}"`` —
        deliberately not the row's ``sample_id``/``ID`` (which encodes the
        date too) so the same location can be joined across origins.
    regime:
        ``"observed"`` if this row's own ``TWS_t`` is a real observation at
        the forecast origin, ``"masked"`` if it's withheld. Train.csv rows
        are always ``"observed"`` (masking is test-set-only per
        ``docs/DATA_DICTIONARY.md``); Test.csv rows follow
        ``TWS_t_masked`` exactly.
    """

    origin_time: pd.Timestamp
    target_time: pd.Timestamp
    horizon: int
    information_cutoff: pd.Timestamp
    location_id: str
    regime: Regime

    def __post_init__(self) -> None:
        origin = pd.Timestamp(self.origin_time)
        target = pd.Timestamp(self.target_time)
        cutoff = pd.Timestamp(self.information_cutoff)

        expected_target = origin + pd.DateOffset(months=self.horizon)
        if target != expected_target:
            raise ValueError(
                f"target_time ({target}) does not equal origin_time + "
                f"{self.horizon} month(s) ({expected_target}) — a "
                "ForecastOrigin's target must be derived, not asserted."
            )
        if cutoff > origin:
            raise ValueError(
                f"information_cutoff ({cutoff}) is after origin_time "
                f"({origin}) — this would mean the forecast could use "
                "information from after its own origin, which is exactly "
                "the leakage this schema exists to make impossible."
            )
        if self.regime not in ("observed", "masked"):
            raise ValueError(f"regime must be 'observed' or 'masked', got {self.regime!r}")

    @classmethod
    def from_row(cls, row: pd.Series | Mapping[str, object], horizon: int = 1) -> ForecastOrigin:
        """Build a ``ForecastOrigin`` from one ``Train.csv``/``Test.csv`` row.

        Parameters
        ----------
        row:
            A row with at least ``time``, ``lat``, ``lon``. Regime is
            inferred from ``TWS_t_masked`` if present (Test.csv), else from
            whether ``TWS_t`` itself is null, else defaults to
            ``"observed"``.
        horizon:
            Months from origin to target. Defaults to 1, the only horizon
            this competition asks for.
        """
        origin_time = pd.Timestamp(row["time"])
        target_time = origin_time + pd.DateOffset(months=horizon)
        location_id = location_id_from_lat_lon(row["lat"], row["lon"])
        regime: Regime = "masked" if _is_masked(row) else "observed"

        return cls(
            origin_time=origin_time,
            target_time=target_time,
            horizon=horizon,
            information_cutoff=origin_time,
            location_id=location_id,
            regime=regime,
        )


def _is_masked(row: pd.Series | Mapping[str, object]) -> bool:
    """Infer whether a row's current-month TWS is masked, robustly across
    ``Train.csv`` (no masking column, ``TWS_t`` always populated) and
    ``Test.csv`` (``TWS_t_masked`` is the authoritative indicator, verified
    with zero mismatches against ``TWS_t`` nullness in
    ``docs/DATA_DICTIONARY.md``)."""
    if "TWS_t_masked" in row:
        return bool(row["TWS_t_masked"])
    if "TWS_t" in row:
        return bool(pd.isna(row["TWS_t"]))
    return False


# ---------------------------------------------------------------------------
# StateSnapshot (Project Phase 4, step 4.1)
# ---------------------------------------------------------------------------

StateStatus = Literal["OBSERVED", "RECONSTRUCTED", "PARTIALLY_RECONSTRUCTED"]

#: The four ACF lags every ``StateSnapshot`` reports, matching
#: ``ARCHITECTURE.md`` §4's ``acf_1_3_6_12`` field and
#: ``phase1_constants.ACF_QUARTILE_AR1_PARAMS``'s own lag structure.
ACF_LAGS: tuple[int, ...] = (1, 3, 6, 12)

#: A ``RECONSTRUCTED`` (vs. ``PARTIALLY_RECONSTRUCTED``) status requires the
#: most recent observation to be no more than this many months stale.
#: Module-level constant rather than a ``configs/features/*.yaml`` entry
#: because this threshold governs ``state/reconstruction.py`` itself, one
#: layer below the ``features/`` package step 4.4 introduces config files
#: for — revisit as a config value if step 4.2's shrinkage weight ends up
#: needing to feed back into this threshold directly.
DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS: int = 12

#: A ``RECONSTRUCTED`` status also requires at least this many historical
#: observations for the location (a simple evidence-count proxy for step
#: 4.2's real shrinkage weight ``w = n / (n + k)``, which does not exist
#: yet in this module — see the class docstring's note on
#: ``location_signature``).
DEFAULT_MIN_EVIDENCE_OBSERVATIONS: int = 24


@dataclass(frozen=True)
class StateSnapshot:
    """What is known about one location's water state at one forecast
    origin — the canonical schema ``docs/ARCHITECTURE.md`` §4 specifies,
    extended per ``docs/adr/0006-statesnapshot-trajectory-fields.md``.

    Every field below is computed strictly from history at-or-before
    ``as_of`` for ``location_id`` (origin-time indexing, not fold-level
    out-of-fold) — see this module's docstring for the exact boundary rule.

    Attributes
    ----------
    location_id, as_of:
        The key this snapshot answers "what do we know" for.
    last_known_tws, last_known_time:
        The most recently *observed* TWS value at or before ``as_of``, and
        the month it was observed. ``None``/``None`` if the location has
        never been observed as of ``as_of``. Note this is the
        *last-observed-lag* quantity, distinct from the *calendar-lag*
        quantity (``TWS`` at exactly ``as_of`` minus ``k`` months, itself
        often missing) — ``docs/PROJECT_PLAN.md``'s "four distinct temporal
        quantities."
    months_since_observation:
        ``as_of`` minus ``last_known_time``, in whole calendar months. ``0``
        when the current month is itself observed. ``None`` if never
        observed.
    previous_known_tws, second_previous_known_tws:
        The second- and third-most-recent *observed* values before (or at)
        ``as_of`` — the observation trajectory. ``None`` when fewer than
        two/three observations exist in history respectively.
    historical_delta:
        State *velocity*: ``last_known_tws - previous_known_tws``. ``None``
        when ``previous_known_tws`` is ``None``.
    state_acceleration:
        State *acceleration*, the second difference of the trajectory:
        ``historical_delta - (previous_known_tws - second_previous_known_tws)``.
        ``None`` when ``second_previous_known_tws`` is ``None`` — per
        ADR-0006.
    local_trend:
        A rolling OLS slope of observed TWS values over the longest
        configured trailing window (see ``build_state_snapshot``'s
        ``trailing_windows``), in TWS units per month. ``None`` with fewer
        than two observed points in that window.
    seasonal_position:
        ``(as_of.month - 1) / 12``, a ``[0, 1)`` cyclical position within
        the calendar year — the raw ingredient a later feature (step 4.5)
        can turn into a sin/cos or hemisphere-interaction encoding; kept
        as a plain scalar here rather than pre-encoded, since encoding
        choice is a feature-engineering decision, not a state-reconstruction
        one.
    acf_1_3_6_12:
        A ``{lag: value | None}`` mapping for ``lag in ACF_LAGS`` — the
        origin-time-indexed autocorrelation of the location's observed TWS
        series at each lag, computed from history strictly at-or-before
        ``as_of`` only (an *expanding* computation, never the full-record
        ACF Project Phase 1 measured once and reused everywhere — that
        would leak future structure into early-history snapshots). ``None``
        for a lag with fewer than two valid ``(t, t-lag)`` observed pairs.
    observation_density:
        A ``{window_months: fraction}`` mapping — the fraction of calendar
        months in each requested trailing window (ending at ``as_of``,
        inclusive) that have an actual observation.
    blackout_streak_length:
        Consecutive calendar months, ending at ``as_of`` inclusive, with no
        observation. ``0`` if ``as_of`` itself is observed.
    location_signature:
        Always ``None`` from this module. Populated by
        ``state.signatures`` (Phase 4 step 4.2), which computes it
        separately and attaches it — this module does not implement
        shrinkage-regularized location statistics itself.
    state_status:
        ``"OBSERVED"`` if ``as_of`` itself has a real observation.
        ``"RECONSTRUCTED"`` if masked, but the last observation is within
        ``max_reconstruction_gap_months`` and the location has at least
        ``min_evidence_observations`` historical observations.
        ``"PARTIALLY_RECONSTRUCTED"`` otherwise (masked with no prior
        observation at all, too stale a last observation, or too little
        historical evidence) — the flag every downstream consumer
        (uncertainty architecture, MoE gating, the eventual deployment UI)
        most needs, per ``docs/ARCHITECTURE.md`` §4.
    """

    location_id: str
    as_of: pd.Timestamp
    last_known_tws: float | None
    last_known_time: pd.Timestamp | None
    months_since_observation: int | None
    previous_known_tws: float | None
    second_previous_known_tws: float | None
    historical_delta: float | None
    state_acceleration: float | None
    local_trend: float | None
    seasonal_position: float
    acf_1_3_6_12: dict[int, float | None]
    observation_density: dict[int, float]
    blackout_streak_length: int
    location_signature: Any | None
    state_status: StateStatus


def ensure_location_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``location_id`` if not already present, without mutating the
    caller's frame. Deliberately independent of
    ``validation.splitters.attach_forecast_origin_columns`` (which itself
    imports from *this* module) to avoid a circular import — this module
    sits below ``validation/`` in the dependency graph, per
    ``docs/ARCHITECTURE.md`` §6.
    """
    if "location_id" in df.columns:
        return df
    out = df.copy()
    out["location_id"] = [
        location_id_from_lat_lon(lat, lon) for lat, lon in zip(out["lat"], out["lon"], strict=True)
    ]
    return out


def month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    """Whole calendar months between two timestamps (``later - earlier``)."""
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _derive_state_status(
    *,
    is_observed_now: bool,
    months_since_observation: int | None,
    n_history_observations: int,
    max_reconstruction_gap_months: int,
    min_evidence_observations: int,
) -> StateStatus:
    if is_observed_now:
        return "OBSERVED"
    if months_since_observation is None:
        return "PARTIALLY_RECONSTRUCTED"
    if (
        months_since_observation <= max_reconstruction_gap_months
        and n_history_observations >= min_evidence_observations
    ):
        return "RECONSTRUCTED"
    return "PARTIALLY_RECONSTRUCTED"


def _compute_local_trend(
    observed: pd.DataFrame, as_of: pd.Timestamp, window_months: int
) -> float | None:
    if observed.empty:
        return None
    window_start = as_of - pd.DateOffset(months=window_months)
    times = pd.to_datetime(observed["time"])
    windowed = observed.loc[(times > window_start) & (times <= as_of)]
    if len(windowed) < 2:
        return None
    windowed_times = pd.to_datetime(windowed["time"])
    x = np.array([month_diff(t, window_start) for t in windowed_times], dtype=float)
    y = windowed["TWS_t"].to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def compute_acf_at_lags(observed: pd.DataFrame, lags: tuple[int, ...]) -> dict[int, float | None]:
    if observed.empty:
        return dict.fromkeys(lags)
    periods = pd.PeriodIndex(pd.to_datetime(observed["time"]), freq="M")
    value_by_period = dict(zip(periods, observed["TWS_t"].astype(float), strict=True))

    result: dict[int, float | None] = {}
    for lag in lags:
        xs, ys = [], []
        for period, value in value_by_period.items():
            lagged = period - lag
            if lagged in value_by_period:
                xs.append(value_by_period[lagged])
                ys.append(value)
        if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            result[lag] = float(np.corrcoef(xs, ys)[0, 1])
        else:
            result[lag] = None
    return result


def _compute_observation_density(
    history: pd.DataFrame, as_of: pd.Timestamp, window_months: int
) -> float:
    window_start = as_of - pd.DateOffset(months=window_months - 1)
    times = pd.to_datetime(history["time"])
    windowed = history.loc[(times >= window_start) & (times <= as_of)]
    observed_months = windowed.dropna(subset=["TWS_t"])["time"].nunique()
    return observed_months / float(window_months)


def build_state_snapshot(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    location_id: str,
    trailing_windows: tuple[int, ...] = (12, 24),
    max_reconstruction_gap_months: int = DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS,
    min_evidence_observations: int = DEFAULT_MIN_EVIDENCE_OBSERVATIONS,
) -> StateSnapshot:
    """Build one ``StateSnapshot`` for ``location_id`` as of ``as_of``.

    Uses only rows with ``time <= as_of`` for this location (see the module
    docstring for why this boundary, not ``<``, is correct here). ``df``
    must contain ``time``, ``TWS_t``, and either ``location_id`` or both
    ``lat``/``lon`` (from which ``location_id`` is derived).

    Parameters
    ----------
    trailing_windows:
        Trailing windows (in months) to report ``observation_density`` for.
        The longest one is also used as ``local_trend``'s rolling window.
    max_reconstruction_gap_months, min_evidence_observations:
        Thresholds governing ``state_status``'s ``RECONSTRUCTED`` vs.
        ``PARTIALLY_RECONSTRUCTED`` split — see :data:`StateSnapshot`'s
        docstring.
    """
    as_of_ts = pd.Timestamp(as_of)
    frame = ensure_location_id(df)
    times = pd.to_datetime(frame["time"])
    history = frame.loc[(frame["location_id"] == location_id) & (times <= as_of_ts)].sort_values(
        "time"
    )

    observed = history.dropna(subset=["TWS_t"])
    n_history_observations = len(observed)

    if observed.empty:
        last_known_tws = None
        last_known_time = None
        months_since_observation = None
        previous_known_tws = None
        second_previous_known_tws = None
        historical_delta = None
        state_acceleration = None
        blackout_streak_length = len(history)
    else:
        last_row = observed.iloc[-1]
        last_known_tws = float(last_row["TWS_t"])
        last_known_time = pd.Timestamp(last_row["time"])
        months_since_observation = month_diff(as_of_ts, last_known_time)

        previous_known_tws = float(observed.iloc[-2]["TWS_t"]) if len(observed) >= 2 else None
        second_previous_known_tws = (
            float(observed.iloc[-3]["TWS_t"]) if len(observed) >= 3 else None
        )

        historical_delta = (
            last_known_tws - previous_known_tws if previous_known_tws is not None else None
        )
        if previous_known_tws is not None and second_previous_known_tws is not None:
            prev_delta = previous_known_tws - second_previous_known_tws
            state_acceleration = historical_delta - prev_delta
        else:
            state_acceleration = None

        blackout_streak_length = int((pd.to_datetime(history["time"]) > last_known_time).sum())

    is_observed_now = bool(
        not history.empty
        and pd.Timestamp(history.iloc[-1]["time"]) == as_of_ts
        and pd.notna(history.iloc[-1]["TWS_t"])
    )

    local_trend = _compute_local_trend(observed, as_of_ts, window_months=max(trailing_windows))
    seasonal_position = ((as_of_ts.month - 1) % 12) / 12.0
    acf_1_3_6_12 = compute_acf_at_lags(observed, lags=ACF_LAGS)
    observation_density = {
        w: _compute_observation_density(history, as_of_ts, w) for w in trailing_windows
    }

    state_status = _derive_state_status(
        is_observed_now=is_observed_now,
        months_since_observation=months_since_observation,
        n_history_observations=n_history_observations,
        max_reconstruction_gap_months=max_reconstruction_gap_months,
        min_evidence_observations=min_evidence_observations,
    )

    return StateSnapshot(
        location_id=location_id,
        as_of=as_of_ts,
        last_known_tws=last_known_tws,
        last_known_time=last_known_time,
        months_since_observation=months_since_observation,
        previous_known_tws=previous_known_tws,
        second_previous_known_tws=second_previous_known_tws,
        historical_delta=historical_delta,
        state_acceleration=state_acceleration,
        local_trend=local_trend,
        seasonal_position=seasonal_position,
        acf_1_3_6_12=acf_1_3_6_12,
        observation_density=observation_density,
        blackout_streak_length=blackout_streak_length,
        location_signature=None,
        state_status=state_status,
    )


def rolling_slope_raw(values: np.ndarray) -> float:
    """OLS slope of the non-NaN points in ``values`` against their integer
    position — used as a ``Series.rolling(...).apply(raw=True)`` callback,
    which is why NaN-dropping happens inside this function rather than via
    ``min_periods`` (that governs window size, not observed-value count)."""
    y = values
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _build_snapshots_for_location(
    group: pd.DataFrame,
    location_id: str,
    as_of_column: str,
    trailing_windows: tuple[int, ...],
    max_reconstruction_gap_months: int,
    min_evidence_observations: int,
) -> pd.DataFrame:
    """The vectorized, per-location core of :func:`build_state_snapshots`.

    Reindexes this location's history onto a continuous monthly
    ``PeriodIndex`` (so gaps become explicit ``NaN`` entries, not absent
    rows), computes every ``StateSnapshot`` field once across the whole
    continuous range using ``ffill``/``rolling``/``expanding`` (never a
    per-row Python loop), then selects out exactly the rows corresponding
    to this group's own ``as_of_column`` timestamps.
    """
    group = group.sort_values("time")
    times = pd.to_datetime(group["time"])
    full_range = pd.period_range(start=times.min(), end=times.max(), freq="M")
    n = len(full_range)

    raw = pd.Series(group["TWS_t"].astype(float).values, index=pd.PeriodIndex(times, freq="M"))
    raw = raw.groupby(level=0).first()
    s = raw.reindex(full_range)
    notna = s.notna()
    positions = np.arange(n)

    last_known_tws = s.ffill().values

    obs_pos_only = np.where(notna.values, positions, np.nan)
    last_obs_pos = pd.Series(obs_pos_only).ffill().values
    months_since = positions - last_obs_pos  # NaN propagates while never observed

    last_known_time_series = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    valid_last_obs = ~np.isnan(last_obs_pos)
    if valid_last_obs.any():
        last_known_time_series.loc[valid_last_obs] = full_range.to_timestamp()[
            last_obs_pos[valid_last_obs].astype(int)
        ]
    last_known_time_arr = last_known_time_series.values

    obs_positions = positions[notna.values]
    obs_values = s.values[notna.values]
    obs_series = pd.Series(obs_values)
    prev_at_obs = obs_series.shift(1).values
    prev2_at_obs = obs_series.shift(2).values
    delta_at_obs = obs_values - prev_at_obs
    accel_at_obs = delta_at_obs - (prev_at_obs - prev2_at_obs)

    def _scatter_ffill(values_at_obs: np.ndarray) -> np.ndarray:
        full = np.full(n, np.nan)
        full[obs_positions] = values_at_obs
        return pd.Series(full).ffill().values

    previous_known_full = _scatter_ffill(prev_at_obs)
    second_previous_known_full = _scatter_ffill(prev2_at_obs)
    historical_delta_full = _scatter_ffill(delta_at_obs)
    state_acceleration_full = _scatter_ffill(accel_at_obs)

    grp_id = notna.astype(int).cumsum()
    blackout_streak = grp_id.groupby(grp_id).cumcount().values
    n_history_observations = grp_id.values

    window_trend = max(trailing_windows)
    local_trend = (
        s.rolling(window=window_trend, min_periods=1).apply(rolling_slope_raw, raw=True).values
    )

    acf_by_lag: dict[int, np.ndarray] = {}
    for lag in ACF_LAGS:
        shifted = s.shift(lag)
        acf_by_lag[lag] = s.expanding(min_periods=2).corr(shifted).values

    # Fixed denominator ``w`` (not the count of periods actually available
    # within the rolling window) to match build_state_snapshot's semantics:
    # a window that partially predates a location's first tracked month
    # still divides by the full window length, not by however much history
    # happens to exist -- ``rolling(...).mean()`` would silently divide by
    # the smaller available-period count near the start of a location's
    # range instead, which is a different (and wrong, for this purpose)
    # quantity.
    density_by_window = {
        w: (notna.rolling(window=w, min_periods=1).sum() / w).values for w in trailing_windows
    }

    seasonal_position = ((full_range.month - 1) / 12.0).to_numpy(dtype=float)

    is_observed_now = notna.values
    state_status = np.array(
        [
            _derive_state_status(
                is_observed_now=bool(is_observed_now[i]),
                months_since_observation=(
                    None if np.isnan(months_since[i]) else int(months_since[i])
                ),
                n_history_observations=int(n_history_observations[i]),
                max_reconstruction_gap_months=max_reconstruction_gap_months,
                min_evidence_observations=min_evidence_observations,
            )
            for i in range(n)
        ],
        dtype=object,
    )

    period_to_pos = {p: i for i, p in enumerate(full_range)}
    row_periods = pd.PeriodIndex(pd.to_datetime(group[as_of_column]), freq="M")
    row_pos = np.array([period_to_pos[p] for p in row_periods])

    out = pd.DataFrame(
        {
            "location_id": location_id,
            "as_of": pd.to_datetime(group[as_of_column]).values,
            "last_known_tws": last_known_tws[row_pos],
            "last_known_time": last_known_time_arr[row_pos],
            "months_since_observation": months_since[row_pos],
            "previous_known_tws": previous_known_full[row_pos],
            "second_previous_known_tws": second_previous_known_full[row_pos],
            "historical_delta": historical_delta_full[row_pos],
            "state_acceleration": state_acceleration_full[row_pos],
            "local_trend": local_trend[row_pos],
            "seasonal_position": seasonal_position[row_pos],
            "blackout_streak_length": blackout_streak[row_pos],
            "state_status": state_status[row_pos],
            "location_signature": None,
        },
        index=group.index,
    )
    for lag in ACF_LAGS:
        out[f"acf_lag{lag}"] = acf_by_lag[lag][row_pos]
    for w in trailing_windows:
        out[f"observation_density_{w}"] = density_by_window[w][row_pos]

    return out


def build_state_snapshots(
    df: pd.DataFrame,
    as_of_column: str = "time",
    trailing_windows: tuple[int, ...] = (12, 24),
    max_reconstruction_gap_months: int = DEFAULT_MAX_RECONSTRUCTION_GAP_MONTHS,
    min_evidence_observations: int = DEFAULT_MIN_EVIDENCE_OBSERVATIONS,
) -> pd.DataFrame:
    """Vectorized batch equivalent of calling :func:`build_state_snapshot`
    for every row of ``df`` (at that row's own ``as_of_column`` value) and
    stacking the results — the function the Phase 4 feature-assembly
    pipeline (step 4.9) actually calls, since building millions of
    ``StateSnapshot`` objects one Python call at a time would dominate
    pipeline wall-clock for no extra correctness (the same design lesson
    ``validation.splitters.attach_forecast_origin_columns`` already applied
    for ``ForecastOrigin``).

    Computation is vectorized per-location (``groupby("location_id")``,
    ``ffill``/``rolling``/``expanding`` within each location's own
    continuous monthly reindex) — a Python-level loop over ~15,715
    locations, never over the millions of underlying rows.

    Returns a ``pd.DataFrame`` indexed identically to ``df`` (so it can be
    joined back with ``pd.concat([df, snapshots], axis=1)``), with one
    column per scalar ``StateSnapshot`` field plus ``acf_lag{lag}`` for
    ``lag in ACF_LAGS`` and ``observation_density_{w}`` for
    ``w in trailing_windows`` (the batch variant's flat-columns
    representation of ``StateSnapshot.acf_1_3_6_12``/``observation_density``'s
    dict-valued fields).
    """
    frame = ensure_location_id(df).copy()
    frame[as_of_column] = pd.to_datetime(frame[as_of_column])

    pieces = [
        _build_snapshots_for_location(
            group,
            location_id,
            as_of_column,
            trailing_windows,
            max_reconstruction_gap_months,
            min_evidence_observations,
        )
        for location_id, group in frame.groupby("location_id", sort=False)
    ]

    if not pieces:
        columns = [
            "location_id",
            "as_of",
            "last_known_tws",
            "last_known_time",
            "months_since_observation",
            "previous_known_tws",
            "second_previous_known_tws",
            "historical_delta",
            "state_acceleration",
            "local_trend",
            "seasonal_position",
            "blackout_streak_length",
            "state_status",
            "location_signature",
            *[f"acf_lag{lag}" for lag in ACF_LAGS],
            *[f"observation_density_{w}" for w in trailing_windows],
        ]
        return pd.DataFrame(columns=columns)

    return pd.concat(pieces, ignore_index=False).sort_index()
