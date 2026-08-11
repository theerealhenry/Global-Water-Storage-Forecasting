"""State-reconstruction schemas.

This module holds the two canonical schemas ``docs/ARCHITECTURE.md`` §4
assigns to it: ``ForecastOrigin`` and ``StateSnapshot``. Only
``ForecastOrigin`` is built here, in Project Phase 2 — it's the origin/cutoff
join key the validation engine (splitters, masking simulator, tiers,
decomposition) needs to exist before it can enforce "nothing after the
forecast origin" as a checkable invariant rather than a convention.
``StateSnapshot`` (last-known value, observation age, trajectory, ACF,
location signature, ``state_status``) is deliberately **not** added until
Project Phase 4, per ``docs/ARCHITECTURE.md`` §6's co-location of both
schemas in this one file and ``docs/PHASE2_EXECUTION_PLAN.md`` step 2.2 —
this file being "incomplete" relative to the architecture doc right now is
expected, not a bug to fix early.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["ForecastOrigin", "location_id_from_lat_lon"]

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
