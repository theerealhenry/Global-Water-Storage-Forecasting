"""Streak-aware masking simulator.

Reproduces the real structure Project Phase 1 measured: masking arrives as a
contiguous run of whole calendar months, affecting nearly the entire grid at
once, with a small number of scattered, non-recurring exceptions — never a
flat, row-independent probability (``docs/ARCHITECTURE.md`` §8; Experiment 1,
``notebooks/02_forecastability.ipynb`` §2).

This module is deliberately generic: it knows how to apply *one*
``MaskingScenario`` to a frame, and nothing about where that scenario came
from. It has no notion of "curve mode" or "replay mode" — those are named,
concrete ``MaskingScenario`` instances registered in
``validation/scenarios.py`` (Phase 2 step 2.5), so this file stays a clean,
reusable primitive instead of accumulating scenario-specific branches.

Two kinds of masking are never conflated (``docs/ARCHITECTURE.md`` §8):
*historical* masking (what actually happened to the real test months) is
characterized by ``phase1_constants.py``; *synthetic* masking, applied here
to historical periods where TWS is in fact known, manufactures training and
validation examples that resemble that real structure. Nothing in this
module ever reads the real test set's own mask pattern as an input.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from tws_forecast.state.reconstruction import location_id_from_lat_lon
from tws_forecast.utils.seeds import RANDOM_SEED, set_seed

logger = logging.getLogger(__name__)

__all__ = ["MaskingScenario", "apply_masking", "apply_blackout_curve"]

TransitionPattern = Literal["abrupt", "ramp_in", "ramp_out"]

# Only "abrupt" is implemented — this is what Experiment 1 actually found
# (every test month is either 0% or 99.58-99.97% masked, never in between).
# "ramp_in"/"ramp_out" are declared in the type now, per
# docs/ARCHITECTURE.md §8's config contract, so a future robustness scenario
# can be *typed* without a schema change — but apply_masking() raises
# NotImplementedError if one is actually used, rather than silently treating
# it as "abrupt", since Phase 1 has no evidence to justify a specific ramp
# shape yet.
_IMPLEMENTED_TRANSITION_PATTERNS = {"abrupt"}


class MaskingScenario(BaseModel):
    """A declarative description of one synthetic blackout episode.

    Fields match ``docs/ARCHITECTURE.md`` §8's ``MaskingScenario`` config
    contract exactly: ``blackout_start, blackout_end, affected_locations,
    exception_rate, streak_length, transition_pattern, source_rationale``.

    Attributes
    ----------
    blackout_start, blackout_end:
        First and last masked calendar month (inclusive), both required to
        be the first of the month, matching every ``time`` value in the raw
        data (``docs/DATA_DICTIONARY.md``).
    affected_locations:
        ``"all"`` (default) — every location in the frame ``apply_masking``
        is called on is a masking *candidate* — or an explicit tuple of
        ``location_id`` strings (``"{lat}_{lon}"``,
        ``state/reconstruction.location_id_from_lat_lon``) to restrict
        masking to. Note this is the *candidate* set; ``exception_rate``
        below is what actually produces the real scattered-exception
        pattern within it, matching how Experiment 1 found blackout months
        behave: not a fixed set of "always-observed" locations, but a
        different scattered handful each time.
    exception_rate:
        Fraction of candidate rows that stay observed despite falling in
        the blackout window — reproduces Experiment 1's finding of a
        handful of scattered, non-recurring unmasked rows per blackout
        month (4 to 65 out of 15,715, mean overlap across month-pairs 1.53).
        Must be in ``[0, 1)``.
    streak_length:
        Number of consecutive months in the blackout window. Stored
        explicitly (rather than only derived from the start/end dates) so
        code building the error-decomposition table (step 2.7) can bucket
        by it without recomputing date arithmetic — validated to actually
        match ``blackout_start``/``blackout_end`` at construction time.
    transition_pattern:
        See module-level note — only ``"abrupt"`` is currently implemented.
    source_rationale:
        Free-text provenance — e.g. "Experiment 4 Method B, real calendar
        replay" or "synthetic, k resampled from BLACKOUT_K_DISTRIBUTION".
        Required, not optional: every scenario used anywhere in this project
        must say where its parameters came from (``docs/ARCHITECTURE.md``
        §11's config-driven-scenario requirement).
    """

    model_config = ConfigDict(frozen=True)

    blackout_start: date
    blackout_end: date
    affected_locations: Literal["all"] | tuple[str, ...] = "all"
    exception_rate: float = 0.0
    streak_length: int
    transition_pattern: TransitionPattern = "abrupt"
    source_rationale: str

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> MaskingScenario:
        if self.blackout_start.day != 1 or self.blackout_end.day != 1:
            raise ValueError(
                "blackout_start/blackout_end must be the first of the month "
                f"(got {self.blackout_start}, {self.blackout_end}) — every "
                "'time' value in the raw data is a month-start."
            )
        if self.blackout_end < self.blackout_start:
            raise ValueError(
                f"blackout_end ({self.blackout_end}) is before blackout_start "
                f"({self.blackout_start})."
            )
        expected_streak = (
            (self.blackout_end.year * 12 + self.blackout_end.month)
            - (self.blackout_start.year * 12 + self.blackout_start.month)
            + 1
        )
        if expected_streak != self.streak_length:
            raise ValueError(
                f"streak_length={self.streak_length} does not match the "
                f"{expected_streak}-month span from {self.blackout_start} to "
                f"{self.blackout_end} — streak_length must be derived, not "
                "asserted independently."
            )
        if not (0.0 <= self.exception_rate < 1.0):
            raise ValueError(f"exception_rate must be in [0, 1), got {self.exception_rate}")
        return self


def apply_masking(
    df: pd.DataFrame,
    scenario: MaskingScenario,
    seed: int = RANDOM_SEED,
    derived_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply one ``MaskingScenario`` to ``df``, returning a masked copy.

    Nulls ``TWS_t`` (and any column in ``derived_columns``) for rows that
    fall in the blackout window, belong to an affected location, and are not
    drawn as a scattered exception. ``derived_columns`` defaults to
    ``[]`` — no state/feature columns exist yet as of Project Phase 2; this
    parameter exists so Project Phase 4's derived columns (last-known value,
    trajectory, etc.) can be nulled alongside ``TWS_t`` without this function
    needing to change.

    The function never adds or removes rows — it only ever operates on rows
    that already exist in ``df`` (Experiments 1 and 3's finding that neither
    raw file ever has the complete 15,715-location grid in a single month is
    respected by construction, not by special-casing).

    Parameters
    ----------
    df:
        A frame with ``time``, ``lat``, ``lon``, ``TWS_t`` columns
        (``Train.csv``- or ``Test.csv``-shaped).
    scenario:
        The blackout episode to apply.
    seed:
        Seeds the exception-rate draw. Two calls with the same seed and
        scenario on the same input produce byte-identical output.
    derived_columns:
        Additional columns to null alongside ``TWS_t`` for masked rows.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``TWS_t`` (and any ``derived_columns``) null
        for masked rows, and a ``TWS_t_masked`` column that exactly equals
        ``TWS_t.isna()`` — the same invariant ``docs/DATA_DICTIONARY.md``
        documents for the real ``Test.csv``, verified with zero mismatches
        there and enforced by construction here.
    """
    if scenario.transition_pattern not in _IMPLEMENTED_TRANSITION_PATTERNS:
        raise NotImplementedError(
            f"transition_pattern={scenario.transition_pattern!r} is declared "
            "in the schema but not yet implemented — only 'abrupt' is "
            "currently backed by Phase 1 evidence (Experiment 1's bimodal "
            "masking finding)."
        )

    set_seed(seed)

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])

    blackout_start = pd.Timestamp(scenario.blackout_start)
    blackout_end = pd.Timestamp(scenario.blackout_end)
    in_window = (out["time"] >= blackout_start) & (out["time"] <= blackout_end)

    if scenario.affected_locations == "all":
        location_selected = pd.Series(True, index=out.index)
    else:
        location_id = pd.Series(
            [location_id_from_lat_lon(lat, lon) for lat, lon in zip(out["lat"], out["lon"])],
            index=out.index,
        )
        location_selected = location_id.isin(scenario.affected_locations)

    candidate_mask = in_window & location_selected

    if scenario.exception_rate > 0 and candidate_mask.any():
        exception_draw = np.random.random(len(out)) < scenario.exception_rate
        final_mask = candidate_mask & ~exception_draw
    else:
        final_mask = candidate_mask

    columns_to_null = ["TWS_t", *(derived_columns or [])]
    for col in columns_to_null:
        if col in out.columns:
            out.loc[final_mask, col] = np.nan
        else:
            logger.warning("apply_masking: column %r not present in df, skipping", col)

    out["TWS_t_masked"] = out["TWS_t"].isna()

    logger.info(
        "Applied scenario %r: %d/%d candidate rows masked (window %s-%s, "
        "exception_rate=%.4f, source=%s)",
        scenario.source_rationale, int(final_mask.sum()), int(candidate_mask.sum()),
        blackout_start.date(), blackout_end.date(), scenario.exception_rate,
        scenario.source_rationale,
    )

    return out


def apply_blackout_curve(
    df: pd.DataFrame,
    k_distribution: list[int] | tuple[int, ...],
    n_windows: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Inject up to ``n_windows`` independent, per-location synthetic
    blackout runs into ``df``, each run's length ``k`` resampled with
    replacement from ``k_distribution``.

    Reproduces Experiment 3's multi-window blackout-curve design
    (``notebooks/02_forecastability.ipynb`` §9-10) as a reusable primitive
    that ``validation/tiers.py``'s Tier 2 calls, rather than each rebuilding
    its own copy of "pick some locations, mask their most recent k months."
    Distinct from ``apply_masking``: that function applies one shared
    date-window scenario to (potentially many) locations at once;
    this one gives *each* selected location its own independently-drawn
    window length, which a single ``MaskingScenario`` can't represent —
    so it lives here, next to ``apply_masking``, as this module's second
    masking primitive, rather than being hand-rolled inside ``tiers.py``
    (``docs/PHASE2_EXECUTION_PLAN.md`` step 2.6's explicit requirement that
    no tier module reimplement masking logic itself).

    Locations are drawn *without* replacement (each gets at most one
    simulated run, so overlapping draws on the same location never need
    reconciling); ``k`` is drawn *with* replacement from ``k_distribution``,
    matching how the real distribution was built (Experiment 4).

    For each selected location, the run masks that location's own most
    recent ``min(k, rows available for that location in df)`` rows —
    "most recent available," not "most recent calendar month," so a
    location with a real gap inside the window (Experiments 1/3's grid-
    irregularity finding) is handled correctly by construction rather than
    by special-casing: the run always contains exactly that many real rows.

    Parameters
    ----------
    df:
        Must already carry a ``location_id`` column — i.e. this is a frame
        produced by ``validation.splitters.expanding_window_splits``, not a
        raw ``Train.csv`` load. This function does not recompute
        ``location_id`` itself, so there is exactly one code path
        (``state.reconstruction.location_id_from_lat_lon``, via the
        splitter) that can produce it.
    k_distribution:
        Values to resample ``k`` from, with replacement — pass
        ``phase1_constants.BLACKOUT_K_DISTRIBUTION`` (via a scenario
        config's ``k_distribution`` field) to match the real measured
        distribution, never an invented one.
    n_windows:
        Number of locations to draw a blackout run for (capped at the
        number of distinct locations actually present in ``df``).
    seed:
        Two calls with the same seed, ``df``, and parameters produce
        byte-identical output.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``TWS_t`` nulled for the selected rows,
        ``TWS_t_masked``/``regime`` recomputed to match, and a new
        ``simulated_k`` column (``NaN`` for untouched rows, the drawn
        ``k`` for masked rows) — this is what lets the error-decomposition
        table (step 2.7) bucket by the real k values instead of an
        approximate one.
    """
    if "location_id" not in df.columns:
        raise ValueError(
            "apply_blackout_curve requires a location_id column — pass a "
            "frame produced by validation.splitters.expanding_window_splits, "
            "not a raw data load."
        )
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if len(k_distribution) == 0:
        raise ValueError("k_distribution must not be empty")

    set_seed(seed)
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])

    locations = out["location_id"].unique()
    n_draws = min(n_windows, len(locations))

    mask_flags = pd.Series(False, index=out.index)
    simulated_k = pd.Series(np.nan, index=out.index)

    if n_draws > 0:
        rng = np.random.default_rng(seed)
        chosen_locations = rng.choice(locations, size=n_draws, replace=False)
        chosen_ks = rng.choice(np.asarray(k_distribution), size=n_draws, replace=True)

        for loc, k in zip(chosen_locations, chosen_ks):
            loc_idx = out.index[out["location_id"] == loc]
            if len(loc_idx) == 0:
                continue
            ordered_idx = out.loc[loc_idx, "time"].sort_values().index
            streak = min(int(k), len(ordered_idx))
            selected_idx = ordered_idx[-streak:]
            mask_flags.loc[selected_idx] = True
            simulated_k.loc[selected_idx] = int(k)

    out.loc[mask_flags, "TWS_t"] = np.nan
    out["TWS_t_masked"] = out["TWS_t"].isna()
    out["regime"] = np.where(out["TWS_t_masked"], "masked", "observed")
    out["simulated_k"] = simulated_k

    logger.info(
        "apply_blackout_curve: drew %d/%d requested windows (locations "
        "available=%d), %d rows newly masked",
        n_draws, n_windows, len(locations), int(mask_flags.sum()),
    )

    return out
