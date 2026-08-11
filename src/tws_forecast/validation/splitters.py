"""Time-respecting expanding-window CV splits.

Never random K-fold. The training period's confirmed non-stationary trend
(``notebooks/01_eda.ipynb`` §7: mean target drifts from +0.23 in 2002 to
about -0.13 in 2012-2015) and the 2015 persistence-RMSE anomaly (A-004,
``docs/ASSUMPTIONS.md``) both make a random split actively misleading — it
would let a fold's training portion include 2015-like months while
validating on an earlier, easier year, silently hiding exactly the kind of
regime drift Project Phase 1 spent an entire experiment sequence
characterizing.

Every fold produced here is genuinely "expanding": training always starts at
``TRAIN_PERIOD_START`` and grows forward to each fold's cutoff — never a
fixed-size window that slides forward, which would throw away early history
a real, growing-over-time system would actually have available.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from tws_forecast.state.reconstruction import location_id_from_lat_lon
from tws_forecast.utils.seeds import RANDOM_SEED, set_seed
from tws_forecast.validation.phase1_constants import (
    CLEAN_TRAIN_SPAN_END,
    TRAIN_PERIOD_END,
    TRAIN_PERIOD_START,
)

logger = logging.getLogger(__name__)

__all__ = ["expanding_window_splits", "FORECAST_ORIGIN_COLUMNS"]

# The ForecastOrigin schema's fields (state/reconstruction.py), attached as
# columns to every fold this module returns. Design note: a real fold can be
# millions of rows, and building one ForecastOrigin dataclass instance per
# row just to get these same six values would be pure overhead for no extra
# safety here — so folds carry the schema's fields as vectorized columns
# instead of a column of objects. Per-row ForecastOrigin objects remain
# available on demand via ForecastOrigin.from_row() for any code path that
# genuinely needs one (e.g. a single-row diagnostic). Correctness of the
# vectorized columns against ForecastOrigin.from_row is pinned by
# tests/test_splitters.py's row-by-row consistency check, not just assumed.
FORECAST_ORIGIN_COLUMNS = [
    "origin_time", "target_time", "horizon", "information_cutoff",
    "location_id", "regime",
]


def _month_index(ts: pd.Timestamp | str) -> int:
    """Months since year 0, as a plain integer — lets fold-boundary
    arithmetic use simple integer spacing instead of repeated date-offset
    arithmetic."""
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month


def _month_index_to_timestamp(idx: int) -> pd.Timestamp:
    year, month = divmod(idx, 12)
    if month == 0:
        year -= 1
        month = 12
    return pd.Timestamp(year=year, month=month, day=1)


def _attach_forecast_origin_columns(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Vectorized equivalent of calling ``ForecastOrigin.from_row`` on every
    row of ``df`` and attaching the results as columns."""
    out = df.copy()
    origin_time = pd.to_datetime(out["time"])
    out["origin_time"] = origin_time
    out["target_time"] = origin_time + pd.DateOffset(months=horizon)
    out["horizon"] = horizon
    out["information_cutoff"] = origin_time
    out["location_id"] = [
        location_id_from_lat_lon(lat, lon)
        for lat, lon in zip(out["lat"], out["lon"], strict=True)
    ]

    if "TWS_t_masked" in out.columns:
        is_masked = out["TWS_t_masked"].astype(bool)
    elif "TWS_t" in out.columns:
        is_masked = out["TWS_t"].isna()
    else:
        is_masked = pd.Series(False, index=out.index)
    out["regime"] = np.where(is_masked, "masked", "observed")

    return out


def expanding_window_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    val_window_months: int = 6,
    min_train_months: int = 84,
    anchor_to_2004: bool = True,
    seed: int = RANDOM_SEED,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train_fold, val_fold)`` expanding-window CV splits.

    Fold boundaries are chosen so that (a) the earliest fold's training
    portion always includes a full pass through the verified gap-free
    2004-2010 span (``phase1_constants.CLEAN_TRAIN_SPAN_END``), and (b) the
    final fold's validation window reaches into 2015 — the training period's
    last and anomalous year (A-004) — rather than stopping short of it. Both
    are deliberate: (a) guarantees every fold has learned from at least the
    one span Project Phase 1 confirmed has zero missing calendar months;
    (b) forces the harness to be evaluated against the 2015 anomaly directly
    in at least one fold, rather than letting an earlier, easier fold boundary
    average it away.

    Parameters
    ----------
    df:
        A frame with a ``time`` column (``Train.csv``-shaped — this function
        is for CV over the *training* period; it does not touch Test.csv).
    n_folds:
        Number of folds. With ``n_folds=1``, the single fold produced is the
        one satisfying both (a) and (b) above simultaneously.
    val_window_months:
        Width of each fold's validation window, in months.
    min_train_months:
        Only used when ``anchor_to_2004=False`` — the earliest cutoff is then
        ``TRAIN_PERIOD_START + min_train_months - 1`` instead of the clean-span
        anchor.
    anchor_to_2004:
        If True (default), the earliest fold's cutoff is pinned to
        ``CLEAN_TRAIN_SPAN_END`` (2010-12), guaranteeing requirement (a)
        above regardless of ``min_train_months``.
    seed:
        Fold construction here is pure integer arithmetic, not stochastic —
        but every public entrypoint in this project seeds explicitly per
        ``utils/seeds.py``'s standing discipline, so this stays correct if
        fold construction ever gains a stochastic element (e.g. jittered
        boundaries) without a silent gap in seeding coverage.

    Yields
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_fold, val_fold)``, both the corresponding slice of ``df``
        with ``FORECAST_ORIGIN_COLUMNS`` attached. Every row in ``val_fold``
        has ``time`` strictly greater than every row in the same
        ``train_fold`` — the literal leakage invariant, enforced by
        construction and checked in ``tests/test_splitters.py``.
    """
    set_seed(seed)

    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if val_window_months < 1:
        raise ValueError(f"val_window_months must be >= 1, got {val_window_months}")

    train_start_idx = _month_index(TRAIN_PERIOD_START)
    train_end_idx = _month_index(TRAIN_PERIOD_END)

    first_cutoff_idx = (
        _month_index(CLEAN_TRAIN_SPAN_END)
        if anchor_to_2004
        else train_start_idx + min_train_months - 1
    )
    last_cutoff_idx = train_end_idx - val_window_months

    if last_cutoff_idx < first_cutoff_idx:
        raise ValueError(
            "Not enough months between the earliest required training cutoff "
            f"({_month_index_to_timestamp(first_cutoff_idx).date()}) and the "
            f"latest possible cutoff ({_month_index_to_timestamp(last_cutoff_idx).date()}) "
            f"to fit val_window_months={val_window_months}. Reduce "
            "val_window_months, or set anchor_to_2004=False."
        )

    if n_folds == 1:
        cutoff_idxs = [last_cutoff_idx]
    else:
        raw = np.linspace(first_cutoff_idx, last_cutoff_idx, n_folds)
        # linspace + rounding can collapse close points to the same integer
        # month when n_folds is large relative to the available range —
        # dedup rather than silently yielding two identical folds.
        cutoff_idxs = sorted({int(round(x)) for x in raw})

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    for cutoff_idx in cutoff_idxs:
        cutoff_time = _month_index_to_timestamp(cutoff_idx)
        val_end_time = _month_index_to_timestamp(cutoff_idx + val_window_months)

        train_mask = df["time"] <= cutoff_time
        val_mask = (df["time"] > cutoff_time) & (df["time"] <= val_end_time)

        train_fold = _attach_forecast_origin_columns(df.loc[train_mask])
        val_fold = _attach_forecast_origin_columns(df.loc[val_mask])

        logger.info(
            "Fold cutoff=%s: train %d rows, val %d rows (val window %s to %s)",
            cutoff_time.date(), len(train_fold), len(val_fold),
            cutoff_time.date(), val_end_time.date(),
        )

        yield train_fold, val_fold
