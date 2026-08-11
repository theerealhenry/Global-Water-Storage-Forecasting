"""Three validation tiers, each answering a distinct question.

Tier 1 (forecastability): can the model predict next-month TWS under
normal, fully-observed conditions? Standard expanding-window CV, no masking.

Tier 2 (blackout): can the model forecast after losing the current
observation? Injects synthetic blackout runs, length resampled from the
real staleness distribution, within each Tier 1 fold.

Tier 3 (test-regime): does the model reproduce the real 18-month test
calendar structure's behavior on historical analogs where the ground truth
is actually known? Diagnostic/robustness only — ``docs/ARCHITECTURE.md``
§11: "Final model selection is always made against Tier 1 and Tier 2...
never by repeated tuning against Tier 3 analogs." This module computes
Tier 3's score; enforcing that a Tier-3-only score can't promote a candidate
is ``validation/harness.py``'s job (Phase 2 step 2.9), not this one's.

None of the three functions here reimplements split or masking logic —
each composes ``validation.splitters`` (fold generation) with
``validation.scenarios`` (which named, versioned config to run) and
``validation.masking_simulator`` (how masking is actually applied).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from tws_forecast.utils.dates import month_index, month_index_to_timestamp
from tws_forecast.utils.seeds import RANDOM_SEED
from tws_forecast.validation.masking_simulator import apply_blackout_curve
from tws_forecast.validation.scenarios import load_scenario
from tws_forecast.validation.splitters import (
    FORECAST_ORIGIN_COLUMNS,
    attach_forecast_origin_columns,
    expanding_window_splits,
)

logger = logging.getLogger(__name__)

__all__ = ["Predictor", "TierResult", "run_tier1", "run_tier2", "run_tier3"]


class Predictor(Protocol):
    """The minimal interface every tier requires of a model.

    Deliberately sklearn-shaped (``fit``/``predict``) so any future
    baseline (Project Phase 3) or real model (Project Phase 5+) plugs in
    without a tiers.py change. Stateless baselines (a global-mean predictor,
    persistence) implement ``fit`` as a no-op.
    """

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class TierResult:
    """The standard output every tier function returns.

    ``predictions`` is keyed by the ``FORECAST_ORIGIN_COLUMNS`` (plus a
    ``fold`` column identifying which fold/anchor produced each row, and
    tier-specific metadata columns — ``simulated_k`` for Tiers 2/3,
    ``replay_offset`` for Tier 3) — this is the frame
    ``validation/decomposition.py`` (step 2.7) reads from to build the
    error-decomposition table; nothing here computes that table itself.
    """

    tier: int
    scenario_name: str
    predictions: pd.DataFrame
    fold_rmses: tuple[float, ...]
    overall_rmse: float

    @property
    def rmse_mean(self) -> float:
        return float(np.mean(self.fold_rmses)) if self.fold_rmses else float("nan")

    @property
    def rmse_std(self) -> float:
        return float(np.std(self.fold_rmses)) if self.fold_rmses else float("nan")

    def __repr__(self) -> str:
        return (
            f"TierResult(tier={self.tier}, scenario={self.scenario_name!r}, "
            f"overall_rmse={self.overall_rmse:.4f}, fold_rmse={self.rmse_mean:.4f}"
            f"±{self.rmse_std:.4f} (n={len(self.fold_rmses)}), "
            f"n_predictions={len(self.predictions)})"
        )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_tier1(
    model: Predictor, df: pd.DataFrame, scenario: str = "expanding_window"
) -> TierResult:
    """Tier 1 — forecastability. No masking; Baseline A (0.5247) is the
    realistic ceiling for whatever this tier measures."""
    config = load_scenario(scenario)
    if config.scenario_type != "expanding_window":
        raise ValueError(
            f"run_tier1 requires scenario_type='expanding_window', got "
            f"{config.scenario_type!r} for scenario={scenario!r}"
        )

    fold_rmses: list[float] = []
    all_predictions: list[pd.DataFrame] = []

    for fold_idx, (train_fold, val_fold) in enumerate(
        expanding_window_splits(df, **config.splitter.model_dump())
    ):
        if len(val_fold) == 0:
            logger.warning("run_tier1 fold %d has an empty validation window, skipping", fold_idx)
            continue

        model.fit(train_fold)
        preds = model.predict(val_fold)
        fold_rmse = _rmse(val_fold["target"].values, preds)
        fold_rmses.append(fold_rmse)

        pred_df = val_fold[FORECAST_ORIGIN_COLUMNS].copy()
        pred_df["prediction"] = preds
        pred_df["target"] = val_fold["target"].values
        pred_df["fold"] = fold_idx
        all_predictions.append(pred_df)

    if not all_predictions:
        raise ValueError("run_tier1 produced no predictions — check df's coverage of the training period.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_rmse = _rmse(predictions["target"].values, predictions["prediction"].values)

    return TierResult(
        tier=1, scenario_name=scenario, predictions=predictions,
        fold_rmses=tuple(fold_rmses), overall_rmse=overall_rmse,
    )


def run_tier2(
    model: Predictor, df: pd.DataFrame, scenario: str = "blackout_curve"
) -> TierResult:
    """Tier 2 — blackout. Within each Tier 1-style fold's validation
    window, ``apply_blackout_curve`` injects synthetic staleness before the
    model is scored — this is the tier Project Phase 4's state-
    reconstruction features are directly tested against."""
    config = load_scenario(scenario)
    if config.scenario_type != "blackout_curve":
        raise ValueError(
            f"run_tier2 requires scenario_type='blackout_curve', got "
            f"{config.scenario_type!r} for scenario={scenario!r}"
        )

    fold_rmses: list[float] = []
    all_predictions: list[pd.DataFrame] = []

    for fold_idx, (train_fold, val_fold) in enumerate(
        expanding_window_splits(df, **config.splitter.model_dump())
    ):
        if len(val_fold) == 0:
            logger.warning("run_tier2 fold %d has an empty validation window, skipping", fold_idx)
            continue

        # A distinct-but-deterministic seed per fold: reusing RANDOM_SEED
        # unmodified across all folds would draw the exact same locations
        # and k-values every time, which would understate how well the
        # model handles the *range* of masking scenarios (docs/PHASE2_
        # EXECUTION_PLAN.md's blackout-curve tier is meant to sample
        # broadly, per Experiment 3's own multi-window design).
        masked_val_fold = apply_blackout_curve(
            val_fold,
            k_distribution=config.k_distribution,
            n_windows=config.n_windows,
            seed=RANDOM_SEED + fold_idx,
        )

        model.fit(train_fold)
        preds = model.predict(masked_val_fold)
        fold_rmse = _rmse(masked_val_fold["target"].values, preds)
        fold_rmses.append(fold_rmse)

        pred_df = masked_val_fold[[*FORECAST_ORIGIN_COLUMNS, "simulated_k"]].copy()
        pred_df["prediction"] = preds
        pred_df["target"] = masked_val_fold["target"].values
        pred_df["fold"] = fold_idx
        all_predictions.append(pred_df)

    if not all_predictions:
        raise ValueError("run_tier2 produced no predictions — check df's coverage of the training period.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_rmse = _rmse(predictions["target"].values, predictions["prediction"].values)

    return TierResult(
        tier=2, scenario_name=scenario, predictions=predictions,
        fold_rmses=tuple(fold_rmses), overall_rmse=overall_rmse,
    )


def _select_replay_anchors(
    df: pd.DataFrame, pattern_length_months: int, n_anchors: int
) -> list[pd.Timestamp]:
    """Evenly-spaced candidate anchor months such that (a) at least one
    month of history exists strictly before the anchor to fit on, and (b)
    the full ``pattern_length_months``-month replay pattern fits within
    ``df``'s available time range."""
    times = pd.to_datetime(df["time"])
    if len(times) == 0:
        return []

    min_idx = month_index(times.min())
    max_idx = month_index(times.max())

    earliest_anchor_idx = min_idx + 1  # leave >=1 month of prior history
    latest_anchor_idx = max_idx - pattern_length_months + 1

    if latest_anchor_idx < earliest_anchor_idx:
        return []

    if n_anchors == 1:
        idxs = [latest_anchor_idx]
    else:
        raw = np.linspace(earliest_anchor_idx, latest_anchor_idx, n_anchors)
        idxs = sorted({int(round(x)) for x in raw})

    return [month_index_to_timestamp(i) for i in idxs]


def run_tier3(
    model: Predictor,
    df: pd.DataFrame,
    scenario: str = "test_regime_replay",
    n_anchors: int = 3,
) -> TierResult:
    """Tier 3 — test-regime replay. Diagnostic/robustness only.

    Replays the real FULL/BLACKOUT calendar-offset pattern
    (``config.full_offsets``/``blackout_offsets``/``blackout_k_by_offset``)
    onto ``n_anchors`` historical analog windows, ground-truth-scorable
    because ``df``'s real values are known there (Experiment 4 Method B).
    For each anchor, the model is fit **only on data strictly before that
    anchor** — the same leakage-safe discipline every other tier uses, not
    a special exception for the "diagnostic" tier.

    Blackout offsets simulate staleness by nulling that offset's own
    origin-row ``TWS_t`` (this tier operates row-wise, the same as Tiers 1
    and 2 — it does not yet null a location's *preceding* k-1 months in the
    broader frame, since no lag/history feature exists to read them until
    Project Phase 4; ``simulated_k`` is still recorded as metadata for the
    decomposition table, describing the *intended* staleness even before a
    model exists that can actually see that deeper history).
    """
    config = load_scenario(scenario)
    if config.scenario_type != "test_regime_replay":
        raise ValueError(
            f"run_tier3 requires scenario_type='test_regime_replay', got "
            f"{config.scenario_type!r} for scenario={scenario!r}"
        )
    if n_anchors < 1:
        raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    full_offsets = list(config.full_offsets)  # type: ignore[arg-type]
    blackout_offsets = list(config.blackout_offsets)  # type: ignore[arg-type]
    blackout_k_by_offset = config.blackout_k_by_offset  # type: ignore[assignment]
    pattern_length = max(full_offsets + blackout_offsets) + 1

    anchors = _select_replay_anchors(df, pattern_length, n_anchors)
    if not anchors:
        raise ValueError(
            "Not enough historical data to fit even one Tier 3 replay anchor "
            f"(need >= {pattern_length + 1} months of coverage in df)."
        )

    offsets_with_meta = [(o, None) for o in full_offsets] + [
        (o, blackout_k_by_offset[o]) for o in blackout_offsets
    ]

    fold_rmses: list[float] = []
    all_predictions: list[pd.DataFrame] = []

    for anchor_idx, anchor in enumerate(anchors):
        train_data = df[df["time"] < anchor]
        if len(train_data) == 0:
            logger.warning("run_tier3 anchor %s has no prior history, skipping", anchor.date())
            continue
        model.fit(train_data)

        anchor_rows: list[pd.DataFrame] = []
        for offset, k in offsets_with_meta:
            origin_time = anchor + pd.DateOffset(months=offset)
            origin_rows = df[df["time"] == origin_time].copy()
            if len(origin_rows) == 0:
                continue  # real grid irregularity — no rows this calendar month

            origin_rows = attach_forecast_origin_columns(origin_rows)
            if k is not None:
                origin_rows["TWS_t"] = np.nan
            origin_rows["TWS_t_masked"] = origin_rows["TWS_t"].isna()
            origin_rows["regime"] = np.where(origin_rows["TWS_t_masked"], "masked", "observed")
            origin_rows["simulated_k"] = k if k is not None else np.nan
            origin_rows["replay_offset"] = offset

            preds = model.predict(origin_rows)
            pred_df = origin_rows[[*FORECAST_ORIGIN_COLUMNS, "simulated_k", "replay_offset"]].copy()
            pred_df["prediction"] = preds
            pred_df["target"] = origin_rows["target"].values
            pred_df["fold"] = anchor_idx
            anchor_rows.append(pred_df)

        if not anchor_rows:
            logger.warning("run_tier3 anchor %s produced no rows, skipping", anchor.date())
            continue

        anchor_df = pd.concat(anchor_rows, ignore_index=True)
        fold_rmse = _rmse(anchor_df["target"].values, anchor_df["prediction"].values)
        fold_rmses.append(fold_rmse)
        all_predictions.append(anchor_df)

    if not all_predictions:
        raise ValueError(
            "run_tier3 produced no predictions across any anchor — check df's "
            "coverage of the calendar months the replay pattern needs."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_rmse = _rmse(predictions["target"].values, predictions["prediction"].values)

    return TierResult(
        tier=3, scenario_name=scenario, predictions=predictions,
        fold_rmses=tuple(fold_rmses), overall_rmse=overall_rmse,
    )
