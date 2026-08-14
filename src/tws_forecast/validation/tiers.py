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
from tws_forecast.validation.phase1_constants import CLEAN_TRAIN_SPAN_END, CLEAN_TRAIN_SPAN_START
from tws_forecast.validation.scenarios import load_scenario
from tws_forecast.validation.splitters import (
    FORECAST_ORIGIN_COLUMNS,
    attach_forecast_origin_columns,
    expanding_window_splits,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Predictor", "TierResult", "run_tier1", "run_tier2", "run_tier3",
    "run_tier3_sequential_state",
]


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
    ``fold`` column identifying which fold/anchor produced each row,
    ``true_tws_t`` — the real, ground-truth current-month value, even for
    rows whose ``TWS_t`` the model itself was never shown — and
    tier-specific metadata columns: ``simulated_k`` for Tiers 2/3,
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
        # The real, ground-truth current-month TWS_t — always equal to what
        # the model was actually shown here (Tier 1 never masks), but kept
        # under this name for consistency with Tiers 2/3, where it's the
        # *pre-masking* value the model did NOT see. validation/
        # decomposition.py's extreme-value/rapid-change slices (step 2.7)
        # read this column, not the model's own (possibly masked) input.
        pred_df["true_tws_t"] = val_fold["TWS_t"].values
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
        # Captured before masking — the real value the model was NOT shown
        # for rows apply_blackout_curve selects, used only by validation/
        # decomposition.py's diagnostic slices, never fed back to the model.
        true_tws_t = val_fold["TWS_t"].values

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
        pred_df["true_tws_t"] = true_tws_t
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
    """Evenly-spaced candidate anchor months, restricted to the verified
    gap-free 2004-2010 span (``phase1_constants.CLEAN_TRAIN_SPAN_START/
    END``) so every anchor's full ``pattern_length_months``-month replay
    window lands on real, uninterrupted historical data — reproducing
    Experiment 4 Method B's own design ("8 independent windows of the
    verified clean 2004-2010 span," ``notebooks/02_forecastability.ipynb``
    §11.3), not merely spanning ``df``'s full available date range.

    Bug found and fixed during Project Phase 2 step 2.11's proof run
    (``notebooks/03_validation_harness.ipynb``): the original version of
    this function used ``df["time"].min()/max()`` as its anchor bounds,
    which let a candidate anchor's replay pattern run into the documented
    post-2010 missing-month gaps (A-012) or start with almost no prior fit
    history near ``TRAIN_PERIOD_START`` — both measured to pull Tier 3's
    score far away from Baseline D's validated 0.6573 (0.894 observed
    with the old, unrestricted anchor selection, at ``n_anchors=3``) even
    though the underlying per-anchor tier logic and masking were already
    correct. See ``docs/ASSUMPTIONS.md`` for the full write-up.
    """
    clean_start_idx = month_index(CLEAN_TRAIN_SPAN_START)
    clean_end_idx = month_index(CLEAN_TRAIN_SPAN_END)

    times = pd.to_datetime(df["time"])
    if len(times) == 0:
        return []

    min_idx = month_index(times.min())
    max_idx = month_index(times.max())

    earliest_anchor_idx = max(clean_start_idx, min_idx + 1)  # leave >=1 month of prior history
    latest_anchor_idx = min(clean_end_idx, max_idx) - pattern_length_months + 1

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
            true_tws_t = origin_rows["TWS_t"].to_numpy(copy=True)
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
            pred_df["true_tws_t"] = true_tws_t
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


def run_tier3_sequential_state(
    model: Predictor,
    df: pd.DataFrame,
    scenario: str = "test_regime_replay",
    n_anchors: int = 3,
    state_attr: str = "_last_known",
) -> TierResult:
    """Diagnostic-only sequential-state variant of :func:`run_tier3`, for
    internally stateful, non-feature-based predictors (Project Phase 3's
    ``LastKnownStatePredictor`` / ``HybridPersistencePredictor``) per A-013
    (``docs/ASSUMPTIONS.md``).

    ``run_tier3`` calls ``model.predict()`` independently for each of the
    18 replay offsets, by design — it's built for Project Phase 4's future
    feature-based models, which will read "what was last observed" from an
    explicit row-level column, not from a predictor's own internal memory.
    That design under-scores a genuinely stateful last-known-state
    predictor: by the time the real replay pattern reaches, say, offset 11
    (a BLACKOUT month), a real last-known-state forecaster should already
    know about earlier FULL offsets within the *same* 41-month pattern —
    but ``run_tier3`` never lets it see them, because offsets are scored in
    the config's own (not necessarily chronological) list order and each
    call is independent.

    This function walks the same replay pattern in **true chronological**
    offset order and, immediately after scoring each FULL (fully-observed)
    offset, updates the model's own ``state_attr`` dict (default
    ``"_last_known"``, matching both ``LastKnownStatePredictor`` and
    ``HybridPersistencePredictor``'s own attribute name) with that offset's
    real observed values — reproducing what a genuinely deployed
    last-known-state forecaster would actually know by the time it reaches
    a later BLACKOUT offset in the same window. First proven as an ad hoc
    notebook cell in ``notebooks/03_validation_harness.ipynb`` §7b (Project
    Phase 2 step 2.11); promoted here, tested, per the Project Phase 3
    handoff §3.0's note that a second copy-paste of the same ~30 lines is
    worth turning into a real function.

    This directly reaches into a private attribute of ``model``
    (``getattr(model, state_attr)``/``setattr``), which is exactly why this
    stays a diagnostic-only helper rather than something every Phase 3+
    candidate is scored with automatically — a real candidate is scored by
    the standard, harness-faithful :func:`run_tier3` only. Raises
    ``AttributeError`` immediately (rather than silently no-op-ing) if
    ``model`` has no ``state_attr`` attribute, since a diagnostic that
    quietly fails to do anything is worse than one that fails loudly.

    Parameters
    ----------
    model, df, scenario, n_anchors:
        Same meaning as :func:`run_tier3`.
    state_attr:
        Name of the ``{location_id: float}``-shaped dict attribute on
        ``model`` that its own ``predict()`` consults for "last known
        value." Both of this project's stateful baselines use
        ``"_last_known"`` (the default); overridable for any future
        stateful predictor using a different attribute name.

    Returns
    -------
    TierResult
        Same shape as :func:`run_tier3`'s return value, so it can be passed
        to ``validation.decomposition.decompose`` identically — but this
        result must never be used for promotion (``harness.promote()``
        doesn't distinguish it from a real Tier 3 result by construction,
        so callers are responsible for keeping it out of any
        ``CandidateReport`` passed to ``promote()``).
    """
    config = load_scenario(scenario)
    if config.scenario_type != "test_regime_replay":
        raise ValueError(
            f"run_tier3_sequential_state requires scenario_type='test_regime_replay', got "
            f"{config.scenario_type!r} for scenario={scenario!r}"
        )
    if n_anchors < 1:
        raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")
    if not hasattr(model, state_attr):
        raise AttributeError(
            f"run_tier3_sequential_state requires model to expose a "
            f"{state_attr!r} attribute (a {{location_id: float}} dict its own "
            "predict() consults) — got a model with no such attribute. This "
            "diagnostic only applies to internally stateful, non-feature-"
            "based predictors (see this function's docstring)."
        )

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

    # True chronological order, unlike run_tier3's full-offsets-then-
    # blackout-offsets grouping — this is the entire point of this function.
    offsets_chronological = sorted(
        [(o, None) for o in full_offsets] + [(o, blackout_k_by_offset[o]) for o in blackout_offsets],
        key=lambda pair: pair[0],
    )

    fold_rmses: list[float] = []
    all_predictions: list[pd.DataFrame] = []

    for anchor_idx, anchor in enumerate(anchors):
        train_data = df[df["time"] < anchor]
        if len(train_data) == 0:
            logger.warning(
                "run_tier3_sequential_state anchor %s has no prior history, skipping", anchor.date()
            )
            continue
        model.fit(train_data)

        anchor_rows: list[pd.DataFrame] = []
        for offset, k in offsets_chronological:
            origin_time = anchor + pd.DateOffset(months=offset)
            origin_rows = df[df["time"] == origin_time].copy()
            if len(origin_rows) == 0:
                continue

            origin_rows = attach_forecast_origin_columns(origin_rows)
            true_tws_t = origin_rows["TWS_t"].to_numpy(copy=True)
            is_blackout = k is not None
            if is_blackout:
                origin_rows["TWS_t"] = np.nan
            origin_rows["TWS_t_masked"] = origin_rows["TWS_t"].isna()
            origin_rows["regime"] = np.where(origin_rows["TWS_t_masked"], "masked", "observed")
            origin_rows["simulated_k"] = k if k is not None else np.nan
            origin_rows["replay_offset"] = offset

            preds = model.predict(origin_rows)

            # The crux: after scoring a FULL offset, teach the model's own
            # state dict about the real values it just saw, *before* any
            # later (chronologically) BLACKOUT offset is scored.
            if not is_blackout:
                state_dict = getattr(model, state_attr)
                for loc, val in zip(origin_rows["location_id"].to_numpy(), true_tws_t, strict=True):
                    state_dict[loc] = float(val)

            pred_df = origin_rows[[*FORECAST_ORIGIN_COLUMNS, "simulated_k", "replay_offset"]].copy()
            pred_df["prediction"] = preds
            pred_df["target"] = origin_rows["target"].values
            pred_df["true_tws_t"] = true_tws_t
            pred_df["fold"] = anchor_idx
            anchor_rows.append(pred_df)

        if not anchor_rows:
            logger.warning("run_tier3_sequential_state anchor %s produced no rows, skipping", anchor.date())
            continue

        anchor_df = pd.concat(anchor_rows, ignore_index=True)
        fold_rmse = _rmse(anchor_df["target"].values, anchor_df["prediction"].values)
        fold_rmses.append(fold_rmse)
        all_predictions.append(anchor_df)

    if not all_predictions:
        raise ValueError(
            "run_tier3_sequential_state produced no predictions across any anchor — check df's "
            "coverage of the calendar months the replay pattern needs."
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_rmse = _rmse(predictions["target"].values, predictions["prediction"].values)

    return TierResult(
        tier=3, scenario_name=scenario, predictions=predictions,
        fold_rmses=tuple(fold_rmses), overall_rmse=overall_rmse,
    )
