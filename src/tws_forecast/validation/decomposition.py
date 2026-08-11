"""Error-decomposition table and degradation-slope diagnostic.

Every model evaluated from Project Phase 3 onward gets a full
error-decomposition table, not a single aggregate RMSE
(``docs/ARCHITECTURE.md`` §11). This module builds that table from a
``validation.tiers.TierResult`` and computes the degradation-slope
comparison against Experiment 5's already-validated AR(1) theoretical
curve. Nothing here re-runs a model or re-applies masking — it only reads
the ``predictions`` frame a tier function already produced.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tws_forecast.validation.phase1_constants import ACF_QUARTILE_AR1_PARAMS
from tws_forecast.validation.tiers import TierResult

logger = logging.getLogger(__name__)

__all__ = ["decompose", "degradation_slope", "ACF_QUARTILE_ORDER"]

# Canonical quartile label order, matching phase1_constants.ACF_QUARTILE_AR1_PARAMS's
# keys exactly — low ACF (least persistent locations) to high ACF (most
# persistent). Any acf_lookup passed to decompose() gets relabeled into
# these same four names, regardless of the raw ACF values' own scale.
ACF_QUARTILE_ORDER = ["Q1_low_ACF", "Q2", "Q3", "Q4_high_ACF"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Small, pure duplicate of validation.tiers._rmse — not imported since
    # that one is private to tiers.py; three lines of arithmetic is a more
    # defensible duplication than reaching into another module's
    # underscore-prefixed internals.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _lat_from_location_id(location_id: pd.Series) -> pd.Series:
    """``location_id`` is always ``"{lat}_{lon}"``
    (``state.reconstruction.location_id_from_lat_lon``) — parsing it back
    out avoids needing a separate lat/lon join just for the hemisphere
    slice."""
    return location_id.str.split("_").str[0].astype(float)


def _compute_acf_quartiles(acf_lookup: pd.Series) -> pd.Series:
    """Map each location in ``acf_lookup`` to one of ``ACF_QUARTILE_ORDER``,
    computed from ``acf_lookup``'s own distribution (data-driven quartile
    boundaries, never a fixed absolute ACF threshold — the same approach
    Experiment 5 itself used).

    Ranks (not raw values) are passed to ``pd.qcut`` so a small or
    tied-heavy ``acf_lookup`` (e.g. in tests) never collapses two quartile
    bins into one from duplicate bin edges.
    """
    if acf_lookup.nunique() < 4:
        raise ValueError(
            f"acf_lookup has only {acf_lookup.nunique()} distinct value(s) — "
            "need at least 4 distinct locations to form quartiles."
        )
    ranks = acf_lookup.rank(method="first")
    quartile = pd.qcut(ranks, 4, labels=ACF_QUARTILE_ORDER)
    return quartile.astype(str)


def decompose(tier_result: TierResult, acf_lookup: pd.Series | None = None) -> pd.DataFrame:
    """Build the standard error-decomposition table for one ``TierResult``.

    Row groups, each a ``(slice_type, slice_value)`` pair with its own
    ``n``/``rmse``:

    - ``overall``: one row, everything.
    - ``regime``: ``observed`` / ``masked``.
    - ``staleness_bucket``: one row per real ``k`` value actually present
      in ``tier_result.predictions["simulated_k"]`` (Tiers 2/3 only) —
      always the real, empirically-measured k=2..7 buckets
      (``phase1_constants.BLACKOUT_K_VALUE_COUNTS``), never an invented
      "1-2mo/3-4mo/5+mo" scheme. Skipped entirely (logged, not silently
      empty) if ``tier_result`` carries no staleness information (Tier 1).
    - ``staleness_x_acf_quartile``: cross-cut of the above with
      ``acf_lookup``-derived quartiles, per A-010's finding that staleness
      alone under-describes the regime. Skipped if ``acf_lookup`` is not
      given (Project Phase 4 supplies the real per-location ACF; Phase 2
      only builds the plumbing) or if there's no staleness information to
      cross-cut in the first place. Locations present in the predictions
      but absent from ``acf_lookup`` are bucketed as ``"unknown_acf"``
      rather than silently dropped — every masked row is accounted for
      somewhere in the cross-cut, always.
    - ``hemisphere``: ``Northern`` (lat >= 0) / ``Southern`` (lat < 0),
      parsed directly from ``location_id``.
    - ``extreme_target``: ``extreme`` (top quartile of ``|target|`` within
      this result) / ``typical``.
    - ``rapid_change``: ``rapid`` (top quartile of ``|target - true_tws_t|``)
      / ``typical`` — skipped if ``true_tws_t`` isn't available.

    Thresholds for the last two slices are computed from the result's own
    distribution (its 75th percentile), not a fixed physical cutoff, so the
    slice stays meaningful across very different data subsets (a 10-location
    test fixture and the full 15,715-location grid alike).

    Parameters
    ----------
    tier_result:
        Output of ``validation.tiers.run_tier1/2/3``.
    acf_lookup:
        Optional ``pd.Series`` indexed by ``location_id``, giving each
        location's ACF(1) (or similar persistence measure). Required only
        for the ``staleness_x_acf_quartile`` rows.

    Returns
    -------
    pd.DataFrame
        Columns ``slice_type, slice_value, n, rmse``.
    """
    preds = tier_result.predictions
    if len(preds) == 0:
        raise ValueError("decompose() called on an empty predictions frame.")

    rows: list[dict[str, object]] = []

    def _add(slice_type: str, slice_value: str, mask: pd.Series) -> None:
        subset = preds[mask]
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "n": int(len(subset)),
                "rmse": (
                    _rmse(subset["target"].to_numpy(), subset["prediction"].to_numpy())
                    if len(subset) > 0
                    else float("nan")
                ),
            }
        )

    # 1. overall
    _add("overall", "overall", pd.Series(True, index=preds.index))

    # 2. regime
    for regime_value in ("observed", "masked"):
        _add("regime", regime_value, preds["regime"] == regime_value)

    # 3. staleness_bucket — real k values only
    has_staleness = "simulated_k" in preds.columns and preds["simulated_k"].notna().any()
    present_ks: list[int] = []
    if has_staleness:
        present_ks = sorted(int(k) for k in preds.loc[preds["simulated_k"].notna(), "simulated_k"].unique())
        for k in present_ks:
            _add("staleness_bucket", f"k={k}", preds["simulated_k"] == k)
    else:
        logger.info(
            "decompose(): no simulated_k values present (a Tier 1 result?) — "
            "skipping staleness_bucket and staleness_x_acf_quartile rows."
        )

    # 4. staleness x ACF quartile
    if has_staleness and acf_lookup is not None:
        quartile_by_location = _compute_acf_quartiles(acf_lookup)
        acf_quartile = preds["location_id"].map(quartile_by_location).fillna("unknown_acf")
        for k in present_ks:
            k_mask = preds["simulated_k"] == k
            for q in acf_quartile[k_mask].unique():
                _add("staleness_x_acf_quartile", f"k={k}|{q}", k_mask & (acf_quartile == q))
    elif has_staleness and acf_lookup is None:
        logger.info(
            "decompose(): acf_lookup not provided — skipping staleness_x_acf_quartile "
            "rows (Project Phase 4 supplies real per-location ACF)."
        )

    # 5. hemisphere
    lat = _lat_from_location_id(preds["location_id"])
    _add("hemisphere", "Northern", lat >= 0)
    _add("hemisphere", "Southern", lat < 0)

    # 6. extreme_target — top quartile of |target|, threshold from this result's own distribution
    abs_target = preds["target"].abs()
    target_threshold = abs_target.quantile(0.75)
    _add("extreme_target", "extreme", abs_target >= target_threshold)
    _add("extreme_target", "typical", abs_target < target_threshold)

    # 7. rapid_change — top quartile of |target - true_tws_t|
    if "true_tws_t" in preds.columns and preds["true_tws_t"].notna().any():
        valid = preds["true_tws_t"].notna()
        delta = (preds["target"] - preds["true_tws_t"]).abs()
        rapid_threshold = delta[valid].quantile(0.75)
        _add("rapid_change", "rapid", valid & (delta >= rapid_threshold))
        _add("rapid_change", "typical", valid & (delta < rapid_threshold))
    else:
        logger.info("decompose(): true_tws_t not available — skipping rapid_change rows.")

    return pd.DataFrame(rows)


def degradation_slope(decomp_df: pd.DataFrame) -> pd.DataFrame:
    """ΔRMSE/Δk per ACF quartile, alongside Experiment 5's AR(1) reference.

    Reads the ``staleness_x_acf_quartile`` rows ``decompose()`` produced
    (so ``decompose`` must have been called with a real ``acf_lookup``) and,
    for each quartile, reconstructs the theoretical degradation curve
    ``sigma * sqrt(2 * (1 - rho**k))`` from
    ``phase1_constants.ACF_QUARTILE_AR1_PARAMS`` — the exact per-quartile
    parameters Experiment 5 measured — as a reference column sitting next to
    the model's own empirical RMSE(k), so a new model's degradation
    behavior is numerically comparable to the mechanistic baseline
    immediately, not just plotted in a vacuum.

    Returns
    -------
    pd.DataFrame
        Columns ``acf_quartile, k, n, empirical_rmse, theoretical_rmse,
        empirical_delta_rmse, theoretical_delta_rmse`` — the last two are
        ``NaN`` at each quartile's first (lowest) k, since a delta needs a
        prior point.
    """
    cross_cut = decomp_df[decomp_df["slice_type"] == "staleness_x_acf_quartile"].copy()
    if cross_cut.empty:
        raise ValueError(
            "degradation_slope() requires staleness_x_acf_quartile rows in "
            "decomp_df — call decompose(tier_result, acf_lookup=...) with a "
            "real acf_lookup first."
        )

    parsed = cross_cut["slice_value"].str.extract(r"^k=(?P<k>\d+)\|(?P<acf_quartile>.+)$")
    cross_cut["k"] = parsed["k"].astype(int)
    cross_cut["acf_quartile"] = parsed["acf_quartile"]
    cross_cut = cross_cut[cross_cut["acf_quartile"] != "unknown_acf"]

    rows: list[dict[str, object]] = []
    for quartile in ACF_QUARTILE_ORDER:
        sub = cross_cut[cross_cut["acf_quartile"] == quartile].sort_values("k")
        if sub.empty:
            continue
        params = ACF_QUARTILE_AR1_PARAMS[quartile]
        prev_empirical: float | None = None
        prev_theoretical: float | None = None
        for _, row in sub.iterrows():
            k = int(row["k"])
            empirical_rmse = float(row["rmse"])
            theoretical_rmse = float(params["sigma"] * np.sqrt(2 * (1 - params["rho"] ** k)))
            rows.append(
                {
                    "acf_quartile": quartile,
                    "k": k,
                    "n": int(row["n"]),
                    "empirical_rmse": empirical_rmse,
                    "theoretical_rmse": theoretical_rmse,
                    "empirical_delta_rmse": (
                        empirical_rmse - prev_empirical if prev_empirical is not None else float("nan")
                    ),
                    "theoretical_delta_rmse": (
                        theoretical_rmse - prev_theoretical if prev_theoretical is not None else float("nan")
                    ),
                }
            )
            prev_empirical = empirical_rmse
            prev_theoretical = theoretical_rmse

    return pd.DataFrame(rows)
