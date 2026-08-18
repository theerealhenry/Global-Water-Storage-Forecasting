"""The leakage firewall, as executable checks — not only a documented promise.

``docs/ARCHITECTURE.md`` §7 names four mechanical checks: "a future-row
shuffle test confirms that shuffling data after the forecast origin does
not change a prediction made before it; a historical-only check confirms
signatures and climatology features never draw on rows at or after their
forecast origin; rolling-window features are verified to stop exactly at
the information cutoff; and the masking simulator is checked to confirm it
cannot leak a value it has just hidden." This module implements exactly
those four, as reusable, importable functions — ``tests/test_leakage_
firewall.py`` is a thin pytest wrapper around them, not where the checking
logic itself lives (so a future pipeline module, Project Phase 5+, can call
these checks directly too, not only through pytest).

Three of the four checks (``future_row_shuffle_test``, ``historical_only_
check``, ``rolling_window_cutoff_check``) are written generically, against
any callable matching the right shape, because Project Phase 4 (state
reconstruction, signatures, lag features) hasn't been built yet — there is
no real signature function or feature function to check today. Each is
exercised in tests against both a correct toy example (must pass) and a
deliberately leaky one (must be caught), so the check itself is proven to
have teeth before Phase 4 ever calls it for real. The fourth
(``masking_simulator_no_leak_check``) is exercised directly against the
real ``apply_masking``, since that function already exists.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from tws_forecast.utils.seeds import RANDOM_SEED, set_seed
from tws_forecast.validation.masking_simulator import MaskingScenario, apply_masking
from tws_forecast.validation.tiers import Predictor

logger = logging.getLogger(__name__)

__all__ = [
    "future_row_shuffle_test",
    "historical_only_check",
    "rolling_window_cutoff_check",
    "masking_simulator_no_leak_check",
    "FeatureNameViolation",
    "DISALLOWED_FEATURE_NAME_PATTERNS",
    "scan_features_module_for_disallowed_names",
]


def _values_close(a: Any, b: Any, atol: float = 1e-9) -> bool:
    """NaN-safe, array-or-scalar-safe equality for check results."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return False
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    close = np.isclose(a_arr, b_arr, atol=atol, equal_nan=False)
    return bool(np.all(close | both_nan))


def future_row_shuffle_test(
    model: Predictor,
    df: pd.DataFrame,
    cutoff_time: pd.Timestamp | str,
    seed: int = RANDOM_SEED,
) -> bool:
    """Confirms shuffling every row's *position* after ``cutoff_time`` does
    not change predictions made from data at or before it.

    The check fits and predicts once on the unmodified ``df``, then again
    after physically reordering (row-content swap, not just an index
    permutation) every row with ``time > cutoff_time`` among themselves —
    leaving each such row's own content intact, just relocated within the
    frame — and confirms the at-or-before-cutoff fit/predict is unaffected.

    For code that correctly selects rows by the ``time`` column's value
    (boolean masking, as every function in ``validation/splitters.py`` and
    ``validation/tiers.py`` does), this is provably always true — which is
    exactly the point: it's a standing regression guard against a future
    refactor accidentally introducing positional/``.iloc``-based row
    selection instead, a realistic and easy mistake to make silently.

    Returns
    -------
    bool
        ``True`` if predictions before/at ``cutoff_time`` are unchanged by
        the future-row shuffle (no leak detected), ``False`` otherwise.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    cutoff_time = pd.Timestamp(cutoff_time)

    history = df[df["time"] <= cutoff_time]
    if len(history) == 0:
        raise ValueError(f"future_row_shuffle_test: no rows at or before cutoff_time={cutoff_time}")

    model.fit(history)
    baseline_preds = np.asarray(model.predict(history))

    future_mask = df["time"] > cutoff_time
    shuffled_df = df.copy()
    if future_mask.any():
        shuffled_df.loc[future_mask, :] = (
            df.loc[future_mask, :].sample(frac=1.0, random_state=seed).to_numpy()
        )

    shuffled_history = shuffled_df[shuffled_df["time"] <= cutoff_time]
    model.fit(shuffled_history)
    shuffled_preds = np.asarray(model.predict(shuffled_history))

    ok = _values_close(baseline_preds, shuffled_preds)
    if not ok:
        logger.warning(
            "future_row_shuffle_test: FAILED — predictions changed after shuffling future rows"
        )
    return ok


def historical_only_check(
    signature_fn: Callable[[pd.DataFrame, pd.Timestamp], Any],
    df: pd.DataFrame,
    evaluate_time: pd.Timestamp | str,
) -> bool:
    """Confirms ``signature_fn(df, t)`` — a location-signature or
    climatology-style function — depends only on rows with
    ``time < evaluate_time``, per ``docs/ARCHITECTURE.md`` §4's
    origin-time-indexed-signature invariant.

    Calls ``signature_fn`` twice: once on the full ``df``, once on ``df``
    with every row at or after ``evaluate_time`` removed *before* the call.
    A correct signature function computes the same result either way,
    because it does its own internal ``time < evaluate_time`` filtering and
    so never actually needed the future rows in the first place. A function
    that instead trusts whatever frame it's handed (i.e., ignores
    ``evaluate_time`` and just aggregates the full input) gives a different
    answer on the two calls, and is caught.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    evaluate_time = pd.Timestamp(evaluate_time)

    full_result = signature_fn(df, evaluate_time)
    truncated = df[df["time"] < evaluate_time]
    truncated_result = signature_fn(truncated, evaluate_time)

    ok = _values_close(full_result, truncated_result)
    if not ok:
        logger.warning(
            "historical_only_check: FAILED — %r depends on rows at/after evaluate_time=%s",
            getattr(signature_fn, "__name__", signature_fn),
            evaluate_time,
        )
    return ok


def rolling_window_cutoff_check(
    feature_fn: Callable[[pd.DataFrame, pd.Timestamp], Any],
    df: pd.DataFrame,
    origin_time: pd.Timestamp | str,
    seed: int = RANDOM_SEED,
    perturb_column: str = "TWS_t",
    include_origin_row: bool = False,
) -> bool:
    """Confirms ``feature_fn(df, t)`` — a rolling/lag-style feature — never
    reflects any row with ``time >= origin_time`` (the default, and Phase
    2's original behavior), or, when ``include_origin_row=True``, never
    reflects any row with ``time > origin_time``.

    Computes the feature once normally, then again after replacing
    ``perturb_column`` (``TWS_t`` by default; Project Phase 4 step 4.8 also
    passes ``SPEI_XX_t``/``SOIL_MOISTURE_t`` to check
    ``features/environmental.py``) with an enormous, unmistakable
    perturbation for every row in the future window — a feature that
    genuinely stops at the information cutoff is numerically unaffected; a
    feature with an off-by-one boundary error picks up the perturbation and
    is caught.

    ``include_origin_row`` exists because Project Phase 4 established two
    equally-legitimate, deliberately different origin-boundary conventions
    (``docs/ARCHITECTURE.md`` §4): ``StateSnapshot`` and everything built on
    it (``features/temporal.py``, ``features/environmental.py``, and the S2
    columns of ``state/spatial_history.py``) use ``time <= as_of`` — the
    origin row's own value counts as "what we know now" when it happens to
    be observed — while ``LocationSignature`` and its S3-tagged consumers
    use the strictly-earlier ``time < as_of`` to avoid circularity as an
    anomaly baseline. ``include_origin_row=True`` narrows the perturbed
    ("future") window to ``time > origin_time``, leaving the origin row's
    own content untouched, so this check is meaningful for the inclusive
    family too — a leak from strictly *after* ``origin_time`` is never
    legitimate under either convention, unlike a dependency on the origin
    row's own value, which the exclusive default (``include_origin_row=
    False``) would otherwise, incorrectly, also flag.
    """
    set_seed(seed)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    origin_time = pd.Timestamp(origin_time)

    baseline = feature_fn(df, origin_time)

    perturbed = df.copy()
    future_mask = (
        perturbed["time"] > origin_time if include_origin_row else perturbed["time"] >= origin_time
    )
    if future_mask.any() and perturb_column in perturbed.columns:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1e6, size=int(future_mask.sum()))
        perturbed.loc[future_mask, perturb_column] = (
            perturbed.loc[future_mask, perturb_column].fillna(0.0).to_numpy() + noise
        )

    perturbed_result = feature_fn(perturbed, origin_time)

    ok = _values_close(baseline, perturbed_result)
    if not ok:
        logger.warning(
            "rolling_window_cutoff_check: FAILED — %r reflects data at/after origin_time=%s "
            "(perturb_column=%r, include_origin_row=%s)",
            getattr(feature_fn, "__name__", feature_fn),
            origin_time,
            perturb_column,
            include_origin_row,
        )
    return ok


def masking_simulator_no_leak_check(
    scenario: MaskingScenario,
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    derived_columns: list[str] | None = None,
) -> bool:
    """Confirms ``apply_masking`` never leaves the true, pre-masking
    ``TWS_t`` value recoverable from anywhere else in a masked row.

    Three sub-checks on every row ``apply_masking`` marks as masked:
    ``TWS_t`` itself is null; every column named in ``derived_columns`` is
    also null (a caller that passes derived columns but the simulator
    silently leaves one populated would otherwise leak the reconstructible
    state right back in); and no other numeric column in that row exactly
    equals the true, pre-masking ``TWS_t`` value (would indicate the value
    was copied elsewhere instead of hidden).
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    masked = apply_masking(df, scenario, seed=seed, derived_columns=derived_columns)
    masked_rows = masked[masked["TWS_t_masked"]]
    if len(masked_rows) == 0:
        logger.info(
            "masking_simulator_no_leak_check: scenario masked 0 rows on this df — nothing to check"
        )
        return True

    if masked_rows["TWS_t"].notna().any():
        logger.warning("masking_simulator_no_leak_check: FAILED — a masked row's TWS_t is not null")
        return False

    for col in derived_columns or []:
        if col in masked_rows.columns and masked_rows[col].notna().any():
            logger.warning(
                "masking_simulator_no_leak_check: FAILED — derived column %r left populated on a masked row",
                col,
            )
            return False

    true_tws = df.loc[masked_rows.index, "TWS_t"]
    skip_columns = {"TWS_t", *(derived_columns or [])}
    for col in masked_rows.columns:
        if col in skip_columns or not pd.api.types.is_numeric_dtype(masked_rows[col]):
            continue
        if (masked_rows[col].to_numpy() == true_tws.to_numpy()).any():
            logger.warning(
                "masking_simulator_no_leak_check: FAILED — column %r reproduces the true masked TWS_t value",
                col,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Disallowed-feature-name static scan.
#
# docs/ARCHITECTURE.md §7's leakage-firewall table forbids "Exploitation of
# the public/private leaderboard split" via any feature "derived from a
# row's position or index within the test file." No feature module exists
# yet (Project Phase 4) to runtime-check the output columns of, so this is
# a static source-text scan for now — a standing guard, written once, that
# stays green vacuously until Phase 4 adds real feature modules, per
# docs/PHASE2_EXECUTION_PLAN.md step 2.8. Once real feature functions exist,
# a stronger runtime check (import the module, call its functions, inspect
# actual output DataFrame column names) can be added alongside this one —
# this scan is not meant to be the only such check forever, just the one
# that can exist starting now.
# ---------------------------------------------------------------------------

DISALLOWED_FEATURE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"test_row_index", re.IGNORECASE),
    re.compile(r"relative_test_position", re.IGNORECASE),
    re.compile(r"\brow_order\b", re.IGNORECASE),
    re.compile(r"file_position", re.IGNORECASE),
    re.compile(r"\brow_index\b", re.IGNORECASE),
)


class FeatureNameViolation(NamedTuple):
    file: Path
    pattern: str
    line_number: int
    line: str


def scan_features_module_for_disallowed_names(
    features_dir: Path,
) -> list[FeatureNameViolation]:
    """Scan every ``.py`` file under ``features_dir`` for identifiers
    matching ``DISALLOWED_FEATURE_NAME_PATTERNS``.

    Returns an empty list if ``features_dir`` doesn't exist or contains no
    matches — both are the expected, correct state before Project Phase 4
    adds real feature modules; this function is written now specifically so
    nobody has to remember to add the check later.
    """
    violations: list[FeatureNameViolation] = []
    if not features_dir.exists():
        return violations

    for py_file in sorted(features_dir.rglob("*.py")):
        text = py_file.read_text()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in DISALLOWED_FEATURE_NAME_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        FeatureNameViolation(py_file, pattern.pattern, line_number, line.strip())
                    )

    return violations
