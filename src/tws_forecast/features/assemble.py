"""Feature-assembly pipeline — Project Phase 4 step 4.9.

``build_feature_matrix(df, train_df=None, ...) -> pd.DataFrame`` composes
every step 4.1-4.6 output into one flat, model-ready frame, indexed
identically to ``df``. This is deliberately the *minimum* needed for step
4.9's own proof run (``notebooks/05_state_features.ipynb``'s leakage-shuffle
test, target-transformation comparison, A-014 confirmation, and
feature-importance pass) — the full ``pipelines/train.py``
``docs/ARCHITECTURE.md`` §6 eventually describes is Project Phase 10's job,
not an early start on it here.

**Performance note (added after Project Phase 4 step 4.9's first proof run):**
``build_state_snapshots``/``compute_location_signatures`` are, at real
~15,715-location scale, a genuinely expensive per-location computation —
minutes, not milliseconds, per call. The original version of this function
called them redundantly: once directly for this module's own ``state_*``/
``signature_*`` columns, and *again*, independently, inside
``SpatialHistoryTransformer`` (both) and ``TrailingTrendTransformer`` (once
per configured trend window). None of those calls shared results. This
function now computes each expensive panel **once** and injects it into
every consumer that needs it (``SpatialHistoryTransformer.transform``'s
``state_panel``/``signature_panel`` parameters,
``TrailingTrendTransformer.transform``'s ``precomputed_state_panels``) —
see :func:`_needed_state_windows` for exactly which window(s) still require
a dedicated call (only ``trend_window_months`` entries that don't already
coincide with ``max(trailing_windows)``, the window the direct ``state_*``
step's own ``local_trend`` already uses). Every one of the underlying
Transformers still defaults to computing everything itself when called
directly outside this function — nothing here changes their own public
default behavior, only this module's own redundancy.

This module does no feature engineering of its own. It only calls each
already-leakage-tested piece with ``train_df`` as history and ``df`` as the
query frame, then concatenates their outputs column-wise:

- **Step 4.1** — ``state.reconstruction.build_state_snapshots`` (no
  ``Transformer`` wrapper exists for this one; the same
  ``pd.concat``-then-content-deduplicate-then-select-back-out pattern every
  other module in this project already uses for the identical reason is
  applied locally here, see :func:`_combined_history`).
- **Step 4.2** — ``state.signatures.LocationSignatureTransformer``.
- **Step 4.3** — ``state.spatial_history.SpatialHistoryTransformer``.
- **Step 4.5** — ``features.temporal.TrailingTrendTransformer`` and
  ``MonthHemisphereTransformer``.
- **Step 4.6** — ``features.environmental.SpeiDifferencingTransformer``,
  ``DroughtPersistenceTransformer``, ``SoilMoistureTrajectoryTransformer``.

Every one of those already carries its own step 4.8 leakage proof
(``tests/test_no_leakage_features.py``); this module adds one more, at the
composed level: ``tests/test_assemble.py``'s own future-row-shuffle
regression test, and ``notebooks/05_state_features.ipynb`` section 2's
literal end-to-end run against the real data.

``train_df`` defaults to ``df`` itself when omitted, matching a first CV
fold with no separate training context — every individual Transformer this
project has built already treats "fit on X, transform X" as a legitimate,
tested call shape (e.g. ``validation.leakage_tests.future_row_shuffle_test``
does exactly this). Callers building leakage-safe CV folds must always pass
an explicit ``train_df`` restricted to genuinely already-known rows.

**Precondition on ``train_df``/``df`` indices** (inherited unchanged from
every composed Transformer, not new here): ``train_df`` and ``df`` must not
share index labels that refer to *different* ``(location_id, time)`` pairs.
This holds automatically whenever both are slices of the same master frame
(the standard CV-fold shape ``validation.splitters`` produces — a partition
of one original index, so shared labels never occur) or when
``train_df is df``. It would not hold for two independently constructed
frames that each default to their own ``RangeIndex`` — the same caution
documented on every ``Transformer.transform()`` in this project.
"""

from __future__ import annotations

import pandas as pd

from tws_forecast.features.environmental import (
    DroughtPersistenceTransformer,
    SoilMoistureTrajectoryTransformer,
    SpeiDifferencingTransformer,
)
from tws_forecast.features.registry import load_feature_config
from tws_forecast.features.temporal import MonthHemisphereTransformer, TrailingTrendTransformer
from tws_forecast.state.reconstruction import build_state_snapshots, ensure_location_id
from tws_forecast.state.signatures import compute_location_signatures
from tws_forecast.state.spatial_history import SpatialHistoryTransformer

__all__ = ["build_feature_matrix", "RAW_PASSTHROUGH_COLUMNS"]

#: Raw Train.csv/Test.csv columns carried through unchanged -- the model's
#: only view of "now" beyond what steps 4.1-4.6 derive from history.
#: ``TWS_t`` is deliberately excluded: it is masked on two-thirds of real
#: test rows, so a single model can't use it as a uniform raw feature (the
#: same reasoning ``models.baselines.RidgeBaselinePredictor`` documents) --
#: ``state_last_known_tws`` (step 4.1, which already resolves the
#: observed/masked distinction via ``StateSnapshot``) is the leakage-safe
#: substitute this pipeline hands a downstream model instead.
RAW_PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "month_sin",
    "month_cos",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
)

# StateSnapshot/LocationSignature columns dropped before assembly: pure
# join keys already present via df's own location_id/time (location_id,
# as_of), a field that's structurally always None as of Project Phase 4
# (location_signature -- StateSnapshot's own signature cross-reference is
# not populated until a future phase wires it up), and a raw Timestamp
# that isn't itself model-ready (months_since_observation is its already
# leakage-safe, numeric equivalent).
_STATE_SNAPSHOT_DROP_COLUMNS = ("location_id", "as_of", "last_known_time", "location_signature")
_SIGNATURE_DROP_COLUMNS = ("location_id", "as_of")


def _combined_history(train_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df, df], ignore_index=False)
    combined = ensure_location_id(combined)
    # Content-based, never index-based -- see the module docstring's
    # precondition note and every composed Transformer's identical fix.
    return combined.loc[~combined.duplicated(subset=["location_id", "time"], keep="last")]


def _resolve_trend_window_months(trend_window_months: tuple[int, ...] | None) -> tuple[int, ...]:
    if trend_window_months is not None:
        return trend_window_months
    return load_feature_config("temporal").trend_window_months


def build_feature_matrix(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    *,
    include_state_snapshot: bool = True,
    include_signatures: bool = True,
    include_spatial_history: bool = True,
    include_temporal: bool = True,
    include_environmental: bool = True,
    trailing_windows: tuple[int, ...] = (12, 24),
    shrinkage_k: int | None = None,
    trend_window_months: tuple[int, ...] | None = None,
    spatial_history_kwargs: dict | None = None,
    environmental_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Build the full, flat, model-ready feature matrix for ``df``, indexed
    identically to it.

    Parameters
    ----------
    df:
        The rows to build features for (a CV fold's validation/test
        portion, or an entire ``Train.csv``/``Test.csv``-shaped frame).
    train_df:
        Historical context every composed step passes to its own ``fit()``.
        Defaults to ``df`` itself -- see the module docstring.
    include_state_snapshot, include_signatures, include_spatial_history,
    include_temporal, include_environmental:
        Toggle each Project Phase 4 sub-step's contribution independently
        -- used by this module's own leakage/ablation tests, and available
        to the step 4.9 notebook for isolating one step's contribution to
        feature importance without rebuilding the whole matrix by hand.
    trailing_windows, shrinkage_k, trend_window_months,
    spatial_history_kwargs, environmental_kwargs:
        Forwarded to the underlying step 4.1/4.2/4.3/4.5/4.6 calls; omitted
        values fall back to each step's own config-driven defaults
        (``configs/features/*.yaml`` via ``features.registry``).

    Returns
    -------
    pd.DataFrame
        One row per row of ``df``, columns namespaced by source step
        (``state_*`` for step 4.1, ``signature_*`` for step 4.2,
        ``neighbor_*`` for step 4.3, ``trend_slope_*``/
        ``month_hemisphere_*`` for step 4.5, ``spei_*_diff_*``/
        ``drought_persistence_*``/``soil_moisture_*`` for step 4.6) plus
        the raw passthrough columns -- no two steps' output columns
        collide by construction (verified in ``tests/test_assemble.py``).
    """
    train_df = df if train_df is None else train_df
    frame = ensure_location_id(df).copy()
    frame["time"] = pd.to_datetime(frame["time"])

    present_raw_columns = [c for c in RAW_PASSTHROUGH_COLUMNS if c in frame.columns]
    pieces: list[pd.DataFrame] = [frame[present_raw_columns]]

    # Every step below that needs a per-location panel draws from the SAME
    # `combined` and the SAME state_panel/signature_panel, computed at most
    # once each -- see the module docstring's performance note. `combined`
    # itself (concat + content-dedup) is cheap; `state_panel`/
    # `signature_panel` are the expensive per-location calls this sharing
    # exists to avoid paying more than once for.
    needs_combined = (
        include_state_snapshot or include_signatures or include_spatial_history or include_temporal
    )
    combined = _combined_history(train_df, df) if needs_combined else None

    needs_state_panel = include_state_snapshot or include_spatial_history or include_temporal
    state_panel = None
    if needs_state_panel:
        state_panel = build_state_snapshots(
            combined, as_of_column="time", trailing_windows=trailing_windows
        )

    needs_signature_panel = include_signatures or include_spatial_history
    signature_panel = None
    if needs_signature_panel:
        signature_panel = compute_location_signatures(
            combined, as_of_column="time", shrinkage_k=shrinkage_k
        )

    if include_state_snapshot:
        keep_columns = [c for c in state_panel.columns if c not in _STATE_SNAPSHOT_DROP_COLUMNS]
        pieces.append(state_panel.loc[df.index, keep_columns].add_prefix("state_"))

    if include_signatures:
        keep_columns = [c for c in signature_panel.columns if c not in _SIGNATURE_DROP_COLUMNS]
        pieces.append(signature_panel.loc[df.index, keep_columns].add_prefix("signature_"))

    if include_spatial_history:
        spatial_kwargs = spatial_history_kwargs or {}
        spatial_kwargs.setdefault("shrinkage_k", shrinkage_k)
        spatial_transformer = SpatialHistoryTransformer(**spatial_kwargs)
        spatial_transformer.fit(train_df)
        pieces.append(
            spatial_transformer.transform(
                df, state_panel=state_panel, signature_panel=signature_panel
            )
        )

    if include_temporal:
        resolved_trend_windows = _resolve_trend_window_months(trend_window_months)
        # state_panel's own local_trend is computed over window_months=
        # max(trailing_windows) (state.reconstruction._compute_local_trend) --
        # reusable directly for whichever configured trend window happens to
        # equal that same value, with zero extra build_state_snapshots calls.
        # Any other configured window still needs its own dedicated call,
        # made inside TrailingTrendTransformer.transform itself.
        precomputed_state_panels = {}
        if (
            state_panel is not None
            and trailing_windows
            and max(trailing_windows) in resolved_trend_windows
        ):
            precomputed_state_panels[max(trailing_windows)] = state_panel

        trend_transformer = TrailingTrendTransformer(trend_window_months=trend_window_months)
        trend_transformer.fit(train_df)
        pieces.append(
            trend_transformer.transform(df, precomputed_state_panels=precomputed_state_panels)
        )

        hemisphere_transformer = MonthHemisphereTransformer()
        hemisphere_transformer.fit(train_df)
        pieces.append(hemisphere_transformer.transform(df))

    if include_environmental:
        env_kwargs = environmental_kwargs or {}

        spei_transformer = SpeiDifferencingTransformer(
            spei_diff_lags=env_kwargs.get("spei_diff_lags")
        )
        spei_transformer.fit(train_df)
        pieces.append(spei_transformer.transform(df))

        drought_transformer = DroughtPersistenceTransformer(
            drought_threshold=env_kwargs.get("drought_threshold")
        )
        drought_transformer.fit(train_df)
        pieces.append(drought_transformer.transform(df))

        soil_transformer = SoilMoistureTrajectoryTransformer(
            trend_window_months=env_kwargs.get("soil_trend_window_months", (12,))
        )
        soil_transformer.fit(train_df)
        pieces.append(soil_transformer.transform(df))

    matrix = pd.concat(pieces, axis=1)
    matrix.index = df.index
    return matrix
