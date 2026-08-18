"""Standing guard: no feature module may derive a column from row
position/index within the file (docs/ARCHITECTURE.md §7's leakage-firewall
table), and no real feature this project has built may leak data from at or
after its own forecast origin (docs/ARCHITECTURE.md §4/§7).

Project Phase 2 (docs/PHASE2_EXECUTION_PLAN.md step 2.8) wrote this file as
a deliberately vacuous pass — no feature modules existed yet. Project Phase
4 step 4.8 is where it stops being vacuous: every real ``Transformer`` built
in steps 4.2-4.6 (``state/signatures.py``, ``state/spatial_history.py``,
``features/temporal.py``, ``features/environmental.py``) is now run through
all four of ``validation.leakage_tests``'s generic checks
(``future_row_shuffle_test``, ``historical_only_check``,
``rolling_window_cutoff_check``, ``masking_simulator_no_leak_check``), plus
the disallowed-feature-name scan against this phase's actual output column
names.

Two deliberately different, both-legitimate origin-boundary conventions
exist across these modules (see each module's own docstring, and
``validation.leakage_tests.rolling_window_cutoff_check``'s docstring for the
``include_origin_row`` extension step 4.8 added to accommodate this):

- **Strict** (``time < as_of``): ``state/signatures.py``'s
  ``LocationSignatureTransformer`` and ``state/spatial_history.py``'s S3
  (signature-derived) columns. A leak here is *any* dependence on the
  origin row's own content, not only rows strictly after it — checked with
  ``historical_only_check`` (a true "same or fewer rows -> same value" test)
  and the exclusive-default ``rolling_window_cutoff_check``.
- **Inclusive** (``time <= as_of``): ``features/temporal.py``'s
  ``TrailingTrendTransformer``, every ``features/environmental.py``
  transformer, and ``state/spatial_history.py``'s S2 (historical) columns.
  The origin row's own value legitimately counts as "what's known now" when
  observed — only rows *strictly after* the origin may never leak in,
  checked with ``rolling_window_cutoff_check(..., include_origin_row=True)``.

``MonthHemisphereTransformer`` is stateless (no ``time``/history dependence
at all) and is checked as a trivial-pass case under every check, which is
itself the correct, intended behavior — not a weaker test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import get_repo_root, load_train
from tws_forecast.features.environmental import (
    DroughtPersistenceTransformer,
    SoilMoistureTrajectoryTransformer,
    SpeiDifferencingTransformer,
)
from tws_forecast.features.temporal import MonthHemisphereTransformer, TrailingTrendTransformer
from tws_forecast.state.signatures import LocationSignatureTransformer
from tws_forecast.state.spatial_history import SPATIAL_FEATURE_TAXONOMY, SpatialHistoryTransformer
from tws_forecast.utils.seeds import RANDOM_SEED
from tws_forecast.validation.leakage_tests import (
    future_row_shuffle_test,
    historical_only_check,
    masking_simulator_no_leak_check,
    rolling_window_cutoff_check,
    scan_features_module_for_disallowed_names,
)
from tws_forecast.validation.masking_simulator import MaskingScenario

# ---------------------------------------------------------------------------
# Disallowed-feature-name static scan — now exercised against this phase's
# actual output column names for the first time (it was vacuous in Phase 2).
# ---------------------------------------------------------------------------


def test_current_features_module_has_no_disallowed_names() -> None:
    features_dir = get_repo_root() / "src" / "tws_forecast" / "features"
    violations = scan_features_module_for_disallowed_names(features_dir)
    assert violations == [], f"Disallowed row-position-derived feature name(s) found: {violations}"


def test_current_state_module_has_no_disallowed_names() -> None:
    # state/signatures.py and state/spatial_history.py are also real,
    # column-producing Phase 4 modules -- scan them too, not only
    # features/.
    state_dir = get_repo_root() / "src" / "tws_forecast" / "state"
    violations = scan_features_module_for_disallowed_names(state_dir)
    assert violations == [], f"Disallowed row-position-derived feature name(s) found: {violations}"


def test_scan_returns_empty_for_nonexistent_directory(tmp_path: Path) -> None:
    violations = scan_features_module_for_disallowed_names(tmp_path / "does_not_exist")
    assert violations == []


def test_scan_returns_empty_for_directory_with_no_python_files(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("nothing to see here: row_index")
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert violations == []


def test_scan_catches_disallowed_pattern_in_a_python_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_features.py"
    bad_file.write_text(
        "def compute(df):\n" "    df['test_row_index'] = range(len(df))\n" "    return df\n"
    )
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert len(violations) == 1
    assert violations[0].file == bad_file
    assert violations[0].line_number == 2
    assert "test_row_index" in violations[0].line


def test_scan_catches_multiple_disallowed_patterns() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        f1 = tmp_path / "a.py"
        f1.write_text("relative_test_position = 1\n")
        f2 = tmp_path / "b.py"
        f2.write_text("file_position = 2\nrow_order = 3\n")

        violations = scan_features_module_for_disallowed_names(tmp_path)
        matched_files = {v.file.name for v in violations}
        assert matched_files == {"a.py", "b.py"}
        assert len(violations) == 3  # relative_test_position, file_position, row_order


def test_scan_does_not_false_positive_on_unrelated_names(tmp_path: Path) -> None:
    ok_file = tmp_path / "ok_features.py"
    ok_file.write_text(
        "def compute_lag(df, origin_time):\n"
        "    # a totally normal, leakage-safe rolling feature\n"
        "    sub = df[df['time'] < origin_time]\n"
        "    return sub['TWS_t'].iloc[-1] if len(sub) else float('nan')\n"
    )
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert violations == []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


def _single_location_frame(n_months: int = 30, start: str = "2003-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(11)
    times = [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(n_months)]
    trend = 1.0 + 0.05 * np.arange(n_months)
    return pd.DataFrame(
        {
            "time": times,
            "lat": 9.5,
            "lon": 21.5,
            "location_id": "9.5_21.5",
            "TWS_t": trend + rng.normal(0.0, 0.2, size=n_months),
            "SPEI_01_t": rng.normal(0.0, 1.0, size=n_months),
            "SPEI_03_t": rng.normal(0.0, 1.0, size=n_months),
            "SPEI_06_t": rng.normal(0.0, 1.0, size=n_months),
            "SPEI_12_t": rng.normal(0.0, 1.0, size=n_months),
            "SOIL_MOISTURE_t": 5.0
            + 0.02 * np.arange(n_months)
            + rng.normal(0.0, 0.1, size=n_months),
        }
    )


def _multi_location_frame(
    n_locations: int = 6, n_months: int = 24, start: str = "2003-01-01"
) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    times = [pd.Timestamp(start) + pd.DateOffset(months=i) for i in range(n_months)]
    frames = []
    for loc_idx in range(n_locations):
        lat, lon = 10.0 + loc_idx * 0.5, 20.0 + loc_idx * 0.5
        trend = 1.0 + 0.03 * np.arange(n_months) + loc_idx
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "lat": lat,
                    "lon": lon,
                    "location_id": f"{lat}_{lon}",
                    "TWS_t": trend + rng.normal(0.0, 0.2, size=n_months),
                    "SPEI_12_t": rng.normal(0.0, 1.0, size=n_months),
                    "SOIL_MOISTURE_t": 5.0 + rng.normal(0.0, 0.1, size=n_months),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Generic query-time adapters -- turn a fit()/transform() Transformer into
# the two-argument Callable[[df, t], array] shape historical_only_check and
# rolling_window_cutoff_check expect, and into the fit()/predict()
# Predictor shape future_row_shuffle_test expects.
# ---------------------------------------------------------------------------


def _query_time_adapter(transformer_factory, columns: list[str] | None = None):
    """``fn(df, t)``: fit a fresh transformer on ``df``, transform a single
    query row at time ``t`` (the row already in ``df`` at that time if one
    exists, else a synthetic row cloning the first row's location), and
    return the (optionally column-subset) numeric result as a flat array."""

    def _fn(df: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
        transformer = transformer_factory()
        transformer.fit(df)

        t = pd.Timestamp(t)
        times = pd.to_datetime(df["time"])
        query = df.loc[times == t]
        if query.empty:
            query = df.iloc[[0]].copy()
            query["time"] = t
            # A guaranteed-unused index label -- reusing df.iloc[0]'s own
            # index would collide with that very row still present in df
            # (or a truncated copy of it) inside the transformer's own
            # pd.concat, making `.loc[df.index]` ambiguous (two rows
            # sharing one index label) rather than leakage-testing anything.
            query.index = pd.Index([-1] * len(query))

        result = transformer.transform(query)
        if columns is not None:
            result = result[columns]
        numeric = result.select_dtypes(include=[np.number])
        return numeric.to_numpy(dtype=float).ravel()

    return _fn


class _TransformerAsPredictor:
    """Adapts any ``features.base.Transformer`` to the ``fit``/``predict``
    shape ``future_row_shuffle_test`` expects, per
    ``docs/PHASE4_EXECUTION_PLAN.md`` §4.8: ``predict`` is the transformer's
    own ``transform`` output, collapsed to one numeric value per row so it
    can stand in for a prediction."""

    def __init__(self, transformer_factory) -> None:
        self._transformer_factory = transformer_factory

    def fit(self, train_df: pd.DataFrame) -> None:
        self._transformer = self._transformer_factory()
        self._transformer.fit(train_df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        result = self._transformer.transform(df)
        numeric = result.select_dtypes(include=[np.number]).fillna(0.0)
        return numeric.to_numpy(dtype=float).sum(axis=1)


# Every real Transformer this phase built (steps 4.2-4.6), by factory.
REAL_TRANSFORMER_FACTORIES = {
    "LocationSignatureTransformer": LocationSignatureTransformer,
    "SpatialHistoryTransformer": SpatialHistoryTransformer,
    "TrailingTrendTransformer": TrailingTrendTransformer,
    "MonthHemisphereTransformer": MonthHemisphereTransformer,
    "SpeiDifferencingTransformer": SpeiDifferencingTransformer,
    "DroughtPersistenceTransformer": DroughtPersistenceTransformer,
    "SoilMoistureTrajectoryTransformer": SoilMoistureTrajectoryTransformer,
}


def _fixture_for(name: str) -> pd.DataFrame:
    return (
        _multi_location_frame() if name == "SpatialHistoryTransformer" else _single_location_frame()
    )


# ---------------------------------------------------------------------------
# Check 1/4 -- future_row_shuffle_test: universal, applies identically to
# every real Transformer regardless of its origin-boundary convention (it
# tests the harness's own time-based row selection, not the boundary
# semantics themselves).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", REAL_TRANSFORMER_FACTORIES.items())
def test_future_row_shuffle_passes_for_every_real_transformer(name: str, factory) -> None:
    df = _fixture_for(name)
    cutoff = df["time"].sort_values().iloc[len(df["time"].unique()) // 2]
    predictor = _TransformerAsPredictor(factory)
    assert future_row_shuffle_test(predictor, df, cutoff_time=cutoff) is True, name


# ---------------------------------------------------------------------------
# Check 2/4 -- rolling_window_cutoff_check, strictly-future window
# (include_origin_row=True): universal, must hold for every real
# Transformer under either boundary convention -- nothing may ever depend
# on a row strictly after the query's own origin time.
# ---------------------------------------------------------------------------

# (name, perturb_column, output_columns_or_None)
STRICTLY_FUTURE_CASES = [
    ("LocationSignatureTransformer", LocationSignatureTransformer, "TWS_t", None),
    (
        "SpatialHistoryTransformer_S2",
        SpatialHistoryTransformer,
        "TWS_t",
        [c for c, cat in SPATIAL_FEATURE_TAXONOMY.items() if cat == "S2"],
    ),
    (
        "SpatialHistoryTransformer_S3",
        SpatialHistoryTransformer,
        "TWS_t",
        [c for c, cat in SPATIAL_FEATURE_TAXONOMY.items() if cat == "S3"],
    ),
    ("TrailingTrendTransformer", TrailingTrendTransformer, "TWS_t", None),
    ("MonthHemisphereTransformer", MonthHemisphereTransformer, "TWS_t", None),
    ("SpeiDifferencingTransformer", SpeiDifferencingTransformer, "SPEI_03_t", None),
    ("DroughtPersistenceTransformer", DroughtPersistenceTransformer, "SPEI_12_t", None),
    (
        "SoilMoistureTrajectoryTransformer",
        SoilMoistureTrajectoryTransformer,
        "SOIL_MOISTURE_t",
        None,
    ),
]


@pytest.mark.parametrize("name,factory,perturb_column,columns", STRICTLY_FUTURE_CASES)
def test_rolling_window_cutoff_check_no_leak_from_strictly_future_rows(
    name: str, factory, perturb_column: str, columns: list[str] | None
) -> None:
    df = _fixture_for(name.split("_S")[0]) if "SpatialHistory" in name else _fixture_for(name)
    origin = df["time"].sort_values().unique()[len(df["time"].unique()) // 2]
    fn = _query_time_adapter(factory, columns=columns)
    assert (
        rolling_window_cutoff_check(
            fn, df, origin_time=origin, perturb_column=perturb_column, include_origin_row=True
        )
        is True
    ), name


# ---------------------------------------------------------------------------
# Check 3/4a -- historical_only_check (strict time<as_of family): the
# origin row's own value must never matter, not only strictly-future rows.
# LocationSignatureTransformer, and SpatialHistoryTransformer's S3
# (signature-derived) columns.
# ---------------------------------------------------------------------------


def test_historical_only_check_location_signature_transformer() -> None:
    df = _single_location_frame()
    origin = df["time"].sort_values().iloc[15]
    fn = _query_time_adapter(LocationSignatureTransformer)
    assert historical_only_check(fn, df, evaluate_time=origin) is True


# NOTE: historical_only_check itself is deliberately NOT run against
# SpatialHistoryTransformer's S3 columns, unlike LocationSignatureTransformer
# above. historical_only_check's mechanism is *row deletion* (it compares
# against `df[df.time < evaluate_time]`), but SpatialHistoryTransformer's
# neighbor join is keyed on (neighbor_location_id, period) -- it needs a
# neighbor's row to *exist* at the query's own period to look anything up at
# all, even though the *value* it looks up (a signature computed strictly
# before that period) is already correctly boundary-safe. Deleting every row
# at/after evaluate_time -- including OTHER locations' legitimate,
# already-elapsed-relative-to-nothing rows at that exact period -- starves
# every neighbor lookup and forces the (equally legitimate) zero-neighbor
# fallback path, which looks like a difference but isn't a leak: it's an
# artifact of a synthetic fixture, not of production data, where a row
# (masked or not) exists for virtually every (location, month) pair
# (`docs/DATA_DICTIONARY.md`). rolling_window_cutoff_check's exclusive
# default (below) is the check that actually fits this shape -- it perturbs
# *values*, not row existence, so the join still succeeds and correctly
# ignores the perturbed content. The precise, real bug this general class of
# check exists to catch was already caught directly by
# tests/test_spatial_history.py::test_neighbor_feature_does_not_use_neighbor_row_after_query_period
# during step 4.3's own development (the index-vs-content dedup bug).


def test_historical_only_check_month_hemisphere_transformer_trivially_passes() -> None:
    # Stateless -- no history dependence at all, so truncating history
    # changes nothing. This is the correct, intended behavior, not a
    # weaker test.
    df = _single_location_frame()
    origin = df["time"].sort_values().iloc[15]
    fn = _query_time_adapter(MonthHemisphereTransformer)
    assert historical_only_check(fn, df, evaluate_time=origin) is True


# ---------------------------------------------------------------------------
# Check 3/4b -- rolling_window_cutoff_check, exclusive default
# (include_origin_row=False): the same strict-boundary confirmation as
# historical_only_check, from the "perturb, don't truncate" angle, for the
# same strict-boundary transformers.
# ---------------------------------------------------------------------------


def test_rolling_window_cutoff_check_exclusive_location_signature_transformer() -> None:
    df = _single_location_frame()
    origin = df["time"].sort_values().iloc[15]
    fn = _query_time_adapter(LocationSignatureTransformer)
    assert rolling_window_cutoff_check(fn, df, origin_time=origin, perturb_column="TWS_t") is True


def test_rolling_window_cutoff_check_exclusive_spatial_history_s3_columns() -> None:
    df = _multi_location_frame()
    origin = df["time"].sort_values().unique()[10]
    s3_columns = [c for c, cat in SPATIAL_FEATURE_TAXONOMY.items() if cat == "S3"]
    fn = _query_time_adapter(SpatialHistoryTransformer, columns=s3_columns)
    assert rolling_window_cutoff_check(fn, df, origin_time=origin, perturb_column="TWS_t") is True


# ---------------------------------------------------------------------------
# Check 4/4 -- masking_simulator_no_leak_check: an end-to-end integration
# check on real data. Each transformer is fit and transformed on data that
# has ALREADY been through apply_masking (the only way it would ever see
# data in production), so its derived columns are computed with the true
# values genuinely hidden from it -- then those columns are checked against
# the true, pre-masking values to confirm none of them exactly reproduce
# what was just hidden.
# ---------------------------------------------------------------------------


def _abrupt_scenario(start: str, end: str) -> MaskingScenario:
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    streak = (end_d.year * 12 + end_d.month) - (start_d.year * 12 + start_d.month) + 1
    return MaskingScenario(
        blackout_start=start_d,
        blackout_end=end_d,
        streak_length=streak,
        source_rationale="Phase 4 step 4.8 leakage integration check",
    )


def _masking_augmented_frame(
    transformer_factory, df: pd.DataFrame, scenario: MaskingScenario
) -> pd.DataFrame:
    from tws_forecast.validation.masking_simulator import apply_masking

    masked_for_features = apply_masking(df, scenario, seed=RANDOM_SEED)
    transformer = transformer_factory()
    transformer.fit(masked_for_features)
    features = transformer.transform(masked_for_features)
    # augmented keeps df's TRUE (pre-masking) TWS_t for the check's own
    # ground-truth comparison, while carrying feature columns computed from
    # data the transformer only ever saw in its already-masked form -- the
    # realistic production shape.
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


@pytest.mark.parametrize("name,factory", REAL_TRANSFORMER_FACTORIES.items())
def test_masking_simulator_no_leak_check_for_every_real_transformer(
    name: str, factory, real_train_df: pd.DataFrame
) -> None:
    scenario = _abrupt_scenario("2010-01-01", "2010-03-01")
    augmented = _masking_augmented_frame(factory, real_train_df, scenario)
    assert (
        masking_simulator_no_leak_check(scenario, augmented, seed=RANDOM_SEED, derived_columns=[])
        is True
    ), name
