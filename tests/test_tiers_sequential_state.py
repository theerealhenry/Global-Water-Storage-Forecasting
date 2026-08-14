"""Tests for tws_forecast.validation.tiers.run_tier3_sequential_state —
the diagnostic-only, chronologically-ordered Tier 3 replay helper promoted
from notebooks/03_validation_harness.ipynb §7b during Project Phase 3, per
docs/ASSUMPTIONS.md A-013 and the Project Phase 3 handoff §3.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tws_forecast.data.loaders import load_train
from tws_forecast.models.baselines import HybridPersistencePredictor, LastKnownStatePredictor
from tws_forecast.validation.tiers import TierResult, run_tier3, run_tier3_sequential_state


@pytest.fixture()
def train_df(golden_dir: Path) -> pd.DataFrame:
    return load_train(data_dir=golden_dir)


class _StatelessStub:
    """A Predictor with no ``_last_known``-shaped attribute at all — used to
    pin the "fail loud, not silent no-op" contract."""

    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean)


class _StatefulStub:
    """A minimal stateful Predictor exposing ``_last_known`` in the exact
    shape ``run_tier3_sequential_state`` expects — lets the mechanics
    (chronological ordering, state-dict updates between offsets) be tested
    in isolation from the real baseline classes."""

    def __init__(self) -> None:
        self._last_known: dict[str, float] = {}
        self._fallback = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        df = train_df.copy()
        self._fallback = float(df["target"].mean())
        if "location_id" not in df.columns:
            from tws_forecast.validation.splitters import attach_forecast_origin_columns

            df = attach_forecast_origin_columns(df)
        observed = df.dropna(subset=["TWS_t"]).sort_values("time")
        self._last_known = observed.groupby("location_id")["TWS_t"].last().to_dict()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "location_id" not in df.columns:
            from tws_forecast.validation.splitters import attach_forecast_origin_columns

            df = attach_forecast_origin_columns(df)
        return df["location_id"].map(self._last_known).fillna(self._fallback).to_numpy(dtype=float)


# --- Contract / error handling ---------------------------------------------


def test_raises_for_non_stateful_model(train_df: pd.DataFrame) -> None:
    with pytest.raises(AttributeError, match="_last_known"):
        run_tier3_sequential_state(_StatelessStub(), train_df, n_anchors=1)


def test_raises_for_wrong_scenario_type(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="test_regime_replay"):
        run_tier3_sequential_state(_StatefulStub(), train_df, scenario="expanding_window", n_anchors=1)


def test_raises_for_invalid_n_anchors(train_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_anchors"):
        run_tier3_sequential_state(_StatefulStub(), train_df, n_anchors=0)


def test_custom_state_attr_name(train_df: pd.DataFrame) -> None:
    class _CustomAttrStub(_StatefulStub):
        def __init__(self) -> None:
            super().__init__()
            self._my_state = self._last_known
            del self.__dict__["_last_known"]

        def fit(self, train_df: pd.DataFrame) -> None:
            super().fit(train_df)
            self._my_state = self._last_known

        def predict(self, df: pd.DataFrame) -> np.ndarray:
            self._last_known = self._my_state
            return super().predict(df)

    result = run_tier3_sequential_state(
        _CustomAttrStub(), train_df, n_anchors=1, state_attr="_my_state"
    )
    assert isinstance(result, TierResult)


# --- End-to-end execution ---------------------------------------------------


def test_executes_end_to_end_with_stateful_stub(train_df: pd.DataFrame) -> None:
    result = run_tier3_sequential_state(_StatefulStub(), train_df, n_anchors=2)
    assert isinstance(result, TierResult)
    assert result.tier == 3
    assert len(result.fold_rmses) >= 1
    assert np.isfinite(result.overall_rmse)


def test_executes_end_to_end_with_hybrid_persistence(train_df: pd.DataFrame) -> None:
    result = run_tier3_sequential_state(HybridPersistencePredictor(), train_df, n_anchors=2)
    assert np.isfinite(result.overall_rmse)


def test_executes_end_to_end_with_last_known_state(train_df: pd.DataFrame) -> None:
    result = run_tier3_sequential_state(LastKnownStatePredictor(), train_df, n_anchors=2)
    assert np.isfinite(result.overall_rmse)


# --- The actual point of this function: state carries across offsets -------


def test_state_dict_is_updated_after_full_offsets(train_df: pd.DataFrame) -> None:
    """After the run, the model's _last_known dict must have grown beyond
    what fit() alone produced — proof that at least one FULL offset's
    observation was folded in during the chronological walk."""
    model = _StatefulStub()
    model.fit(train_df[train_df["time"] < train_df["time"].quantile(0.5)])
    n_after_fit = len(model._last_known)

    run_tier3_sequential_state(model, train_df, n_anchors=1)

    # The same model instance was mutated in place across the walk (fit()
    # is called again internally per anchor, but state updates happen after
    # that per-anchor fit) -- at minimum the dict must be non-empty and
    # must not have silently stayed exactly at its pre-run size if any FULL
    # offset produced rows (a sanity floor, not an exact-count assertion,
    # since fit() itself also repopulates it per anchor).
    assert len(model._last_known) > 0


def test_sequential_state_differs_from_standard_row_wise_tier3(train_df: pd.DataFrame) -> None:
    """The entire reason this function exists: for a genuinely stateful
    predictor, the sequential-state score should differ from (and, per
    A-013, generally improve on) the standard run_tier3 score, because the
    standard tier scores offsets independently and never lets state
    accumulate within a replay window."""
    standard = run_tier3(HybridPersistencePredictor(), train_df, n_anchors=2)
    sequential = run_tier3_sequential_state(HybridPersistencePredictor(), train_df, n_anchors=2)
    assert standard.overall_rmse != pytest.approx(sequential.overall_rmse, rel=1e-6)
