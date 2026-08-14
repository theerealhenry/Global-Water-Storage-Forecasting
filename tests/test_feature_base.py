"""Tests for tws_forecast.features.base.Transformer and
tws_forecast.features.registry — Project Phase 4 step 4.4, per
docs/PHASE4_EXECUTION_PLAN.md §4.4.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tws_forecast.features.base import Transformer
from tws_forecast.features.registry import (
    FEATURE_CONFIG_DIR,
    FEATURE_CONFIG_REGISTRY,
    FeatureConfig,
    list_feature_configs,
    load_feature_config,
)

# --- Transformer protocol ---------------------------------------------------


class _StubTransformer:
    """A minimal object satisfying the Transformer protocol structurally
    (fit/transform), with no relation to any real feature module — pins
    the protocol's shape independent of any concrete implementation, the
    same pattern validation.tiers.Predictor's own protocol test uses."""

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["value"].mean())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["value_centered"] = out["value"] - self._mean
        return out


def test_stub_satisfies_transformer_protocol_structurally() -> None:
    stub = _StubTransformer()
    assert isinstance(stub, Transformer)


def test_object_missing_transform_does_not_satisfy_protocol() -> None:
    class _MissingTransform:
        def fit(self, train_df: pd.DataFrame) -> None:
            pass

    assert not isinstance(_MissingTransform(), Transformer)


def test_object_missing_fit_does_not_satisfy_protocol() -> None:
    class _MissingFit:
        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    assert not isinstance(_MissingFit(), Transformer)


def test_stub_transformer_fit_transform_round_trip() -> None:
    train_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    stub = _StubTransformer()
    stub.fit(train_df)

    other_df = pd.DataFrame({"value": [10.0, 20.0]})
    result = stub.transform(other_df)

    assert list(result["value_centered"]) == pytest.approx([8.0, 18.0])
    # transform must not mutate its input
    assert list(other_df.columns) == ["value"]


def test_transform_is_pure_given_fixed_fit_state() -> None:
    train_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    stub = _StubTransformer()
    stub.fit(train_df)

    df = pd.DataFrame({"value": [5.0]})
    first = stub.transform(df)
    second = stub.transform(df)
    pd.testing.assert_frame_equal(first, second)


# --- Feature config registry -------------------------------------------------


def test_all_four_expected_configs_are_registered() -> None:
    assert set(list_feature_configs()) == {
        "signatures",
        "spatial_history",
        "temporal",
        "environmental",
    }


@pytest.mark.parametrize("name", ["signatures", "spatial_history", "temporal", "environmental"])
def test_every_config_yaml_parses_into_a_valid_typed_config(name: str) -> None:
    config = load_feature_config(name)
    assert isinstance(config, FeatureConfig)
    assert config.name == name
    assert config.description
    assert config.source_rationale


def test_signatures_config_has_its_type_specific_fields() -> None:
    config = load_feature_config("signatures")
    assert config.feature_type == "signatures"
    assert config.shrinkage_k is not None
    assert config.shrinkage_k > 0
    assert config.trailing_windows == (12, 24)


def test_spatial_history_config_has_its_type_specific_fields() -> None:
    config = load_feature_config("spatial_history")
    assert config.feature_type == "spatial_history"
    assert config.n_neighbors is not None and config.n_neighbors > 0
    assert config.distance_weighting in ("inverse_distance", "flat_mean")
    assert config.max_neighbor_distance_km is not None


def test_temporal_config_has_its_type_specific_fields() -> None:
    config = load_feature_config("temporal")
    assert config.feature_type == "temporal"
    assert config.trend_window_months is not None
    assert len(config.trend_window_months) > 0


def test_environmental_config_has_its_type_specific_fields() -> None:
    config = load_feature_config("environmental")
    assert config.feature_type == "environmental"
    assert config.spei_diff_lags is not None
    assert config.drought_threshold is not None


def test_load_feature_config_raises_keyerror_for_unknown_name() -> None:
    with pytest.raises(KeyError, match="nonexistent_feature"):
        load_feature_config("nonexistent_feature")


def test_feature_config_registry_matches_directory_contents() -> None:
    yaml_stems = {p.stem for p in FEATURE_CONFIG_DIR.glob("*.yaml")}
    assert set(FEATURE_CONFIG_REGISTRY) == yaml_stems


def test_signatures_config_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="requires"):
        FeatureConfig(
            name="bad_signatures",
            feature_type="signatures",
            description="x",
            source_rationale="x",
            # shrinkage_k and trailing_windows both omitted
        )


def test_spatial_history_config_rejects_nonpositive_n_neighbors() -> None:
    with pytest.raises(ValueError, match="n_neighbors"):
        FeatureConfig(
            name="bad_spatial",
            feature_type="spatial_history",
            description="x",
            source_rationale="x",
            n_neighbors=0,
            distance_weighting="inverse_distance",
        )


def test_config_name_must_match_filename_stem() -> None:
    # load_feature_config enforces this for real files; this test pins the
    # same check independent of any file on disk staying correctly named.
    from tws_forecast.features import registry

    original = dict(registry.FEATURE_CONFIG_REGISTRY)
    try:
        # Point the "temporal" key at a file whose declared name differs.
        registry.FEATURE_CONFIG_REGISTRY["temporal"] = FEATURE_CONFIG_DIR / "environmental.yaml"
        with pytest.raises(ValueError, match="does not match"):
            load_feature_config("temporal")
    finally:
        registry.FEATURE_CONFIG_REGISTRY.clear()
        registry.FEATURE_CONFIG_REGISTRY.update(original)


def test_feature_config_is_frozen() -> None:
    config = load_feature_config("signatures")
    with pytest.raises(Exception):
        config.shrinkage_k = 999  # type: ignore[misc]
