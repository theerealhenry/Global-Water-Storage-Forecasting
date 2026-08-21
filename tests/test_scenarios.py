"""Tests for tws_forecast.validation.scenarios.

Two jobs: (1) confirm every real configs/validation/*.yaml file parses into
a valid ScenarioConfig and, for test_regime_replay/blackout_curve, matches
phase1_constants.py exactly (drift detection between the two places these
numbers live); (2) confirm ScenarioConfig's own validation logic rejects
malformed scenarios, using small in-memory fixtures rather than the real
files.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tws_forecast.data.loaders import load_train
from tws_forecast.validation.phase1_constants import (
    BLACKOUT_K_BY_OFFSET,
    BLACKOUT_K_DISTRIBUTION,
    TEST_BLACKOUT_OFFSETS,
    TEST_FULL_OFFSETS,
)
from tws_forecast.validation.scenarios import (
    SCENARIO_DIR,
    SCENARIO_REGISTRY,
    ScenarioConfig,
    list_scenarios,
    load_scenario,
)
from tws_forecast.validation.splitters import expanding_window_splits

# --- real registered scenarios ---------------------------------------------


def test_all_six_expected_scenarios_are_registered() -> None:
    # "_quick" variants (added after Project Phase 4 step 4.9's first proof
    # run found the full-rigor scenarios an intractable per-candidate cost
    # for notebooks/05_state_features.ipynb's exploratory/comparative
    # sections -- see configs/validation/expanding_window_quick.yaml and
    # blackout_curve_quick.yaml's own docstrings) are cheap, same-shape
    # siblings of expanding_window/blackout_curve, never used for a report
    # a promote() call is actually based on.
    assert set(list_scenarios()) == {
        "expanding_window",
        "expanding_window_quick",
        "blackout_curve",
        "blackout_curve_quick",
        "test_regime_replay",
        "2015_like",
    }


@pytest.mark.parametrize(
    "name",
    [
        "expanding_window",
        "expanding_window_quick",
        "blackout_curve",
        "blackout_curve_quick",
        "test_regime_replay",
        "2015_like",
    ],
)
def test_every_registered_scenario_loads_and_validates(name: str) -> None:
    config = load_scenario(name)
    assert isinstance(config, ScenarioConfig)
    assert config.name == name


def test_load_unknown_scenario_raises_key_error() -> None:
    with pytest.raises(KeyError, match="No scenario named"):
        load_scenario("does_not_exist")


def test_test_regime_replay_offsets_match_phase1_constants_exactly() -> None:
    # Drift-detection test: these numbers exist in two places (the YAML and
    # phase1_constants.py) on purpose (the YAML must be readable on its own
    # without importing Python), so this test is what keeps them from
    # silently diverging if one is edited without the other.
    config = load_scenario("test_regime_replay")
    assert list(config.full_offsets) == TEST_FULL_OFFSETS
    assert list(config.blackout_offsets) == TEST_BLACKOUT_OFFSETS
    assert config.blackout_k_by_offset == BLACKOUT_K_BY_OFFSET


def test_blackout_curve_k_distribution_matches_phase1_constants() -> None:
    config = load_scenario("blackout_curve")
    assert list(config.k_distribution) == BLACKOUT_K_DISTRIBUTION


def test_2015_like_scenario_isolates_exactly_2015_jan_to_aug(golden_dir: Path) -> None:
    import pandas as pd

    config = load_scenario("2015_like")
    train = load_train(data_dir=golden_dir)
    ((train_fold, val_fold),) = list(expanding_window_splits(train, **config.splitter.model_dump()))
    assert train_fold["time"].max() == pd.Timestamp("2014-12-01")
    assert val_fold["time"].min() == pd.Timestamp("2015-01-01")
    assert val_fold["time"].max() == pd.Timestamp("2015-08-01")


def test_scenario_dir_and_registry_are_consistent() -> None:
    assert SCENARIO_DIR.exists()
    yaml_stems = {p.stem for p in SCENARIO_DIR.glob("*.yaml")}
    assert yaml_stems == set(SCENARIO_REGISTRY)


# --- ScenarioConfig validation logic, via in-memory fixtures ---------------


def _base_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="fixture",
        tier=1,
        scenario_type="expanding_window",
        description="test fixture",
        source_rationale="unit test",
    )
    kwargs.update(overrides)
    return kwargs


def test_blackout_curve_missing_required_fields_raises() -> None:
    with pytest.raises(ValidationError, match="blackout_curve requires"):
        ScenarioConfig(**_base_kwargs(tier=2, scenario_type="blackout_curve"))


def test_test_regime_replay_missing_required_fields_raises() -> None:
    with pytest.raises(ValidationError, match="test_regime_replay requires"):
        ScenarioConfig(**_base_kwargs(tier=3, scenario_type="test_regime_replay"))


def test_test_regime_replay_k_keys_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="must exactly match"):
        ScenarioConfig(
            **_base_kwargs(
                tier=3,
                scenario_type="test_regime_replay",
                full_offsets=(0,),
                blackout_offsets=(1, 2),
                blackout_k_by_offset={1: 2},  # missing key 2
            )
        )


def test_test_regime_replay_overlapping_offsets_raises() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ScenarioConfig(
            **_base_kwargs(
                tier=3,
                scenario_type="test_regime_replay",
                full_offsets=(0, 1),
                blackout_offsets=(1, 2),
                blackout_k_by_offset={1: 2, 2: 3},
            )
        )


def test_exception_rate_out_of_range_raises() -> None:
    with pytest.raises(ValidationError, match="exception_rate"):
        ScenarioConfig(**_base_kwargs(exception_rate=1.5))


def test_scenario_config_is_frozen() -> None:
    config = ScenarioConfig(**_base_kwargs())
    with pytest.raises(ValidationError):
        config.tier = 2  # type: ignore[misc]


def test_load_scenario_name_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tws_forecast.validation.scenarios as scenarios_module

    bad_dir = tmp_path / "validation"
    bad_dir.mkdir()
    bad_file = bad_dir / "actual_filename.yaml"
    bad_file.write_text(yaml.dump(_base_kwargs(name="declared_name_does_not_match")))
    monkeypatch.setattr(scenarios_module, "SCENARIO_DIR", bad_dir)
    monkeypatch.setattr(scenarios_module, "SCENARIO_REGISTRY", {"actual_filename": bad_file})
    with pytest.raises(ValueError, match="does not match its filename stem"):
        scenarios_module.load_scenario("actual_filename")
