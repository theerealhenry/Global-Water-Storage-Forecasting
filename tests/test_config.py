"""Tests for tws_forecast.utils.config."""

from datetime import date
from pathlib import Path

import pytest

from tws_forecast.utils.config import DEFAULT_CONFIG_PATH, load_base_config


def test_default_config_path_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()
    assert DEFAULT_CONFIG_PATH.name == "base.yaml"


def test_load_base_config_matches_phase1_dates() -> None:
    config = load_base_config()
    assert config.random_seed == 42
    assert config.data_dir == "data/raw"
    assert config.train_period.start == date(2002, 5, 1)
    assert config.train_period.end == date(2015, 8, 1)
    assert config.test_period.start == date(2015, 9, 1)
    assert config.test_period.end == date(2018, 12, 1)


def test_load_base_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_base_config(tmp_path / "does_not_exist.yaml")


def test_load_base_config_explicit_path(tmp_path: Path) -> None:
    fixture = tmp_path / "base.yaml"
    fixture.write_text(
        "random_seed: 7\n"
        "data_dir: some/dir\n"
        "train_period:\n  start: '2000-01-01'\n  end: '2001-01-01'\n"
        "test_period:\n  start: '2001-02-01'\n  end: '2001-03-01'\n"
    )
    config = load_base_config(fixture)
    assert config.random_seed == 7
    assert config.data_dir == "some/dir"
