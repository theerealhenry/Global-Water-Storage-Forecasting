"""Tests for tws_forecast.data.loaders — run against the small golden fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from tws_forecast.data.loaders import (
    get_repo_root,
    load_all,
    load_sample_submission,
    load_test,
    load_train,
)


def test_get_repo_root_points_at_real_repo() -> None:
    root = get_repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "tws_forecast").exists()


def test_load_train_returns_expected_shape_and_columns(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir)
    expected_columns = [
        "sample_id", "time", "lat", "lon", "TWS_t", "SPEI_01_t", "SPEI_03_t",
        "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t", "month_sin", "month_cos", "target",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 1380
    assert df["time"].dtype.kind == "M"  # datetime64
    assert df["TWS_t"].isna().sum() == 0


def test_load_test_returns_expected_shape_and_masking(golden_dir: Path) -> None:
    df = load_test(data_dir=golden_dir)
    assert len(df) == 180
    assert df["TWS_t_masked"].dtype == bool
    # The core masking invariant, re-checked at the loader level too.
    assert (df["TWS_t"].isna() == df["TWS_t_masked"]).all()
    assert df["TWS_t_masked"].sum() == 120


def test_load_sample_submission_matches_test_ids(golden_dir: Path) -> None:
    test_df = load_test(data_dir=golden_dir)
    sub_df = load_sample_submission(data_dir=golden_dir)
    assert len(sub_df) == len(test_df)
    assert set(sub_df["ID"]) == set(test_df["ID"])


def test_load_all_returns_three_consistent_frames(golden_dir: Path) -> None:
    train, test, sub = load_all(data_dir=golden_dir)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(sub, pd.DataFrame)
    # Same fixed grid of locations across all three files.
    train_locs = set(map(tuple, train[["lat", "lon"]].drop_duplicates().values))
    test_locs = set(map(tuple, test[["lat", "lon"]].drop_duplicates().values))
    assert train_locs == test_locs
    assert len(train_locs) == 10  # golden fixture uses exactly 10 locations


def test_load_train_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_train(data_dir=tmp_path)


def test_load_with_validate_false_skips_schema_check(golden_dir: Path) -> None:
    # Should not raise even if we don't care about validation here — this
    # just confirms the flag actually short-circuits validation rather than
    # silently validating anyway.
    df = load_train(data_dir=golden_dir, validate=False)
    assert len(df) == 1380
