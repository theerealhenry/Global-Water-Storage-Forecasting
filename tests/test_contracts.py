"""Tests for tws_forecast.data.contracts.

Covers both directions deliberately: schemas must accept genuinely valid
data (the golden fixtures) *and* must reject specific, realistic corruption
of each invariant they claim to enforce. A schema that never fails hasn't
been shown to do anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from tws_forecast.data.contracts import (
    N_GRID_LOCATIONS,
    SAMPLE_SUBMISSION_SCHEMA,
    TEST_SCHEMA,
    TRAIN_SCHEMA,
    assert_full_grid,
)
from tws_forecast.data.loaders import load_sample_submission, load_test, load_train


# ---------------------------------------------------------------------------
# Positive cases: the golden fixtures are genuinely valid data and must pass.
# ---------------------------------------------------------------------------


def test_train_schema_accepts_golden_fixture(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    validated = TRAIN_SCHEMA.validate(df)
    assert len(validated) == len(df)


def test_test_schema_accepts_golden_fixture(golden_dir: Path) -> None:
    df = load_test(data_dir=golden_dir, validate=False)
    validated = TEST_SCHEMA.validate(df)
    assert len(validated) == len(df)


def test_sample_submission_schema_accepts_golden_fixture(golden_dir: Path) -> None:
    df = load_sample_submission(data_dir=golden_dir, validate=False)
    validated = SAMPLE_SUBMISSION_SCHEMA.validate(df)
    assert len(validated) == len(df)


# ---------------------------------------------------------------------------
# Negative cases: each check must actually reject the thing it claims to.
# ---------------------------------------------------------------------------


# pandera raises SchemaError for most single-check failures, but strict-mode
# column violations (and any lazy=True validation) raise SchemaErrors
# (plural) instead — the two do NOT share a common base class other than
# Exception. Verified empirically (SchemaErrors.__mro__ does not include
# SchemaError) rather than assumed, since the two look like they should be
# related by name. Negative tests below accept either.
_SCHEMA_FAILURE = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)


def test_test_schema_rejects_masking_invariant_violation(golden_dir: Path) -> None:
    df = load_test(data_dir=golden_dir, validate=False)
    bad = df.copy()
    masked_idx = bad[bad["TWS_t_masked"]].index[0]
    bad.loc[masked_idx, "TWS_t_masked"] = False  # now contradicts TWS_t being null
    with pytest.raises(_SCHEMA_FAILURE):
        TEST_SCHEMA.validate(bad)


def test_train_schema_rejects_null_tws_t(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    bad.loc[bad.index[0], "TWS_t"] = np.nan
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


def test_train_schema_rejects_out_of_range_latitude(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    bad.loc[bad.index[0], "lat"] = 999.0
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


def test_train_schema_rejects_out_of_range_longitude(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    bad.loc[bad.index[0], "lon"] = -200.0
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


def test_train_schema_rejects_broken_month_encoding(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    # Break the sin^2 + cos^2 == 1 invariant.
    bad.loc[bad.index[0], "month_sin"] = 0.9
    bad.loc[bad.index[0], "month_cos"] = 0.9
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


def test_train_schema_rejects_duplicate_sample_id(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


def test_train_schema_rejects_unexpected_extra_column(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    bad["unexpected_column"] = 0
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)  # strict=True should reject unknown columns


def test_train_schema_rejects_time_outside_training_window(golden_dir: Path) -> None:
    df = load_train(data_dir=golden_dir, validate=False)
    bad = df.copy()
    bad.loc[bad.index[0], "time"] = pd.Timestamp("1999-01-01")
    with pytest.raises(_SCHEMA_FAILURE):
        TRAIN_SCHEMA.validate(bad)


# ---------------------------------------------------------------------------
# assert_full_grid: deliberately separate from the reusable schemas (see
# contracts.py docstring) — must reject a subset and accept a full grid.
# ---------------------------------------------------------------------------


def test_assert_full_grid_rejects_golden_fixture_subset(golden_dir: Path) -> None:
    # The golden fixture has exactly 10 locations, not the full 15,715 —
    # assert_full_grid must reject it, which also confirms it isn't
    # accidentally baked into TRAIN_SCHEMA/TEST_SCHEMA (tested above, both
    # of which *do* accept this same fixture).
    df = load_train(data_dir=golden_dir, validate=False)
    with pytest.raises(ValueError, match="expected exactly"):
        assert_full_grid(df, name="golden fixture")


def test_assert_full_grid_accepts_synthetic_full_grid() -> None:
    # Construct a minimal synthetic frame with exactly N_GRID_LOCATIONS
    # distinct (lat, lon) pairs to confirm the *accept* path works too,
    # without needing the real 289MB Train.csv in the test suite.
    lat = np.arange(N_GRID_LOCATIONS, dtype=float)
    lon = np.zeros(N_GRID_LOCATIONS, dtype=float)
    synthetic = pd.DataFrame({"lat": lat, "lon": lon})
    assert_full_grid(synthetic, name="synthetic") # should not raise
