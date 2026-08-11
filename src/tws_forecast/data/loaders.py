"""Data loading utilities for the TWS forecasting pipeline.

All raw-data access in this project goes through this module. Every notebook,
script, and later pipeline stage should load ``Train.csv`` / ``Test.csv`` /
``SampleSubmission.csv`` through the functions here rather than writing its
own ``pd.read_csv`` call, so dtypes, path resolution, and schema validation
stay in exactly one place.

Column semantics are documented in ``docs/DATA_DICTIONARY.md`` and were
verified directly against the raw files (full-column null counts, dtype
inference, grid/location cross-checks) before being encoded here — nothing
in this module is guessed from the competition description alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DATA_DIR",
    "get_repo_root",
    "load_train",
    "load_test",
    "load_sample_submission",
    "load_all",
]

# This file lives at <repo_root>/src/tws_forecast/data/loaders.py, so the
# repo root is three parents up. Resolved once, at import time.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "raw"

# Dtypes verified against the full raw files (see docs/DATA_DICTIONARY.md).
# ``time`` is parsed separately via ``parse_dates`` rather than listed here.
_SHARED_FLOAT_COLUMNS = [
    "lat",
    "lon",
    "TWS_t",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
    "month_sin",
    "month_cos",
]

_TRAIN_DTYPES: dict[str, str] = {
    "sample_id": "string",
    **{c: "float64" for c in _SHARED_FLOAT_COLUMNS},
    "target": "float64",
}

_TEST_DTYPES: dict[str, str] = {
    "ID": "string",
    **{c: "float64" for c in _SHARED_FLOAT_COLUMNS},
    # TWS_t_masked is left to pandas' own inference: the raw column is a
    # clean "True"/"False" string with zero irregular values (verified),
    # and pandas' C parser infers it as a native bool column without help.
}

_SAMPLE_SUBMISSION_DTYPES: dict[str, str] = {
    "ID": "string",
    "Target": "float64",
}


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def _resolve_data_dir(data_dir: Path | str | None) -> Path:
    resolved = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not resolved.exists():
        raise FileNotFoundError(
            f"Data directory not found: {resolved}. If you're running this "
            "outside the repo's expected layout, pass data_dir= explicitly."
        )
    return resolved


def load_train(
    data_dir: Path | str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the training set.

    Parameters
    ----------
    data_dir:
        Directory containing ``Train.csv``. Defaults to ``data/raw/`` at the
        repo root. Overridable so tests can point at the golden fixture
        directory (``tests/data/golden/``) instead of the full raw file.
    validate:
        If True (default), validate the loaded frame against
        ``tws_forecast.data.contracts.TRAIN_SCHEMA`` before returning it.
        Set False only for quick, non-authoritative inspection — every
        pipeline code path should leave this on.

        The stronger "complete 15,715-location grid" check
        (``assert_full_grid``) only runs when ``data_dir`` is left at its
        default (i.e. loading the real, complete raw file). It is
        deliberately skipped for any explicitly-passed ``data_dir``, since
        that's the mechanism tests use to point at intentionally-partial
        fixtures (e.g. ``tests/data/golden/``, a CV fold written to a temp
        dir) — those are legitimate subsets and TRAIN_SCHEMA alone is the
        right level of validation for them.

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_id, time, lat, lon, TWS_t, SPEI_01_t, SPEI_03_t,
        SPEI_06_t, SPEI_12_t, SOIL_MOISTURE_t, month_sin, month_cos,
        target``. ``TWS_t`` and ``target`` are always populated in this
        file — masking is a test-set-only phenomenon in this dataset.
    """
    is_default_dir = data_dir is None
    path = _resolve_data_dir(data_dir) / "Train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Train.csv not found at {path}")

    df = pd.read_csv(path, dtype=_TRAIN_DTYPES, parse_dates=["time"])
    logger.info("Loaded Train.csv: %d rows, %d columns, %s to %s", len(df), df.shape[1],
                df["time"].min().date(), df["time"].max().date())

    if validate:
        from tws_forecast.data.contracts import TRAIN_SCHEMA, assert_full_grid

        df = TRAIN_SCHEMA.validate(df)
        if is_default_dir:
            assert_full_grid(df, name="Train.csv")
        logger.info("Train.csv passed schema validation")

    return df


def load_test(
    data_dir: Path | str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the test set.

    Parameters
    ----------
    data_dir:
        Directory containing ``Test.csv``. Defaults to ``data/raw/`` at the
        repo root.
    validate:
        If True (default), validate against
        ``tws_forecast.data.contracts.TEST_SCHEMA`` before returning. As
        with ``load_train``, ``assert_full_grid`` only runs when
        ``data_dir`` is left at its default — see ``load_train`` docstring.

    Returns
    -------
    pd.DataFrame
        Columns: ``ID, time, lat, lon, TWS_t, SPEI_01_t, SPEI_03_t,
        SPEI_06_t, SPEI_12_t, SOIL_MOISTURE_t, month_sin, month_cos,
        TWS_t_masked``. ``TWS_t`` is null exactly where ``TWS_t_masked`` is
        True (verified with zero mismatches across all 280,961 rows) — no
        ``target`` column, since it's withheld by the competition.
    """
    is_default_dir = data_dir is None
    path = _resolve_data_dir(data_dir) / "Test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test.csv not found at {path}")

    df = pd.read_csv(path, dtype=_TEST_DTYPES, parse_dates=["time"])
    n_masked = int(df["TWS_t_masked"].sum())
    logger.info(
        "Loaded Test.csv: %d rows, %d columns, %s to %s, %d masked (%.1f%%)",
        len(df), df.shape[1], df["time"].min().date(), df["time"].max().date(),
        n_masked, 100 * n_masked / len(df),
    )

    if validate:
        from tws_forecast.data.contracts import TEST_SCHEMA, assert_full_grid

        df = TEST_SCHEMA.validate(df)
        if is_default_dir:
            assert_full_grid(df, name="Test.csv")
        logger.info("Test.csv passed schema validation")

    return df


def load_sample_submission(
    data_dir: Path | str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load ``SampleSubmission.csv`` — the exact schema a submission file must match."""
    path = _resolve_data_dir(data_dir) / "SampleSubmission.csv"
    if not path.exists():
        raise FileNotFoundError(f"SampleSubmission.csv not found at {path}")

    df = pd.read_csv(path, dtype=_SAMPLE_SUBMISSION_DTYPES)
    logger.info("Loaded SampleSubmission.csv: %d rows", len(df))

    if validate:
        from tws_forecast.data.contracts import SAMPLE_SUBMISSION_SCHEMA

        df = SAMPLE_SUBMISSION_SCHEMA.validate(df)
        logger.info("SampleSubmission.csv passed schema validation")

    return df


def load_all(
    data_dir: Path | str | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper loading train, test, and sample submission together."""
    return (
        load_train(data_dir=data_dir, validate=validate),
        load_test(data_dir=data_dir, validate=validate),
        load_sample_submission(data_dir=data_dir, validate=validate),
    )
