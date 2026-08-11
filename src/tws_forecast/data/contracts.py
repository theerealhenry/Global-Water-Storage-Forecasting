"""Pandera schema contracts for the TWS forecasting dataset.

Every constraint below was checked directly against the full raw files
before being encoded here (full-column null counts, exact min/max per
numeric column, exact-match location-set comparison between Train and
Test) — see ``docs/DATA_DICTIONARY.md`` for the evidence. Nothing here is
guessed from the competition description; if a bound below ever looks
wrong, the fix is to re-verify against the raw data, not to loosen the
check to make it pass.

These schemas are the leakage/integrity firewall's first line of defense
(``ARCHITECTURE.md`` §7): if the raw data ever changes shape in a way that
would silently break a downstream assumption, loading fails loudly here
instead of producing a quietly-wrong result three pipeline stages later.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Verified exact grid size: 15,715 unique (lat, lon) pairs, identical sets
# in Train.csv and Test.csv (verified by exact set comparison, not just
# count) — see docs/DATA_DICTIONARY.md.
N_GRID_LOCATIONS = 15_715

# Verified exact lat/lon extent of the fixed grid (data/raw/Train.csv and
# data/raw/Test.csv both scanned in full).
_LAT_BOUNDS = Check.in_range(-55.5, 83.5)
_LON_BOUNDS = Check.in_range(-179.5, 179.5)

# TWS/SPEI/soil-moisture are standardized-anomaly-like variables (mean
# near 0, std near 0.8-1.0). Verified exact min/max across the full files
# fall within roughly [-5.4, 5.4]; bounds below add headroom so a
# legitimate extreme value doesn't fail validation, while still catching
# a gross encoding error (e.g. accidentally loading raw un-standardized
# units, which would be wildly outside this range).
_ANOMALY_BOUNDS = Check.in_range(-8.0, 8.0)

# month_sin / month_cos are a cyclical encoding of calendar month and are
# mathematically bounded to [-1, 1] exactly.
_UNIT_CIRCLE_BOUNDS = Check.in_range(-1.0, 1.0)


def assert_full_grid(df: "pa.typing.DataFrame", name: str = "dataset") -> None:
    """Assert a dataframe covers the complete, exact 15,715-location grid.

    Deliberately kept **out** of ``TRAIN_SCHEMA`` / ``TEST_SCHEMA``: those
    schemas validate the structural correctness of any conforming
    dataframe, including legitimate subsets (a single CV fold, a single
    month, a small test fixture) that will never have all 15,715
    locations. This function is the separate, stronger check that only
    the *complete* raw file needs to pass, called explicitly by the
    loaders right after schema validation — see ``loaders.py``.
    """
    n = df[["lat", "lon"]].drop_duplicates().shape[0]
    if n != N_GRID_LOCATIONS:
        raise ValueError(
            f"{name}: expected exactly {N_GRID_LOCATIONS} unique (lat, lon) "
            f"locations, found {n}. This check is only meaningful against "
            "the complete raw file, not a subset — if you're validating a "
            "fold or fixture, use TRAIN_SCHEMA/TEST_SCHEMA directly instead."
        )


def _month_sin_cos_on_unit_circle(df: "pa.typing.DataFrame") -> bool:
    # sin^2 + cos^2 == 1 for a valid cyclical-month encoding; small
    # floating-point tolerance for the check.
    total = df["month_sin"] ** 2 + df["month_cos"] ** 2
    return bool(((total - 1.0).abs() < 1e-6).all())


TRAIN_SCHEMA = DataFrameSchema(
    name="TrainSchema",
    columns={
        "sample_id": Column(str, Check.str_matches(r"^\d{8}_-?\d+\.\d_-?\d+\.\d$"), unique=True),
        "time": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("2002-05-01"), pd.Timestamp("2015-08-01")),
        ),
        "lat": Column(float, _LAT_BOUNDS),
        "lon": Column(float, _LON_BOUNDS),
        # TWS_t is never null in Train.csv — masking is a test-set-only
        # phenomenon in this dataset (verified: 0/2,154,021 nulls).
        "TWS_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_01_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_03_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_06_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_12_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SOIL_MOISTURE_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "month_sin": Column(float, _UNIT_CIRCLE_BOUNDS, nullable=False),
        "month_cos": Column(float, _UNIT_CIRCLE_BOUNDS, nullable=False),
        # target = next calendar month's TWS_t (verified directly against
        # the raw file); never null in Train.csv.
        "target": Column(float, _ANOMALY_BOUNDS, nullable=False),
    },
    checks=[
        Check(_month_sin_cos_on_unit_circle, error="month_sin^2 + month_cos^2 must equal 1"),
    ],
    strict=True,
    coerce=False,
)


TEST_SCHEMA = DataFrameSchema(
    name="TestSchema",
    columns={
        "ID": Column(str, Check.str_matches(r"^\d{8}_-?\d+\.\d_-?\d+\.\d$"), unique=True),
        "time": Column(
            "datetime64[ns]",
            Check.in_range(pd.Timestamp("2015-09-01"), pd.Timestamp("2018-12-01")),
        ),
        "lat": Column(float, _LAT_BOUNDS),
        "lon": Column(float, _LON_BOUNDS),
        # TWS_t IS nullable here — nullness is the masking mechanism
        # itself, and must equal TWS_t_masked exactly (checked below).
        "TWS_t": Column(float, _ANOMALY_BOUNDS, nullable=True),
        "SPEI_01_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_03_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_06_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SPEI_12_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "SOIL_MOISTURE_t": Column(float, _ANOMALY_BOUNDS, nullable=False),
        "month_sin": Column(float, _UNIT_CIRCLE_BOUNDS, nullable=False),
        "month_cos": Column(float, _UNIT_CIRCLE_BOUNDS, nullable=False),
        "TWS_t_masked": Column(bool, nullable=False),
    },
    checks=[
        Check(_month_sin_cos_on_unit_circle, error="month_sin^2 + month_cos^2 must equal 1"),
        Check(
            lambda df: bool((df["TWS_t"].isna() == df["TWS_t_masked"]).all()),
            error="TWS_t_masked must equal TWS_t.isna() for every row (masking firewall invariant, ARCHITECTURE.md §7)",
        ),
    ],
    strict=True,
    coerce=False,
)


SAMPLE_SUBMISSION_SCHEMA = DataFrameSchema(
    name="SampleSubmissionSchema",
    columns={
        "ID": Column(str, Check.str_matches(r"^\d{8}_-?\d+\.\d_-?\d+\.\d$"), unique=True),
        "Target": Column(float, nullable=False),
    },
    strict=True,
    coerce=False,
)
