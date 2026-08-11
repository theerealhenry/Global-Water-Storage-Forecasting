"""Tests for tws_forecast.state.reconstruction.ForecastOrigin."""

from pathlib import Path

import pandas as pd
import pytest

from tws_forecast.data.loaders import load_test, load_train
from tws_forecast.state.reconstruction import ForecastOrigin


def test_valid_construction_round_trips_fields() -> None:
    origin = ForecastOrigin(
        origin_time=pd.Timestamp("2015-09-01"),
        target_time=pd.Timestamp("2015-10-01"),
        horizon=1,
        information_cutoff=pd.Timestamp("2015-09-01"),
        location_id="-55.5_-68.5",
        regime="observed",
    )
    assert origin.origin_time == pd.Timestamp("2015-09-01")
    assert origin.target_time == pd.Timestamp("2015-10-01")
    assert origin.horizon == 1
    assert origin.information_cutoff == pd.Timestamp("2015-09-01")
    assert origin.location_id == "-55.5_-68.5"
    assert origin.regime == "observed"


def test_target_time_must_equal_origin_plus_horizon() -> None:
    with pytest.raises(ValueError, match="target_time"):
        ForecastOrigin(
            origin_time=pd.Timestamp("2015-09-01"),
            target_time=pd.Timestamp("2015-11-01"),  # wrong: should be 2015-10-01
            horizon=1,
            information_cutoff=pd.Timestamp("2015-09-01"),
            location_id="x",
            regime="observed",
        )


def test_information_cutoff_after_origin_raises() -> None:
    with pytest.raises(ValueError, match="information_cutoff"):
        ForecastOrigin(
            origin_time=pd.Timestamp("2015-09-01"),
            target_time=pd.Timestamp("2015-10-01"),
            horizon=1,
            information_cutoff=pd.Timestamp("2015-10-01"),  # after origin: not allowed
            location_id="x",
            regime="observed",
        )


def test_invalid_regime_raises() -> None:
    with pytest.raises(ValueError, match="regime"):
        ForecastOrigin(
            origin_time=pd.Timestamp("2015-09-01"),
            target_time=pd.Timestamp("2015-10-01"),
            horizon=1,
            information_cutoff=pd.Timestamp("2015-09-01"),
            location_id="x",
            regime="unknown",  # type: ignore[arg-type]
        )


def test_multi_month_horizon_is_respected() -> None:
    origin = ForecastOrigin(
        origin_time=pd.Timestamp("2015-09-01"),
        target_time=pd.Timestamp("2015-12-01"),
        horizon=3,
        information_cutoff=pd.Timestamp("2015-09-01"),
        location_id="x",
        regime="observed",
    )
    assert origin.horizon == 3


def test_from_row_on_train_golden_fixture(golden_dir: Path) -> None:
    train = load_train(data_dir=golden_dir)
    row = train.iloc[0]
    origin = ForecastOrigin.from_row(row)
    assert origin.origin_time == row["time"]
    assert origin.target_time == pd.Timestamp(row["time"]) + pd.DateOffset(months=1)
    assert origin.information_cutoff == origin.origin_time
    assert origin.location_id == f"{float(row['lat'])}_{float(row['lon'])}"
    # Train.csv has no masking column — TWS_t is always populated, so every
    # train-row-derived origin must be "observed".
    assert origin.regime == "observed"


def test_from_row_on_test_golden_fixture_respects_masking(golden_dir: Path) -> None:
    test_df = load_test(data_dir=golden_dir)
    masked_row = test_df[test_df["TWS_t_masked"]].iloc[0]
    unmasked_row = test_df[~test_df["TWS_t_masked"]].iloc[0]

    masked_origin = ForecastOrigin.from_row(masked_row)
    unmasked_origin = ForecastOrigin.from_row(unmasked_row)

    assert masked_origin.regime == "masked"
    assert unmasked_origin.regime == "observed"


def test_from_row_default_horizon_is_one(golden_dir: Path) -> None:
    train = load_train(data_dir=golden_dir)
    row = train.iloc[0]
    origin = ForecastOrigin.from_row(row)
    assert origin.horizon == 1


def test_from_row_explicit_horizon(golden_dir: Path) -> None:
    train = load_train(data_dir=golden_dir)
    row = train.iloc[0]
    origin = ForecastOrigin.from_row(row, horizon=2)
    assert origin.horizon == 2
    assert origin.target_time == pd.Timestamp(row["time"]) + pd.DateOffset(months=2)


def test_from_row_accepts_plain_dict() -> None:
    row = {"time": "2016-02-01", "lat": 10.5, "lon": -20.5, "TWS_t_masked": True}
    origin = ForecastOrigin.from_row(row)
    assert origin.regime == "masked"
    assert origin.location_id == "10.5_-20.5"
