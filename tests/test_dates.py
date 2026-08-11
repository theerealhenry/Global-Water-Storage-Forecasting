"""Tests for tws_forecast.utils.dates."""

import pandas as pd
import pytest

from tws_forecast.utils.dates import month_index, month_index_to_timestamp


@pytest.mark.parametrize(
    "s", ["2004-01-01", "2010-12-01", "2015-08-01", "2002-05-01", "1999-12-01"]
)
def test_round_trip(s: str) -> None:
    ts = pd.Timestamp(s)
    assert month_index_to_timestamp(month_index(ts)) == ts


def test_month_index_difference_matches_month_count() -> None:
    a = month_index("2004-01-01")
    b = month_index("2010-12-01")
    # 2004-01 to 2010-12 inclusive is 84 months, i.e. 83 steps apart.
    assert b - a == 83


def test_month_index_accepts_string_or_timestamp_identically() -> None:
    assert month_index("2015-09-01") == month_index(pd.Timestamp("2015-09-01"))


def test_month_index_drops_day_of_month() -> None:
    assert month_index("2015-09-01") == month_index("2015-09-28")


def test_month_index_to_timestamp_always_returns_day_one() -> None:
    ts = month_index_to_timestamp(month_index("2015-09-01"))
    assert ts.day == 1


def test_month_index_to_timestamp_handles_december_wraparound() -> None:
    # december's month index modulo 12 is 0, which needs the year-1 branch.
    dec_idx = month_index("2004-12-01")
    assert month_index_to_timestamp(dec_idx) == pd.Timestamp("2004-12-01")
    assert month_index_to_timestamp(dec_idx + 1) == pd.Timestamp("2005-01-01")
