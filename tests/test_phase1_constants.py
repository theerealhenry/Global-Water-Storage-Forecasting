"""Tests for tws_forecast.validation.phase1_constants.

These are drift-detection and internal-consistency tests, not a
re-verification of the source notebook itself (that verification was done
directly against notebooks/02_forecastability.ipynb's executed output cells
before this module was written — see the module docstring). What's checked
here: internal consistency between related constants, and that a few
hand-derived values (counts, orderings) match what the notebook reported.
"""

from collections import Counter

import pandas as pd

from tws_forecast.validation import phase1_constants as c


def test_blackout_k_distribution_has_twelve_entries() -> None:
    assert len(c.BLACKOUT_K_DISTRIBUTION) == 12
    assert sum(c.BLACKOUT_K_VALUE_COUNTS.values()) == 12


def test_blackout_k_value_counts_match_distribution() -> None:
    assert Counter(c.BLACKOUT_K_DISTRIBUTION) == c.BLACKOUT_K_VALUE_COUNTS


def test_blackout_k_distribution_matches_verified_notebook_order() -> None:
    # Exact order re-verified against notebooks/02_forecastability.ipynb
    # Cell 62 (stale_df, ascending by blackout offset).
    assert c.BLACKOUT_K_DISTRIBUTION == [2, 3, 2, 3, 4, 2, 3, 4, 5, 6, 7, 2]


def test_blackout_k_range_is_2_to_7() -> None:
    assert min(c.BLACKOUT_K_DISTRIBUTION) == 2
    assert max(c.BLACKOUT_K_DISTRIBUTION) == 7


def test_full_and_blackout_offsets_partition_eighteen_months() -> None:
    assert len(c.TEST_FULL_OFFSETS) == 6
    assert len(c.TEST_BLACKOUT_OFFSETS) == 12
    assert set(c.TEST_FULL_OFFSETS).isdisjoint(c.TEST_BLACKOUT_OFFSETS)
    assert len(c.TEST_FULL_OFFSETS) + len(c.TEST_BLACKOUT_OFFSETS) == 18


def test_test_months_match_full_plus_blackout_sets() -> None:
    assert len(c.TEST_MONTHS) == 18
    assert len(c.TEST_FULL_MONTHS) == 6
    assert len(c.TEST_BLACKOUT_MONTHS) == 12
    assert set(c.TEST_FULL_MONTHS) | set(c.TEST_BLACKOUT_MONTHS) == set(c.TEST_MONTHS)
    assert set(c.TEST_FULL_MONTHS).isdisjoint(c.TEST_BLACKOUT_MONTHS)


def test_offset_lists_align_positionally_with_month_lists() -> None:
    # TEST_FULL_OFFSETS[i] must correspond to TEST_FULL_MONTHS[i], verified
    # via the known 2015-09 = offset 0 anchor and month arithmetic.
    t0 = pd.Timestamp(c.TEST_FULL_MONTHS[0])
    for offset, month_str in zip(c.TEST_FULL_OFFSETS, c.TEST_FULL_MONTHS):
        expected = t0 + pd.DateOffset(months=offset)
        assert expected.strftime("%Y-%m") == month_str
    for offset, month_str in zip(c.TEST_BLACKOUT_OFFSETS, c.TEST_BLACKOUT_MONTHS):
        expected = t0 + pd.DateOffset(months=offset)
        assert expected.strftime("%Y-%m") == month_str


def test_baseline_ordering_matches_known_difficulty_ranking() -> None:
    # Oracle persistence (easiest, only scored on FULL months) is the lowest
    # RMSE; the realistic hybrid floor sits strictly between the oracle
    # ceiling and the last-known/climatology baselines.
    assert c.BASELINE_A < c.BASELINE_D < c.BASELINE_B < c.BASELINE_C


def test_promotion_thresholds_strictly_decreasing_in_difficulty() -> None:
    order = ["naive_floor", "oracle_ceiling", "beat_mohar", "serious_contender", "exceptional"]
    values = [c.PROMOTION_THRESHOLDS[k] for k in order]
    assert values == sorted(values, reverse=True)
    assert c.PROMOTION_THRESHOLDS["naive_floor"] == c.BASELINE_D


def test_clean_train_span_is_84_months() -> None:
    span = pd.date_range(c.CLEAN_TRAIN_SPAN_START, c.CLEAN_TRAIN_SPAN_END, freq="MS")
    assert len(span) == 84


def test_missing_train_months_counts_and_partition() -> None:
    assert len(c.MISSING_TRAIN_MONTHS) == 22
    assert len(c.MISSING_TRAIN_MONTHS_PRE_2011) == 5
    assert len(c.MISSING_TRAIN_MONTHS_2011_ONWARD) == 17
    assert (
        set(c.MISSING_TRAIN_MONTHS_PRE_2011) | set(c.MISSING_TRAIN_MONTHS_2011_ONWARD)
        == set(c.MISSING_TRAIN_MONTHS)
    )
    assert all(m < "2011" for m in c.MISSING_TRAIN_MONTHS_PRE_2011)
    assert all(m >= "2011" for m in c.MISSING_TRAIN_MONTHS_2011_ONWARD)


def test_calendar_month_test_share_sums_near_100_and_october_is_zero() -> None:
    assert c.CALENDAR_MONTH_TEST_SHARE_PCT["Oct"] == 0.00
    total = sum(c.CALENDAR_MONTH_TEST_SHARE_PCT.values())
    assert 99.5 <= total <= 100.5  # rounding tolerance; source values are .round(2)
    assert set(c.CALENDAR_MONTH_TEST_SHARE_PCT) == set(c.CALENDAR_MONTH_TRAIN_SHARE_PCT)


def test_calendar_month_train_share_has_no_zero_months() -> None:
    # Unlike test, train has full calendar-month coverage (A-011's finding
    # is specifically that this gap is test-only).
    assert all(v > 0 for v in c.CALENDAR_MONTH_TRAIN_SHARE_PCT.values())


def test_acf_quartile_ar1_params_rho_increasing_with_quartile() -> None:
    order = ["Q1_low_ACF", "Q2", "Q3", "Q4_high_ACF"]
    rhos = [c.ACF_QUARTILE_AR1_PARAMS[q]["rho"] for q in order]
    assert rhos == sorted(rhos)
    assert all(0 < r < 1 for r in rhos)
