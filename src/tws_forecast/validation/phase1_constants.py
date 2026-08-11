"""Phase 1 measured constants — the single source of truth Phase 2+ binds to.

Every value below was measured in ``notebooks/02_forecastability.ipynb``
(Project Phase 1, Experiments 1-7) or ``notebooks/01_eda.ipynb``, and is
cited with its exact source cell/section. Values were re-verified directly
against the executed notebook's output cells (not transcribed from prose or
from memory) on 2026-08-11, specifically to catch any transcription drift
before this file became the thing every later module trusts.

Nothing here is re-derived, approximated, or adjusted — per ADR-0004,
Project Phase 2's validation harness is required to reproduce these
distributions exactly, not build a generic/synthetic substitute. If Phase 1
is ever revisited and one of these numbers changes, this is the one file
that needs to change — no other module is permitted to hardcode a competing
copy of any of these values (see ``docs/PHASE2_EXECUTION_PLAN.md`` §0).
"""

from __future__ import annotations

__all__ = [
    "TRAIN_PERIOD_START",
    "TRAIN_PERIOD_END",
    "TEST_PERIOD_START",
    "TEST_PERIOD_END",
    "TEST_MONTHS",
    "TEST_FULL_MONTHS",
    "TEST_BLACKOUT_MONTHS",
    "TEST_FULL_OFFSETS",
    "TEST_BLACKOUT_OFFSETS",
    "BLACKOUT_K_BY_OFFSET",
    "BLACKOUT_K_DISTRIBUTION",
    "BLACKOUT_K_VALUE_COUNTS",
    "BASELINE_A",
    "BASELINE_B",
    "BASELINE_C",
    "BASELINE_D",
    "PROMOTION_THRESHOLDS",
    "CLEAN_TRAIN_SPAN_START",
    "CLEAN_TRAIN_SPAN_END",
    "MISSING_TRAIN_MONTHS",
    "MISSING_TRAIN_MONTHS_PRE_2011",
    "MISSING_TRAIN_MONTHS_2011_ONWARD",
    "CALENDAR_MONTH_TEST_SHARE_PCT",
    "CALENDAR_MONTH_TRAIN_SHARE_PCT",
    "ACF_QUARTILE_AR1_PARAMS",
]

# ---------------------------------------------------------------------------
# Train/test period boundaries.
# Source: notebooks/02_forecastability.ipynb, Cell 104 (Experiment 7 §17.2);
# cross-checked against Cell 2 row counts and docs/DATA_DICTIONARY.md.
# ---------------------------------------------------------------------------
TRAIN_PERIOD_START = "2002-05-01"
TRAIN_PERIOD_END = "2015-08-01"
TEST_PERIOD_START = "2015-09-01"
TEST_PERIOD_END = "2018-12-01"

# ---------------------------------------------------------------------------
# The real 18 test calendar months, and which of them are FULL (fully
# observed, 0% masked) vs. BLACKOUT (99.58-99.97% masked).
# Source: notebooks/02_forecastability.ipynb, Cells 4/7/61 (Experiment 1 §2,
# Experiment 4 §11.2).
# ---------------------------------------------------------------------------
TEST_MONTHS = [
    "2015-09", "2016-01", "2016-02", "2016-03", "2016-06", "2016-07",
    "2016-08", "2016-09", "2016-12", "2017-01", "2017-02", "2017-03",
    "2017-04", "2017-05", "2017-06", "2018-07", "2018-11", "2018-12",
]

TEST_FULL_MONTHS = [
    "2015-09", "2016-01", "2016-06", "2016-12", "2018-07", "2018-11",
]

TEST_BLACKOUT_MONTHS = [
    "2016-02", "2016-03", "2016-07", "2016-08", "2016-09", "2017-01",
    "2017-02", "2017-03", "2017-04", "2017-05", "2017-06", "2018-12",
]

# Offsets in months from the first test month (2015-09 == offset 0). Source:
# Cell 61. TEST_FULL_OFFSETS[i] corresponds to TEST_FULL_MONTHS[i], and
# likewise for the BLACKOUT pair — order is preserved and tested.
TEST_FULL_OFFSETS = [0, 4, 9, 15, 34, 38]
TEST_BLACKOUT_OFFSETS = [5, 6, 10, 11, 12, 16, 17, 18, 19, 20, 21, 39]

# ---------------------------------------------------------------------------
# Staleness-to-target (k) for each real blackout month, keyed by its offset.
# k = months between the blackout month's target and the most recent FULL
# (fully-observed) month strictly before it. Source: Cell 62 (Experiment 4
# §11.2, the `stale_df` table).
# ---------------------------------------------------------------------------
BLACKOUT_K_BY_OFFSET: dict[int, int] = {
    5: 2, 6: 3, 10: 2, 11: 3, 12: 4, 16: 2, 17: 3, 18: 4, 19: 5, 20: 6,
    21: 7, 39: 2,
}

# The same k-values as a flat list, in TEST_BLACKOUT_OFFSETS order — this is
# the exact empirical distribution the masking simulator (step 2.4) resamples
# with replacement from, never a synthetic geometric/uniform prior.
BLACKOUT_K_DISTRIBUTION = [BLACKOUT_K_BY_OFFSET[offset] for offset in TEST_BLACKOUT_OFFSETS]

# Value-counts view (k -> how many of the 12 blackout months carry it).
BLACKOUT_K_VALUE_COUNTS: dict[int, int] = {2: 4, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1}

# ---------------------------------------------------------------------------
# Four baselines, computed on the real/replayed 18-month test structure.
# Source: notebooks/02_forecastability.ipynb, Cells 67/69/71 (Experiment 4
# §11.4-11.5). Baseline C is sourced from notebooks/01_eda.ipynb (only 3dp
# available there; not recomputed in notebook 02, per Cell 71's inline
# comment).
# ---------------------------------------------------------------------------
BASELINE_A = 0.5247  # Oracle persistence, FULL months only (n=747,365)
BASELINE_B = 0.7145  # Last-known-state, BLACKOUT months only (n=1,491,960)
BASELINE_C = 0.8170  # Seasonal climatology, all 18 months (notebook 01 EDA)
BASELINE_D = 0.6573  # Hybrid (A on FULL + B on BLACKOUT), all 18 months (n=2,239,325)

# Internal target ladder, recalibrated against the real floor (A-009).
# Source: docs/COMPETITIVE_ANALYSIS.md §6.
PROMOTION_THRESHOLDS: dict[str, float] = {
    "naive_floor": BASELINE_D,   # < 0.6573 clears the realistic naive floor
    "oracle_ceiling": 0.572,     # < 0.572 matches the in-sample oracle ceiling
    "beat_mohar": 0.559,         # < 0.559 beats MOHAR's public score (0.55958798)
    "serious_contender": 0.53,   # < 0.53 serious first-place contender
    "exceptional": 0.50,         # < 0.50 exceptional
}

# ---------------------------------------------------------------------------
# Verified gap-free training span (the anchor block for the blackout-curve
# scenario and the earliest CV fold's training portion). Source: Cell 38
# (Experiment 3 §9.1) — 84 consecutive calendar months, confirmed with zero
# gaps.
# ---------------------------------------------------------------------------
CLEAN_TRAIN_SPAN_START = "2004-01-01"
CLEAN_TRAIN_SPAN_END = "2010-12-01"

# ---------------------------------------------------------------------------
# The 22 missing training calendar months, split by documented cause.
# Source: Cells 104/110 (Experiment 7 §17.2/17.4).
# ---------------------------------------------------------------------------
MISSING_TRAIN_MONTHS_PRE_2011 = [
    "2002-06", "2002-07", "2002-08", "2003-06", "2003-07",
]

MISSING_TRAIN_MONTHS_2011_ONWARD = [
    "2011-01", "2011-02", "2011-06", "2011-07", "2012-05", "2012-06",
    "2012-10", "2012-11", "2013-03", "2013-04", "2013-08", "2013-09",
    "2013-10", "2014-02", "2014-03", "2014-07", "2014-08",
]

MISSING_TRAIN_MONTHS = MISSING_TRAIN_MONTHS_PRE_2011 + MISSING_TRAIN_MONTHS_2011_ONWARD

# ---------------------------------------------------------------------------
# Calendar-month coverage: share of TEST/TRAIN rows falling in each calendar
# month. Source: Cell 99 (Experiment 6 §15.4, `month_compare` table,
# .round(2) precision — the notebook does not display a higher-precision
# version).
# ---------------------------------------------------------------------------
CALENDAR_MONTH_TEST_SHARE_PCT: dict[str, float] = {
    "Jan": 11.13, "Feb": 11.14, "Mar": 11.16, "Apr": 5.57, "May": 5.54,
    "Jun": 11.09, "Jul": 11.10, "Aug": 5.52, "Sep": 11.06, "Oct": 0.00,
    "Nov": 5.57, "Dec": 11.13,
}

CALENDAR_MONTH_TRAIN_SHARE_PCT: dict[str, float] = {
    "Jan": 8.72, "Feb": 8.00, "Mar": 8.00, "Apr": 8.71, "May": 9.40,
    "Jun": 7.23, "Jul": 7.23, "Aug": 7.93, "Sep": 8.66, "Oct": 7.97,
    "Nov": 8.71, "Dec": 9.43,
}

# ---------------------------------------------------------------------------
# Per-ACF-quartile AR(1) parameters, used to reconstruct the theoretical
# degradation curve sigma*sqrt(2*(1-rho**k)) as the reference overlay in the
# degradation-slope diagnostic (Phase 2 step 2.7). Source: Cell 48
# (Experiment 5 §13, `quartile_summary` table).
# ---------------------------------------------------------------------------
ACF_QUARTILE_AR1_PARAMS: dict[str, dict[str, float]] = {
    "Q1_low_ACF": {"rho": 0.509673, "sigma": 0.830399},
    "Q2": {"rho": 0.733500, "sigma": 0.822421},
    "Q3": {"rho": 0.826161, "sigma": 0.829846},
    "Q4_high_ACF": {"rho": 0.928743, "sigma": 0.759810},
}
