"""Shared month-index arithmetic.

Several modules (``validation/splitters.py``, ``validation/tiers.py``) need
to move calendar months forward/backward by simple integer offsets rather
than repeated ``pd.DateOffset`` arithmetic — this is the one place that
logic lives, so those modules can't drift into two subtly different
definitions of "how many months apart are these two dates."
"""

from __future__ import annotations

import pandas as pd

__all__ = ["month_index", "month_index_to_timestamp"]


def month_index(ts: pd.Timestamp | str) -> int:
    """Months since year 0, as a plain integer (``year * 12 + month``).

    Only year and month matter — day-of-month is dropped, since every
    ``time`` value in the raw data is already a month-start
    (``docs/DATA_DICTIONARY.md``) and fold/window boundaries are always
    whole calendar months.
    """
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month


def month_index_to_timestamp(idx: int) -> pd.Timestamp:
    """Inverse of ``month_index`` — always returns the first of the month."""
    year, month = divmod(idx, 12)
    if month == 0:
        year -= 1
        month = 12
    return pd.Timestamp(year=year, month=month, day=1)
