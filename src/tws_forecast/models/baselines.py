"""Project Phase 3 — six state-aware reference predictors.

Four *distinct* baselines (``docs/PROJECT_PLAN.md`` Project Phase 3), kept
deliberately separate rather than collapsed into one baseline with a
fallback bolted on, because conflating them obscures what's actually being
measured — plus a global mean and a Ridge regression as further reference
points:

- :class:`GlobalMeanPredictor` — the absolute floor.
- :class:`OraclePersistencePredictor` (**Baseline A**) — how hard is the
  unmasked problem, on its own?
- :class:`LastKnownStatePredictor` (**Baseline B**) — how far can historical
  state reconstruction alone get us, with *zero* use of the current
  observation, even when it exists?
- :class:`SeasonalClimatologyPredictor` (**Baseline C**) — per-location,
  per-calendar-month climatology.
- :class:`HybridPersistencePredictor` (**Baseline D**) — the realistic
  "no ML at all" floor for the real test structure (current observation when
  available, last-known-state otherwise). Promoted near-verbatim from the
  throwaway ``BaselineDPredictor`` in ``notebooks/03_validation_harness.ipynb``
  (Project Phase 2 step 2.11's proof-run stand-in) into real, tested,
  importable ``src/`` code, per the Project Phase 3 handoff §3.1.
- :class:`RidgeBaselinePredictor` — a thin linear reference point.

Every class implements ``validation.tiers.Predictor`` (``fit``/``predict``)
so each plugs directly into ``harness.evaluate_candidate()`` with no
special-casing, never mutates its input frame, and always returns a plain
``np.ndarray`` with no NaNs — the harness calls every candidate this way
across all three tiers (Tier 1: no masking; Tier 2/3: some rows have
``TWS_t`` nulled).

**Baseline A vs. B, precisely — the distinction the Project Phase 3 handoff
flags as easy to get subtly wrong:** :class:`OraclePersistencePredictor` and
:class:`LastKnownStatePredictor` are not "the same logic with the masked
branch toggled." ``OraclePersistencePredictor`` reads a row's *own*
``TWS_t`` at predict time — its whole definition is "how good is the current
observation, when we have it." ``LastKnownStatePredictor`` never reads the
predict-time frame's ``TWS_t`` at all, even on rows where it happens to be
populated (e.g. a Tier-1 call, which never masks) — it only ever answers from
a ``{location_id: last_observed_TWS_t}`` dictionary built once, at ``fit()``
time, from history strictly before the predict call. This is a materially
different (and deliberately weaker) predictor than
:class:`HybridPersistencePredictor`, whose own ``predict()`` legitimately
forward-fills *within* the frame it's given (see that class's docstring) —
a naive copy of Baseline D's logic with only the "use TWS_t when observed"
branch removed would still let today's real observations leak into
tomorrow's history-only answer, which is exactly what Baseline B exists to
rule out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from tws_forecast.validation.splitters import attach_forecast_origin_columns

__all__ = [
    "GlobalMeanPredictor",
    "OraclePersistencePredictor",
    "LastKnownStatePredictor",
    "SeasonalClimatologyPredictor",
    "HybridPersistencePredictor",
    "RidgeBaselinePredictor",
]

# The raw numeric feature columns available on every row of Train.csv/
# Test.csv (docs/DATA_DICTIONARY.md). TWS_t is deliberately excluded here —
# RidgeBaselinePredictor decides per-regime whether to include it (see that
# class's docstring).
_ENV_FEATURE_COLUMNS = [
    "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
    "SOIL_MOISTURE_t", "month_sin", "month_cos",
]


def _with_location_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``location_id`` (and the rest of ``FORECAST_ORIGIN_COLUMNS``)
    if not already present, without mutating the caller's frame.

    Every baseline in this module goes through this helper rather than
    re-deriving ``location_id`` its own way, for the same reason
    ``notebooks/03_validation_harness.ipynb``'s ``BaselineDPredictor`` did:
    one canonical code path
    (``state.reconstruction.location_id_from_lat_lon``, via
    ``validation.splitters.attach_forecast_origin_columns``).
    """
    return df if "location_id" in df.columns else attach_forecast_origin_columns(df)


def _target_calendar_month(df: pd.DataFrame) -> pd.Series:
    """The calendar month (1-12) of the *target* — i.e. next month, not the
    row's own month — used by :class:`SeasonalClimatologyPredictor`.

    Uses ``target_time`` if the frame already carries
    ``FORECAST_ORIGIN_COLUMNS``; otherwise derives it from ``time + 1
    month``. Precisely re-deriving "which month should climatology key off"
    matters here: Phase 1's original 0.817 measurement
    (``notebooks/01_eda.ipynb``) used per-location-per-calendar-month mean of
    the *target* itself, i.e. the month the prediction is *for*, not the
    month the row was observed in.
    """
    if "target_time" in df.columns:
        target_time = pd.to_datetime(df["target_time"])
    else:
        target_time = pd.to_datetime(df["time"]) + pd.DateOffset(months=1)
    return target_time.dt.month


class GlobalMeanPredictor:
    """The absolute floor reference: predict the training set's mean
    ``target`` for every row, regardless of any covariate.

    Phase 1 measured this at approximately 0.912 in-sample
    (``docs/PROJECT_PLAN.md``'s "Key findings" section, `ARCHITECTURE.md`
    §3). Deterministic given a fitted frame — no randomness, so nothing to
    seed.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._mean = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean, dtype=float)


class OraclePersistencePredictor:
    """**Baseline A** — ŷ = TWS_t, read directly from each row at predict
    time. Only *meaningful* on rows where ``TWS_t`` is actually observed —
    that's this baseline's whole definition, "how hard is the unmasked
    problem." Its real answer is its **Tier 1** number (Tier 1 never masks)
    and Tier 2/3's ``regime=observed`` decomposition slice.

    **Explicit fallback policy** (required per the Project Phase 3 handoff
    §3.1, since ``harness.run_tier2``/``run_tier3`` call ``predict()`` on
    masked rows too, where ``TWS_t`` is ``NaN``): masked rows fall back to
    the fitted global mean of ``target`` — the same floor
    :class:`GlobalMeanPredictor` uses. This keeps the harness from crashing,
    but Baseline A's masked-regime numbers are a fallback artifact, not the
    quantity this baseline exists to measure — do not read Baseline A's
    Tier 2/3 *masked*-regime RMSE as a real result in any report.
    """

    def __init__(self) -> None:
        self._fallback: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self._fallback = float(train_df["target"].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        tws_t = df["TWS_t"].to_numpy(dtype=float, copy=True)
        missing = np.isnan(tws_t)
        if missing.any():
            tws_t[missing] = self._fallback
        return tws_t


class LastKnownStatePredictor:
    """**Baseline B** — how far can last-known-state alone get us, with
    *zero* use of the current observation, even when it exists?

    ``fit()`` builds a ``{location_id: last_observed_TWS_t}`` dictionary
    from the training fold (sorted by time, last value per location), plus
    a global-mean-of-target fallback for any location never observed in
    that training history. ``predict()`` looks up each row's location in
    that fixed, fit-time dictionary — it never reads the predict-time
    frame's own ``TWS_t`` column at all, unlike
    :class:`HybridPersistencePredictor`, whose ``predict()`` legitimately
    forward-fills using whatever real observations appear earlier *within*
    the frame it's given. See this module's top-level docstring for why
    that distinction is deliberate and easy to get subtly wrong.
    """

    def __init__(self) -> None:
        self._last_known: dict[str, float] = {}
        self._fallback: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        df = _with_location_id(train_df)
        self._fallback = float(df["target"].mean())
        observed = df.dropna(subset=["TWS_t"]).sort_values("time")
        self._last_known = observed.groupby("location_id")["TWS_t"].last().to_dict()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        location_id = _with_location_id(df)["location_id"]
        preds = location_id.map(self._last_known)
        preds = preds.fillna(self._fallback)
        return preds.to_numpy(dtype=float)


class SeasonalClimatologyPredictor:
    """**Baseline C** — independent fallback, per-location per-calendar-month
    mean of ``target``. Phase 1 measured this at 0.817 in-sample
    (``notebooks/01_eda.ipynb``) — weaker than intuition suggests, meaning
    most error in this problem is within-location month-to-month deviation,
    not baseline-level miscalibration.

    ``fit()`` groups by ``(location_id, target_calendar_month)`` — the
    calendar month of ``target`` itself (i.e. next month, not the row's own
    month; see :func:`_target_calendar_month`), matching Phase 1's original
    measurement precisely rather than guessing. ``predict()`` looks up the
    same key, falling back to the fitted global mean of ``target`` for any
    ``(location, month)`` combination never seen in training.
    """

    def __init__(self) -> None:
        self._climatology: dict[tuple[str, int], float] = {}
        self._fallback: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        df = _with_location_id(train_df).copy()
        self._fallback = float(df["target"].mean())
        df["_target_month"] = _target_calendar_month(df)
        grouped = df.groupby(["location_id", "_target_month"])["target"].mean()
        self._climatology = {(loc, int(month)): val for (loc, month), val in grouped.items()}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        work = _with_location_id(df)
        target_month = _target_calendar_month(work)
        keys = list(zip(work["location_id"], target_month.astype(int), strict=True))
        preds = np.array(
            [self._climatology.get(key, self._fallback) for key in keys],
            dtype=float,
        )
        return preds


class HybridPersistencePredictor:
    """**Baseline D** — if ``TWS_t`` is observed for a row, use it; else use
    last-known-state; else (a never-observed location) fall back to the
    fitted global mean. The realistic "no ML at all" floor for the actual
    test structure — Phase 1 measured 0.6573 on the real/replayed test
    structure (A-009), and Project Phase 2's proof run
    (``notebooks/03_validation_harness.ipynb``) already confirmed this exact
    logic reproduces that number through the validation harness (with the
    A-013 caveat on Tier 3's row-wise scoring — see this module's Tier 3
    handling note below).

    Promoted near-verbatim from the throwaway ``BaselineDPredictor`` defined
    inline in that notebook's build script, per the Project Phase 3 handoff
    §3.1's explicit instruction not to reimplement this logic a second time.

    ``fit()`` records each location's last observed ``TWS_t`` (by time) from
    the training data, plus a global mean fallback. ``predict()``
    forward-fills each row's own ``TWS_t`` at/before its own time *within
    the predict() call's own frame* — so a masked run spanning an entire
    validation window still resolves to the correct last-known value instead
    of NaN — seeded by ``fit()``-time history for locations with no
    in-frame observation yet. Vectorized (groupby + ffill), not a per-row
    Python loop.

    Note this is precisely the behavior :class:`LastKnownStatePredictor`
    (Baseline B) deliberately does *not* replicate: this predictor legitimately
    uses real, earlier-in-frame observations (e.g. an earlier FULL month in
    the same Tier 3 replay window) to answer a later masked row, because that
    is what "last known value" means for a genuine hybrid persistence
    baseline. A-013 (``docs/ASSUMPTIONS.md``) documents that
    ``validation.tiers.run_tier3`` under-scores this kind of internally
    stateful, non-feature-based predictor anyway, since it calls
    ``predict()`` independently per replay offset rather than letting state
    accumulate across the replay pattern the way a real deployed forecaster
    would — see the Project Phase 3 notebook's handling of that caveat
    (step 3.0 of the handoff) before reading this class's Tier 3 number as
    directly comparable to Phase 1's 0.6573.
    """

    def __init__(self) -> None:
        self._last_known: dict[str, float] = {}
        self._fallback: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        df = _with_location_id(train_df)
        self._fallback = float(df["TWS_t"].mean())
        observed = df.dropna(subset=["TWS_t"]).sort_values("time")
        self._last_known = observed.groupby("location_id")["TWS_t"].last().to_dict()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        work = _with_location_id(df).sort_values(["location_id", "time"])
        filled = work.groupby("location_id")["TWS_t"].ffill()
        still_na = filled.isna()
        if still_na.any():
            fallback_vals = work.loc[still_na, "location_id"].map(self._last_known).fillna(self._fallback)
            filled.loc[still_na] = fallback_vals
        return filled.reindex(df.index).to_numpy(dtype=float)


class RidgeBaselinePredictor:
    """A thin linear reference point: ``sklearn.linear_model.Ridge`` fit on
    the raw available numeric columns.

    **Design decision, stated explicitly per the Project Phase 3 handoff
    §3.1** (masked rows have ``TWS_t = NaN``, so a single model can't use it
    as a feature uniformly): this class fits **two internal Ridge models**
    rather than dropping ``TWS_t`` from the feature set entirely. The
    "observed" model is trained only on rows with a real ``TWS_t`` and uses
    ``[TWS_t, SPEI_01_t, SPEI_03_t, SPEI_06_t, SPEI_12_t, SOIL_MOISTURE_t,
    month_sin, month_cos]``; the "masked" model is trained on *every* row
    (``TWS_t`` dropped) using the remaining six environmental/calendar
    columns, so it doesn't waste the (larger, always-available) training
    signal on only the rows that happen to be masked in some *other*
    dataset's regime. Both are fit at ``fit()`` time from the same
    ``train_df``. At ``predict()`` time, each row is routed to whichever
    internal model matches its own observed/masked status.

    Why this choice over a single TWS_t-free model: dropping ``TWS_t``
    entirely would throw away the single strongest predictor this dataset
    has (Baseline A's persistence result) on the two-thirds of rows where
    it's actually available, understating what a linear model can do in the
    observed regime. The cost is that this baseline's Tier 2/3
    ``regime=observed`` vs. ``regime=masked`` decomposition rows are
    produced by two different fitted models, not one model evaluated
    uniformly — keep that firmly in mind when interpreting the regime
    split for this specific baseline (it is not read the same way as, say,
    a GBM's regime split in Project Phase 5+, where a single model with
    native NaN handling sees both regimes).
    """

    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        # Ridge itself is a closed-form solver (no stochastic fitting), so
        # `seed` has nothing to randomize today -- kept as an explicit,
        # accepted parameter (rather than silently absent) per this
        # project's standing seed-everywhere discipline (utils/seeds.py),
        # so this class doesn't need a signature change if a future variant
        # (e.g. an SGD-based Ridge) needs it to do something.
        self._alpha = alpha
        self._seed = seed
        self._observed_model = Ridge(alpha=alpha)
        self._masked_model = Ridge(alpha=alpha)
        self._fallback: float = 0.0

    @staticmethod
    def _observed_features(df: pd.DataFrame) -> pd.DataFrame:
        return df[["TWS_t", *_ENV_FEATURE_COLUMNS]]

    @staticmethod
    def _masked_features(df: pd.DataFrame) -> pd.DataFrame:
        return df[_ENV_FEATURE_COLUMNS]

    def fit(self, train_df: pd.DataFrame) -> None:
        self._fallback = float(train_df["target"].mean())

        observed_rows = train_df.dropna(subset=["TWS_t"])
        if len(observed_rows) > 0:
            self._observed_model.fit(
                self._observed_features(observed_rows), observed_rows["target"]
            )
        else:
            # Degenerate but real possibility for a tiny/synthetic fixture --
            # never let fit() raise; predict() falls back to the global mean
            # for any row that would have used this model.
            self._observed_model = None  # type: ignore[assignment]

        self._masked_model.fit(self._masked_features(train_df), train_df["target"])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(df), self._fallback, dtype=float)
        is_observed = df["TWS_t"].notna().to_numpy()

        if is_observed.any() and self._observed_model is not None:
            observed_slice = df.loc[is_observed]
            preds[is_observed] = self._observed_model.predict(
                self._observed_features(observed_slice)
            )

        if (~is_observed).any():
            masked_slice = df.loc[~is_observed]
            preds[~is_observed] = self._masked_model.predict(
                self._masked_features(masked_slice)
            )

        return preds
