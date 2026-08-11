# Data Dictionary

Verified directly against `data/raw/Train.csv`, `data/raw/Test.csv`, and `data/raw/SampleSubmission.csv` (hashes pinned in `data/raw/dataset_manifest.json`). Every claim below was checked programmatically against the actual files, not assumed from the competition description.

## Grid and time coverage

- **Grid:** a fixed set of 15,715 land-only locations on an approximately 1-degree lat/lon grid (grid points end in `.5`, e.g. `-55.5, -68.5`), identical between `Train.csv` and `Test.csv` — confirmed by counting distinct `(lat, lon)` pairs in each file (15,715 in both). Latitude observed from -55.5 to 83.5 in a partial scan; longitude spans the full -179.5 to 179.5 range. Land-only and Northern-Hemisphere-weighted per `ARCHITECTURE.md` §7. **This 15,715-location completeness holds only in aggregate across the whole file, not per individual month, in EITHER file:** every one of the 18 test months is short of the full grid (38-195 locations absent as *rows*, not merely masked, per month — verified exhaustively in `notebooks/02_forecastability.ipynb` §2, Project Phase 1 Experiment 1). Experiment 3 (`notebooks/02_forecastability.ipynb` §9.1) found the same holds for `Train.csv`: even within the one fully gap-free 84-month span (2004-01 through 2010-12, no missing calendar months), individual months still range from 15,510 to 15,681 rows, never the full 15,715 — this is a general property of both raw files, not specific to the test set's masking mechanism. Code that assumes a given month has all 15,715 rows will be wrong in either file; `assert_full_grid()` in `src/tws_forecast/data/contracts.py` is deliberately only ever called against the complete file, never a single month, for exactly this reason.
- **Training period:** May 2002 through August 2015 (not January 2002 — verified by taking the min/max of the `time` column).
- **Test period:** 18 non-contiguous months from September 2015 through December 2018: 2015-09, 2016-01, 2016-02, 2016-03, 2016-06, 2016-07, 2016-08, 2016-09, 2016-12, 2017-01 through 2017-06, 2018-07, 2018-11, 2018-12. **The first test month, September 2015, is the calendar month immediately after the last month of training data (August 2015)** — this was not previously stated precisely in the project's documents (earlier drafts said "2016–2018") and is directly relevant to the unresolved 2015 persistence-RMSE anomaly (`ARCHITECTURE.md` §3, `docs/OPEN_QUESTIONS.md`).
- Row counts: 2,154,021 in `Train.csv`, 280,961 in `Test.csv`, matching `dataset_manifest.json` exactly.

## Train.csv — 13 columns

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | `{YYYYMMDD}_{lat}_{lon}` composite key, e.g. `20020501_-55.5_-68.5` |
| `time` | date | First of the observation month, `YYYY-MM-DD` |
| `lat` | float | Grid latitude |
| `lon` | float | Grid longitude |
| `TWS_t` | float | Total Water Storage anomaly at time `t`, standardized (both positive and negative values observed; not a raw physical unit) |
| `SPEI_01_t` | float | Standardised Precipitation-Evapotranspiration Index, 1-month timescale, at time `t` |
| `SPEI_03_t` | float | SPEI, 3-month timescale |
| `SPEI_06_t` | float | SPEI, 6-month timescale |
| `SPEI_12_t` | float | SPEI, 12-month timescale |
| `SOIL_MOISTURE_t` | float | Soil moisture at time `t`, standardized |
| `month_sin` | float | `sin` component of a cyclical calendar-month encoding |
| `month_cos` | float | `cos` component of the same cyclical calendar-month encoding |
| `target` | float | TWS at time `t+1` (next calendar month) — **verified directly**: for a given location, `target` at month `t` equals `TWS_t` at the row for month `t+1` whenever both rows exist. Some locations have gaps in monthly cadence in `Train.csv` itself (most visible in the earliest months of the record, consistent with GRACE's early, less regular sampling), so a naive "next row" check can look like a mismatch when the actual next *calendar* month's row is absent — the target is defined by calendar month, not by row adjacency. |

`Train.csv` has no explicit masking-indicator column — every row's `TWS_t` is a real observed value; masking is a test-set-only phenomenon in this dataset.

## Test.csv — 13 columns

Same as `Train.csv` except: `sample_id` is named `ID`, there is no `target` column (withheld), and the final column is `TWS_t_masked` (boolean `True`/`False`) instead of `target`.

`TWS_t_masked` is an exact, verified indicator of missingness: it is `True` if and only if `TWS_t` is blank for that row, checked against all 280,961 rows with zero mismatches. 186,913 of 280,961 rows (66.5%) have `TWS_t_masked = True`, matching the documented "~66% of test rows have no current TWS observation" finding exactly. When `TWS_t_masked` is `False`, `TWS_t` holds a real value in the same standardized units as `Train.csv`; when `True`, `TWS_t` is blank and must be reconstructed from history per the state-reconstruction architecture (`ARCHITECTURE.md` §3–§4).

All other columns (`time`, `lat`, `lon`, the four `SPEI_*_t` columns, `SOIL_MOISTURE_t`, `month_sin`, `month_cos`) are always populated in `Test.csv`, including on masked rows — only `TWS_t` itself goes missing.

## SampleSubmission.csv — 2 columns

| Column | Type | Description |
|---|---|---|
| `ID` | string | Matches `Test.csv`'s `ID` column exactly, same `{YYYYMMDD}_{lat}_{lon}` format |
| `Target` | float | Placeholder (`0` in the sample file) — this is the column a submission file must populate with the predicted next-month TWS |

## Open items

Exact numeric ranges for `TWS_t`, the SPEI variables, and `SOIL_MOISTURE_t` (min/max/std per variable, and whether they're globally standardized vs. standardized per location) are not yet computed — that's Project Phase 1 EDA work, not a Phase 0 documentation task, and belongs in `notebooks/01_eda.ipynb` once written.
