# Golden test fixtures

`Train.csv`, `Test.csv`, `SampleSubmission.csv` in this directory are small, deterministic subsets of the real competition data — the first 10 unique `(lat, lon)` locations by sort order, with every row belonging to those locations kept (all months). Sizes: 1,380 train rows, 180 test rows, 180 sample-submission rows (120 of the 180 test rows are masked, 66.7% — consistent with the full dataset's masking rate).

Generated directly from the real, hash-verified `data/raw/*.csv` files via `pd.merge` on location, sorted by `(lat, lon, time)`, so relationships within a location's time series are preserved exactly as they appear in the real data — this is a genuine subset, not synthetic data.

Used by `tests/test_loaders.py` and `tests/test_contracts.py` for fast, deterministic tests that don't require the full ~330MB of raw data to be present. Deliberately small enough to be committed to git (unlike `data/raw/`, which is gitignored).

Note: `tws_forecast.data.contracts.assert_full_grid()` will correctly reject this fixture (it has 10 locations, not the full 15,715) — that's expected and is itself tested in `tests/test_contracts.py`, not a bug in the fixture.
