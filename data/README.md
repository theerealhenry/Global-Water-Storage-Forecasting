# Data

`Train.csv`, `Test.csv`, `SampleSubmission.csv` are the official Zindi competition files, kept in `data/raw/`. They are gitignored (`data/raw/*.csv`) — never commit raw data.

Integrity is pinned in `data/raw/dataset_manifest.json` (SHA256 + row counts) per ADR-0002. If you re-download the data and the hashes don't match, stop and log an ADR before proceeding — the dataset may have been updated.

Derived or intermediate data goes in `data/processed/` or `data/interim/` (both gitignored), keeping the raw/derived boundary intact — nothing under `data/raw/` is ever modified in place.
