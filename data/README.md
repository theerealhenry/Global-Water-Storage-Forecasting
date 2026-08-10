# Data

`Train.csv`, `Test.csv`, `SampleSubmission.csv` are the official Zindi competition files. They are gitignored (`data/*.csv`) — never commit raw data.

Deviation note: `ARCHITECTURE.md` §4/§5 originally specified `data/raw/`. Since these files were placed directly under `data/` locally and are large (Train.csv is ~289MB), they're kept here rather than duplicated into a nested `raw/` folder. This is a cosmetic path difference only — no scripts should assume `data/raw/`; treat `data/` as the raw-data root. Any derived/processed data still goes in `data/processed/` (gitignored) to keep the raw/derived boundary intact.

Integrity is pinned in `dataset_manifest.json` (SHA256 + row counts) per ADR-0002. If you re-download the data and the hashes don't match, stop and log an ADR before proceeding — the dataset may have been updated.
