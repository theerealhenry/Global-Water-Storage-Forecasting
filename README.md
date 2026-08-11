# TWS Forecast

Entry for Zindi/ITU AI for Good — *A Step Ahead of Drought: Forecasting Global Water Storage Challenge* (closes 2026-09-13).

**Start here:** `docs/ARCHITECTURE.md` is the project's single source of truth. `docs/PROJECT_PLAN.md` is the phase-by-phase roadmap; `docs/COMPETITIVE_ANALYSIS.md` holds strategy/hypotheses; `docs/ASSUMPTIONS.md` and `docs/OPEN_QUESTIONS.md` track what we believe vs. what's unresolved. Any deviation from the blueprint gets logged in `docs/adr/` — see `docs/adr/README.md`.

## Layout

- `data/raw/` — raw competition CSVs (gitignored) + `dataset_manifest.json` (integrity hashes)
- `docs/` — architecture, plan, analysis, ADRs
- `src/tws_forecast/` — package code (data, state, features, validation, models, pipelines, serving, utils)
- `configs/` — run configs
- `notebooks/` — exploratory notebooks
- `tests/` — unit/integration tests
- `submissions/` — submission log, candidate registry, submitted files, prediction manifests
- `artifacts/oof/` — out-of-fold prediction stores per candidate
- `reports/` — figures, final report
- `docker/`, `.github/workflows/` — deployment and CI

## Status

Project Phase 0 (foundation) in progress. Next: Project Plan Phase 1, experiment 2 (resolve 2015 persistence-RMSE anomaly).
