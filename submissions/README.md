# Submission subsystem

`submission_log.csv` is the append-only registry of every leaderboard submission this project makes. Every row is produced by `python -m tws_forecast.pipelines.submit --config <config>`, never edited by hand except to fill in `public_lb_score` once Zindi returns it (and `selected_for_private` for the two final choices before close).

## Schema

| Column | Meaning |
|---|---|
| `submission_id` | Unique ID (e.g. `sub-0007`), also the filename stem in `files/` |
| `timestamp` | UTC, generation time |
| `git_commit` | Exact commit the model/config was built from — reproducibility requirement |
| `project_phase` | Which `PROJECT_PLAN.md` phase produced this (e.g. `5`) |
| `champion_ladder_level` | Which `COMPETITIVE_ANALYSIS.md` §8 ladder rung this represents (e.g. `6 - state-reconstruction GBM`) |
| `model_name` / `config_path` | What was run |
| `cv_tier1_rmse` / `cv_tier2_rmse` / `cv_tier3_rmse` | The three validation tiers (forecastability / blackout / test-regime) |
| `cv_overall_rmse` / `cv_masked_rmse` / `cv_unmasked_rmse` | Full decomposition, not just the headline number |
| `degradation_slope` | ΔRMSE / Δmonths-since-observation from this model's blackout-tier evaluation |
| `hypothesis_being_tested` | One sentence — what this submission is actually testing, not just "another model" |
| `submission_file_path` | Path under `files/` |
| `uploaded_to_zindi` | Whether it was actually submitted (budget guard may generate-but-not-submit) |
| `public_lb_score` | Filled in once known — the running check on whether our CV is honest (§ARCHITECTURE.md §6) |
| `selected_for_private` | True for exactly the two submissions chosen for private-leaderboard judging, per the error-correlation-based selection in `COMPETITIVE_ANALYSIS.md` §13 |

## Budget

5 submissions/day, 200 total (Zindi rule). `submit.py` checks both against this log before generating a new file and refuses rather than silently spending a slot.

## Files

`files/<submission_id>.csv` — the actual Zindi-format upload (`ID`, `Target`), schema-validated against `SampleSubmission.csv` before being written.
