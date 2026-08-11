# Experiment log

`experiment_log.csv` is the flat, queryable record of every experiment run during research (Project Phase 1 onward) — distinct from `submissions/submission_log.csv`, which tracks only what was actually uploaded to Zindi. Not every experiment becomes a submission; every experiment should get a row here.

This is the required Project Phase 0 "minimal reproducible foundation" logging mechanism (`PROJECT_PLAN.md` Project Phase 0), and stays live even after MLflow is fully wired up in Project Phase 11 — MLflow tracks run-level parameters/artifacts/metrics in detail, this stays as the flat, human-skimmable summary that's fast to open and diff in git.

## Columns

| Column | Meaning |
|---|---|
| `experiment_id` | Short unique identifier, e.g. `E001` |
| `experiment_name` | Short human-readable label |
| `timestamp` | When the experiment was run |
| `git_commit` | Commit hash the experiment ran against |
| `data_version` | Reference to `data/raw/dataset_manifest.json`'s hash, or a note if using a derived/processed dataset version |
| `training_cutoff` | Latest date included in the training fold for this experiment |
| `blackout_scenario_id` | Reference to a `configs/validation/` `MaskingScenario` ID (`ARCHITECTURE.md` §8), or `n/a` for experiments that don't use synthetic masking |
| `model` | Model family/config used |
| `seed` | Random seed |
| `cv_tier1_rmse`, `cv_tier2_rmse`, `cv_tier3_rmse` | Cross-validated RMSE per validation tier (`ARCHITECTURE.md` §11) — blank if not applicable to that experiment (e.g. a pure EDA/forensics experiment with no model) |
| `notes` | What was learned, in one line |

Every row in Project Phase 1's ordered experiment sequence gets an entry here, even the ones that aren't primarily about a model (e.g. the masking-process reproduction, the 2015 anomaly investigation) — `cv_*_rmse` columns can be left blank and `notes` used to record the finding instead.
