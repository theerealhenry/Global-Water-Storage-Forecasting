# ADR-0003 — Dependency Management Strategy: Conda-for-Interpreter + Pip-for-Packages, Pinned Version Set

**Status:** Accepted
**Date:** 2026-08-10
**Category:** Deployment
**Deciders:** Steve, Claude

## Context

The project uses miniconda locally (Steve's stated tooling) but also needs CI/CD (GitHub Actions) and a deployment image (Docker) that don't need the full conda distribution. `ARCHITECTURE.md` requires MLflow, a GBM-based modeling stack, SHAP for interpretability (feeds the Trustworthiness Evaluation, Project Phase 9), and a FastAPI/Streamlit serving layer — several of these packages (numpy, numba, llvmlite, shap, mlflow) have historically tight, easy-to-violate cross-version pins. Steve explicitly asked for a thorough compatibility check up front, to avoid discovering a broken environment mid-project.

## Evidence

Checked PyPI release metadata (`requires_dist`, `requires_python`, per-release upload history) for the full candidate stack on 2026-08-10, targeting Python 3.11 on Windows. Key findings:

- **numpy 2.5.x, scipy 1.18.x, and xgboost 3.3.x+ have already dropped Python 3.11 support** (`requires_python>=3.12`) as of their latest releases. The last 3.11-compatible releases are numpy 2.4.6, scipy 1.17.1, xgboost 3.2.0.
- **numba 0.66.0 requires `numpy<2.5,>=1.22`** — numpy 2.4.6 fits; anything newer (2.5.x) would break numba, and therefore break SHAP (which depends on numba on most platforms).
- **shap's latest release (0.52.0) requires Python >=3.12** — not installable on 3.11 at all. The prior release, shap 0.51.0, requires `>=3.11` and was used instead.
- **Real conflict found: mlflow 3.15.1 pins `pandas<3`**, but the latest Python-3.11-compatible pandas is 3.0.5 (a pandas 3.x major release). Pinning pandas 3.0.5 would make `pip install mlflow` unsatisfiable. Resolved by pinning pandas to **2.3.3** (latest 2.x release, still satisfies scikit-learn's `numpy>=1.24.1` and streamlit's `pandas<4,>=1.4.0`).
- **streamlit pins `pyarrow<25`**, but the latest pyarrow (25.0.1) has already moved past that. Pinned pyarrow to **24.0.0** instead.
- Cross-checked lightgbm, xgboost, pyarrow, numba, llvmlite, shap wheel availability for `win_amd64` on Python 3.11 — all confirmed present (lightgbm/xgboost ship `py3-none-win_amd64`, universal across CPython 3.x minors; the rest ship `cp311-win_amd64` wheels directly).
- Did not individually re-verify every transitive sub-dependency (e.g., mlflow-skinny/mlflow-tracing's own pins, starlette's exact range) — scoped the check to the packages with known history of tight cross-pins (numpy/numba/llvmlite/shap/scipy/scikit-learn/pandas/pyarrow/mlflow), which is where the real-world breakage risk concentrates.
- Could not empirically test conda-forge solving directly — `conda.anaconda.org` / `repo.anaconda.com` are not reachable from the verification sandbox. `pypi.org` was reachable, so verification was done against pip/PyPI metadata directly.

## Current architecture

Previously specified a single `environment.yml` with unpinned/loosely-pinned conda-forge + pip dependencies (no version numbers), created before this compatibility check.

## Decision

1. **Conda is used only to create the isolated Python 3.11 interpreter** (`environment.yml` now contains just `python=3.11` and `pip`).
2. **All actual package versions are pinned in `requirements.txt` (runtime) and `requirements-dev.txt` (adds jupyterlab/pytest/black/ruff/pre-commit)**, installed via pip inside that conda env. This is the single dependency resolver and single source of truth for versions.
3. Exact pins as verified above: numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.0, pyarrow 24.0.0, lightgbm 4.7.0, xgboost 3.2.0, mlflow 3.15.1, optuna 4.9.0, shap 0.51.0, numba 0.66.0, llvmlite 0.48.0, fastapi 0.141.1, uvicorn 0.52.1, pydantic 2.13.4, streamlit 1.61.1, matplotlib 3.11.1, seaborn 0.13.2, pyyaml 6.0.3, python-dotenv 1.2.2, tqdm 4.70.0.

## Reason

Splitting packages across conda-forge and pip (the originally-proposed `environment.yml`) risks the two ecosystems disagreeing about what's "latest" or compatible — conda-forge and PyPI don't always publish the same version at the same time, and solving half the graph with each resolver independently can silently produce two different, both-locally-"successful" environments. Using pip as the single resolver for the whole package graph means the exact set above is testable and reproducible one way, and the same `requirements.txt` can build a slim Docker image later (Project Phase 13) without needing conda in CI/deployment at all — only local dev needs conda, purely for interpreter management.

## Alternatives considered

- **Pure `environment.yml` with all packages via conda-forge:** rejected — could not be empirically verified (conda-forge endpoints unreachable from the verification environment), and conda-forge lags PyPI on some of these packages (notably shap, optuna), risking stale or unavailable builds.
- **Pure `environment.yml` with pip: section for everything (original draft):** rejected — same single-resolver benefit is achievable more simply and more portably (for CI/Docker) via plain `requirements.txt`, without needing conda's `pip:` sub-resolution step.
- **pandas 3.0.5 (latest 3.11-compatible) with mlflow pinned to an older release that might allow it:** not pursued — no mlflow release currently supports pandas 3.x; would mean giving up recent MLflow features for no real benefit.

## Consequences

- Local dev: `conda create -n tws-forecast python=3.11 pip -c conda-forge`, then `pip install -r requirements-dev.txt`.
- CI/Docker: plain `pip install -r requirements.txt` on a `python:3.11-slim` base, no conda dependency at all — smaller, faster images.
- Version bumps (e.g., when mlflow eventually supports pandas 3.x) should be re-verified the same way before changing pins, not assumed compatible.

## Risks

- Pins will go stale as the competition (closes 2026-09-13) and later portfolio work continues; if a new package is added later (e.g., a deep-learning rung per `COMPETITIVE_ANALYSIS.md`), it must be checked against this same chain, especially numpy's numba ceiling.
- Transitive sub-dependencies of mlflow-skinny/mlflow-tracing/starlette were not individually re-verified (see Evidence) — low risk given they're bundled/version-locked with mlflow and fastapi/streamlit respectively, but not a zero-risk gap.

## Validation

`pip install -r requirements-dev.txt` inside a fresh `tws-forecast` conda env completes without resolver conflicts, and `python -c "import numpy, scipy, pandas, sklearn, lightgbm, xgboost, mlflow, shap, optuna, fastapi, streamlit; print('ok')"` succeeds.

## Affected components

- [x] deployment
- [x] documentation
- [ ] data
- [ ] validation
- [ ] features
- [ ] modeling

## Related

- Experiments: none (tooling decision, not a modeling experiment)
- MLflow runs: n/a
- Submissions: n/a
- Supersedes: none
- Superseded by: none
- Related ADRs: none

## Follow-up actions

`ARCHITECTURE.md` §4 (repo/module map) and `docs/SETUP.md` updated to reflect the conda-for-interpreter + pip-for-packages split — confirmed.
