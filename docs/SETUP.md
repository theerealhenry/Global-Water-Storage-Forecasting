# Local Environment Setup

Conda creates the isolated Python 3.11 interpreter only; pip installs every actual package
from pinned `requirements.txt` / `requirements-dev.txt`. See `docs/adr/0003-dependency-management-strategy.md`
for why (single resolver, avoids conda-forge/pip version drift).

Run these from a terminal (Anaconda Prompt, or a regular terminal with conda on PATH) at the
project root, `D:\PROJECTS\tws-forecast`.

## 1. Create the environment

```bash
conda create -n tws-forecast python=3.11 pip -c conda-forge
```

Confirm with `y` when prompted. This only installs Python and pip — nothing else yet.

## 2. Activate it

```bash
conda activate tws-forecast
```

Your terminal prompt should now show `(tws-forecast)` at the start of the line.

## 3. Install pinned dependencies

```bash
pip install -r requirements-dev.txt
```

This installs everything in `requirements.txt` (numpy, pandas, scikit-learn, lightgbm, xgboost,
mlflow, shap, optuna, fastapi, streamlit, etc. — all exact-pinned versions) plus dev tooling
(jupyterlab, pytest, black, ruff, pre-commit). Expect this to take several minutes — several of
these packages (scipy, pyarrow, lightgbm, xgboost) are large binary wheels.

## 4. Verify

```bash
python -c "import numpy, scipy, pandas, sklearn, lightgbm, xgboost, mlflow, shap, optuna, fastapi, streamlit; print('all imports ok')"
python -c "import numpy, pandas, sklearn; print(numpy.__version__, pandas.__version__, sklearn.__version__)"
pytest --version
mlflow --version
```

If any import fails, stop and paste the exact error — don't try to work around it by upgrading
one package by hand, since that's exactly the kind of drift the pinned versions in
`requirements.txt` were chosen to prevent (see ADR-0003 for the specific conflicts already found
and resolved, e.g. mlflow forces pandas below 3.0).

## 5. Install the project package in editable mode

Once `src/tws_forecast/` has a `pyproject.toml` (Project Phase 0 foundation work — not yet
created), run:

```bash
pip install -e .
```

so imports work consistently across notebooks, scripts, and tests.

## 6. Git

This folder isn't a git repo yet. Initialize it yourself, locally, so you keep control of the
remote/branch setup (conda/pip steps above don't touch git):

```bash
git init
git add .
git commit -m "chore: project skeleton (docs, ADRs, submissions subsystem, src layout, pinned deps)"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Updating dependencies later

Edit `requirements.txt` (or `requirements-dev.txt` for dev-only tools), then:

```bash
pip install -r requirements-dev.txt --upgrade
```

Re-run the verification step afterward. If you add a genuinely new package (not just a version
bump), check it against the compatibility notes in ADR-0003 first — numpy's version ceiling
(driven by numba, which SHAP depends on) and mlflow's pandas ceiling are the two tightest
constraints in this stack and the most likely to break silently.

## Why not just `environment.yml` with everything in it?

`environment.yml` still exists (just `python=3.11` + `pip` now) so conda has something to solve,
but every actual package lives in `requirements.txt`/`requirements-dev.txt`. This keeps one
dependency resolver (pip) instead of two (conda-forge and pip solving different halves of the
graph independently), and means the same `requirements.txt` can build the CI/Docker image later
without needing conda there at all.
