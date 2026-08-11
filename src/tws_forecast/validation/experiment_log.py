"""Experiment logging: the flat CSV (Phase 0) and the MLflow tracking
backend (this step) as two views onto the same event, written by one call
so they can never silently drift apart.

``docs/ARCHITECTURE.md`` §6 names ``mlruns/, mlflow.db`` as infrastructure
that is "live from the project's first working phase, not deferred" — but
Phase 0 itself (§20) explicitly deferred the *full* backend, keeping only
"a lightweight, queryable experiment log — initially a flat table, upgraded
to a full MLflow tracking backend once the validation harness in Project
Phase 2 stabilizes." The harness (``validation/harness.py``, step 2.9) is
that stabilization point, so this module is where the upgrade actually
happens: every ``harness.evaluate_candidate()`` result now gets logged both
ways through the single entrypoint below (``log_candidate``), rather than
each caller having to remember to do both.

The flat CSV is kept, not replaced — ``docs/PHASE2_EXECUTION_PLAN.md``
step 2.10 is explicit that it stays "for quick grep-able review." Its
column schema (``EXPERIMENT_LOG_COLUMNS``) is exactly the one Phase 0
established (``reports/experiments/experiment_log.csv``, rows EXP-001
through EXP-007) — only appended to here, via ``next_experiment_id``
continuing the same ``EXP-NNN`` sequence, never reordered or renamed, so
every existing Phase 1 row stays byte-compatible with every future one.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from tws_forecast.data.loaders import get_repo_root
from tws_forecast.utils.seeds import RANDOM_SEED
from tws_forecast.validation.harness import CandidateReport, PromotionDecision
from tws_forecast.validation.tiers import TierResult

logger = logging.getLogger(__name__)

__all__ = [
    "EXPERIMENT_LOG_COLUMNS",
    "DEFAULT_EXPERIMENT_LOG_PATH",
    "DEFAULT_MLFLOW_DB_PATH",
    "DEFAULT_MLRUNS_DIR",
    "MLFLOW_EXPERIMENT_NAME",
    "LoggedExperiment",
    "next_experiment_id",
    "log_candidate",
]

DEFAULT_EXPERIMENT_LOG_PATH = get_repo_root() / "reports" / "experiments" / "experiment_log.csv"
DEFAULT_MLFLOW_DB_PATH = get_repo_root() / "mlflow.db"
DEFAULT_MLRUNS_DIR = get_repo_root() / "mlruns"
MLFLOW_EXPERIMENT_NAME = "tws-forecast-validation"

# Exact column order established in Phase 0 -- never reordered, only ever
# appended to as new rows, so EXP-001..EXP-007 (Phase 1's data-forensics
# experiments, already committed) stay byte-compatible with every row this
# module writes.
EXPERIMENT_LOG_COLUMNS = [
    "experiment_id", "experiment_name", "timestamp", "git_commit", "data_version",
    "training_cutoff", "blackout_scenario_id", "model", "seed",
    "cv_tier1_rmse", "cv_tier2_rmse", "cv_tier3_rmse", "notes",
]


@dataclass(frozen=True)
class LoggedExperiment:
    """What one ``log_candidate()`` call produced — enough for a caller (or
    a test) to verify both the CSV row and the MLflow run actually landed."""

    experiment_id: str
    csv_path: Path
    mlflow_run_id: str | None


def _git_commit(repo_root: Path) -> str:
    """Best-effort short git commit hash. Falls back to Phase 0's own
    ``"PENDING_COMMIT"`` convention (see EXP-001..EXP-007's rows) for
    anything not yet committed, or not a git checkout at all — a throwaway
    sandbox run should never raise just because logging couldn't resolve a
    commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "PENDING_COMMIT"
    except Exception:
        return "PENDING_COMMIT"


def next_experiment_id(csv_path: Path) -> str:
    """Next ``EXP-NNN`` id, continuing Phase 1's own sequence rather than
    starting a separate numbering scheme for model candidates — both kinds
    of row share the same file and the same schema."""
    if not csv_path.exists():
        return "EXP-001"

    max_n = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            exp_id = row.get("experiment_id", "")
            if exp_id.startswith("EXP-"):
                try:
                    max_n = max(max_n, int(exp_id.removeprefix("EXP-")))
                except ValueError:
                    continue
    return f"EXP-{max_n + 1:03d}"


def _tier_rmse_str(tier: TierResult | None) -> str:
    return f"{tier.overall_rmse:.4f}" if tier is not None else "N/A"


def _append_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _get_or_create_experiment(name: str, artifact_location: Path) -> str:
    """Idempotent: reuses the MLflow experiment if it already exists (every
    candidate logged from this project lands in the same named experiment),
    otherwise creates it with an explicit artifact root — pinned rather than
    left to MLflow's cwd-relative default, so artifact location doesn't
    silently depend on where a script happened to be invoked from."""
    client = MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    artifact_location.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=artifact_location.as_uri())


def _log_decomposition_artifacts(report: CandidateReport) -> None:
    named_frames = (
        ("tier1_decomposition", report.tier1_decomposition),
        ("tier2_decomposition", report.tier2_decomposition),
        ("tier3_decomposition", report.tier3_decomposition),
        ("degradation_slope", report.degradation_slope),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, df in named_frames:
            if df is None:
                continue
            file_path = tmp_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
            mlflow.log_artifact(str(file_path))


def _log_to_mlflow(
    report: CandidateReport,
    decision: PromotionDecision | None,
    model_name: str,
    seed: int,
    mlflow_db_path: Path,
    mlruns_dir: Path,
) -> str:
    mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    experiment_id = _get_or_create_experiment(MLFLOW_EXPERIMENT_NAME, mlruns_dir)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name=report.candidate_id) as run:
        mlflow.log_param("candidate_id", report.candidate_id)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("seed", seed)
        if report.tier2 is not None:
            mlflow.log_param("tier2_scenario", report.tier2.scenario_name)

        if report.tier1 is not None:
            mlflow.log_metric("cv_tier1_rmse", report.tier1.overall_rmse)
        if report.tier2 is not None:
            mlflow.log_metric("cv_tier2_rmse", report.tier2.overall_rmse)
        if report.tier3 is not None:
            mlflow.log_metric("cv_tier3_rmse", report.tier3.overall_rmse)

        if decision is not None:
            mlflow.log_param("promoted", decision.promoted)
            mlflow.log_param("rung", decision.rung or "none")
            if decision.regressed_buckets:
                mlflow.log_param("regressed_buckets", ",".join(decision.regressed_buckets))

        _log_decomposition_artifacts(report)

        run_id = run.info.run_id

    return run_id


def log_candidate(
    report: CandidateReport,
    decision: PromotionDecision | None = None,
    model_name: str | None = None,
    notes: str = "",
    seed: int = RANDOM_SEED,
    csv_path: Path | None = None,
    mlflow_db_path: Path | None = None,
    mlruns_dir: Path | None = None,
    log_to_mlflow: bool = True,
) -> LoggedExperiment:
    """The single place every ``harness.evaluate_candidate()`` result gets
    recorded — one row appended to the flat CSV, and (unless disabled) one
    MLflow run logging the same tier RMSEs plus the full decomposition
    tables and degradation slope as CSV artifacts.

    Parameters
    ----------
    report:
        Output of ``harness.evaluate_candidate()``.
    decision:
        Output of ``harness.promote(report, ...)``, if already computed.
        When given, its rung/regression outcome is folded into the CSV
        ``notes`` field and logged as MLflow params — logging never calls
        ``promote()`` itself, since not every logged candidate is meant to
        be evaluated for promotion (e.g. a quick Tier-1/2-only exploration).
    model_name:
        Free-text model family/version label for the CSV ``model`` column
        and the MLflow ``model_name`` param. Defaults to ``report.
        candidate_id`` when not given.
    notes:
        Caller-supplied free text, appended to (not replacing) the
        auto-generated promotion-outcome note when ``decision`` is given.
    seed:
        Recorded in both the CSV and MLflow — defaults to the project's
        standing ``RANDOM_SEED``.
    csv_path, mlflow_db_path, mlruns_dir:
        Override the real project paths — tests point these at ``tmp_path``
        so no test run ever touches the actual ``reports/experiments/
        experiment_log.csv`` or ``mlflow.db``.
    log_to_mlflow:
        Set False to skip the MLflow run entirely (CSV row still written).
        Useful for a fast, dependency-light logging path if MLflow is ever
        unavailable; every real harness run should leave this True.

    Returns
    -------
    LoggedExperiment
        The new ``experiment_id``, the CSV path written to, and the MLflow
        ``run_id`` (``None`` if ``log_to_mlflow=False``).
    """
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_EXPERIMENT_LOG_PATH
    mlflow_db_path = Path(mlflow_db_path) if mlflow_db_path is not None else DEFAULT_MLFLOW_DB_PATH
    mlruns_dir = Path(mlruns_dir) if mlruns_dir is not None else DEFAULT_MLRUNS_DIR
    model_name = model_name or report.candidate_id

    experiment_id = next_experiment_id(csv_path)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    blackout_scenario_id = report.tier2.scenario_name if report.tier2 is not None else "N/A"

    full_notes = notes
    if decision is not None:
        decision_note = (
            f"promoted at rung '{decision.rung}'"
            if decision.promoted
            else f"not promoted -- {decision.reason}"
        )
        full_notes = f"{notes} {decision_note}".strip()

    row = {
        "experiment_id": experiment_id,
        "experiment_name": f"Phase2_Harness_{report.candidate_id}",
        "timestamp": timestamp,
        "git_commit": _git_commit(get_repo_root()),
        "data_version": "dataset_manifest.json (see data/raw/)",
        "training_cutoff": "N/A (multi-fold expanding-window CV -- see MLflow run for per-fold detail)",
        "blackout_scenario_id": blackout_scenario_id,
        "model": model_name,
        "seed": seed,
        "cv_tier1_rmse": _tier_rmse_str(report.tier1),
        "cv_tier2_rmse": _tier_rmse_str(report.tier2),
        "cv_tier3_rmse": _tier_rmse_str(report.tier3),
        "notes": full_notes,
    }
    _append_csv_row(csv_path, row)
    logger.info("log_candidate(%r): appended %s to %s", report.candidate_id, experiment_id, csv_path)

    mlflow_run_id = None
    if log_to_mlflow:
        mlflow_run_id = _log_to_mlflow(report, decision, model_name, seed, mlflow_db_path, mlruns_dir)
        logger.info("log_candidate(%r): logged MLflow run %s", report.candidate_id, mlflow_run_id)

    return LoggedExperiment(experiment_id=experiment_id, csv_path=csv_path, mlflow_run_id=mlflow_run_id)
