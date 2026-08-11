"""Tests for tws_forecast.validation.experiment_log.

Every test points csv_path/mlflow_db_path/mlruns_dir at tmp_path, so no
test run here ever touches the real reports/experiments/experiment_log.csv
or mlflow.db -- the same hermetic-fixture convention used throughout
validation/ (golden_dir, tmp_path-based scenario configs, etc.).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from tws_forecast.validation.experiment_log import (
    EXPERIMENT_LOG_COLUMNS,
    LoggedExperiment,
    _git_commit,
    log_candidate,
    next_experiment_id,
)
from tws_forecast.validation.harness import CandidateReport, PromotionDecision
from tws_forecast.validation.tiers import TierResult


def _tier(overall_rmse: float, tier: int = 2, scenario_name: str = "blackout_curve") -> TierResult:
    return TierResult(
        tier=tier, scenario_name=scenario_name, predictions=pd.DataFrame(),
        fold_rmses=(overall_rmse,), overall_rmse=overall_rmse,
    )


def _decomp(rmse: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame([{"slice_type": "staleness_bucket", "slice_value": "k=2", "n": 10, "rmse": rmse}])


# --- next_experiment_id ------------------------------------------------------


def test_next_experiment_id_empty_file_gives_exp_001(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment_log.csv"
    assert next_experiment_id(csv_path) == "EXP-001"


def test_next_experiment_id_continues_existing_sequence(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment_log.csv"
    csv_path.write_text(
        "experiment_id,experiment_name\nEXP-001,a\nEXP-002,b\nEXP-007,g\n"
    )
    assert next_experiment_id(csv_path) == "EXP-008"


def test_next_experiment_id_ignores_malformed_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiment_log.csv"
    csv_path.write_text(
        "experiment_id,experiment_name\nEXP-001,a\nNOT-AN-EXP-ID,b\nEXP-oops,c\n"
    )
    assert next_experiment_id(csv_path) == "EXP-002"


# --- _git_commit --------------------------------------------------------------


def test_git_commit_falls_back_when_not_a_git_repo(tmp_path: Path) -> None:
    assert _git_commit(tmp_path) == "PENDING_COMMIT"


def test_git_commit_returns_a_hash_in_the_real_repo() -> None:
    from tws_forecast.data.loaders import get_repo_root

    commit = _git_commit(get_repo_root())
    assert commit == "PENDING_COMMIT" or len(commit) >= 7


# --- log_candidate: CSV half ---------------------------------------------------


@pytest.fixture()
def mlflow_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "csv_path": tmp_path / "experiment_log.csv",
        "mlflow_db_path": tmp_path / "mlflow.db",
        "mlruns_dir": tmp_path / "mlruns",
    }


def test_log_candidate_writes_header_and_row_with_expected_columns(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(
        candidate_id="cand_a", tier1=_tier(0.55, tier=1), tier2=_tier(0.60, tier=2),
        tier2_decomposition=_decomp(),
    )
    logged = log_candidate(report, model_name="TestModel", notes="hello", **mlflow_paths, log_to_mlflow=False)

    assert isinstance(logged, LoggedExperiment)
    assert logged.experiment_id == "EXP-001"
    assert logged.mlflow_run_id is None

    with open(mlflow_paths["csv_path"]) as f:
        header = f.readline().strip().split(",")
    assert header == EXPERIMENT_LOG_COLUMNS

    # keep_default_na=False: pandas otherwise silently parses the literal
    # string "N/A" (this module's own not-applicable marker, matching Phase
    # 0's EXP-001..EXP-007 rows) as a float NaN, which isn't what we're
    # testing here.
    df = pd.read_csv(mlflow_paths["csv_path"], keep_default_na=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["experiment_id"] == "EXP-001"
    assert row["model"] == "TestModel"
    assert float(row["cv_tier1_rmse"]) == pytest.approx(0.55)
    assert float(row["cv_tier2_rmse"]) == pytest.approx(0.60)
    assert row["cv_tier3_rmse"] == "N/A"  # no tier3 on this report
    assert row["blackout_scenario_id"] == "blackout_curve"
    assert "hello" in row["notes"]


def test_log_candidate_defaults_model_name_to_candidate_id(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_b", tier1=_tier(0.5, tier=1), tier2=_tier(0.5, tier=2))
    log_candidate(report, **mlflow_paths, log_to_mlflow=False)
    df = pd.read_csv(mlflow_paths["csv_path"])
    assert df.iloc[0]["model"] == "cand_b"


def test_log_candidate_blackout_scenario_id_is_na_without_tier2(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_c", tier1=_tier(0.5, tier=1), tier2=None)
    log_candidate(report, **mlflow_paths, log_to_mlflow=False)
    df = pd.read_csv(mlflow_paths["csv_path"], keep_default_na=False)
    assert df.iloc[0]["blackout_scenario_id"] == "N/A"


def test_log_candidate_appends_promotion_outcome_to_notes(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_d", tier1=_tier(0.5, tier=1), tier2=_tier(0.5, tier=2))
    promoted_decision = PromotionDecision(
        candidate_id="cand_d", promoted=True, rung="serious_contender", reason="cleared rung"
    )
    log_candidate(report, decision=promoted_decision, notes="run 1", **mlflow_paths, log_to_mlflow=False)

    not_promoted = CandidateReport(candidate_id="cand_e", tier1=_tier(0.7, tier=1), tier2=_tier(0.7, tier=2))
    not_promoted_decision = PromotionDecision(
        candidate_id="cand_e", promoted=False, rung=None, reason="did not clear the naive floor"
    )
    log_candidate(not_promoted, decision=not_promoted_decision, notes="run 2", **mlflow_paths, log_to_mlflow=False)

    df = pd.read_csv(mlflow_paths["csv_path"])
    assert "promoted at rung 'serious_contender'" in df.iloc[0]["notes"]
    assert "run 1" in df.iloc[0]["notes"]
    assert "not promoted -- did not clear the naive floor" in df.iloc[1]["notes"]
    assert "run 2" in df.iloc[1]["notes"]


def test_log_candidate_multiple_calls_increment_experiment_id(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_f", tier1=_tier(0.5, tier=1), tier2=_tier(0.5, tier=2))
    first = log_candidate(report, **mlflow_paths, log_to_mlflow=False)
    second = log_candidate(report, **mlflow_paths, log_to_mlflow=False)
    assert first.experiment_id == "EXP-001"
    assert second.experiment_id == "EXP-002"
    df = pd.read_csv(mlflow_paths["csv_path"])
    assert len(df) == 2


# --- log_candidate: MLflow half ------------------------------------------------


def test_log_candidate_logs_real_mlflow_run_with_metrics_params_and_artifacts(
    mlflow_paths: dict[str, Path],
) -> None:
    report = CandidateReport(
        candidate_id="cand_mlflow", tier1=_tier(0.52, tier=1), tier2=_tier(0.58, tier=2),
        tier3=_tier(0.65, tier=3, scenario_name="test_regime_replay"),
        tier1_decomposition=_decomp(0.5), tier2_decomposition=_decomp(0.58),
        tier3_decomposition=_decomp(0.65), degradation_slope=_decomp(0.4),
    )
    decision = PromotionDecision(
        candidate_id="cand_mlflow", promoted=False, rung=None,
        reason="regressed", regressed_buckets=("k=6", "k=7"),
    )
    logged = log_candidate(report, decision=decision, model_name="LGBM_v1", **mlflow_paths)

    assert logged.mlflow_run_id is not None

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_paths['mlflow_db_path']}")
    client = MlflowClient()
    run = client.get_run(logged.mlflow_run_id)

    assert run.data.metrics["cv_tier1_rmse"] == pytest.approx(0.52)
    assert run.data.metrics["cv_tier2_rmse"] == pytest.approx(0.58)
    assert run.data.metrics["cv_tier3_rmse"] == pytest.approx(0.65)
    assert run.data.params["model_name"] == "LGBM_v1"
    assert run.data.params["promoted"] == "False"
    assert run.data.params["rung"] == "none"
    assert run.data.params["regressed_buckets"] == "k=6,k=7"
    assert run.data.params["tier2_scenario"] == "blackout_curve"

    artifact_paths = {a.path for a in client.list_artifacts(logged.mlflow_run_id)}
    assert artifact_paths == {
        "tier1_decomposition.csv", "tier2_decomposition.csv",
        "tier3_decomposition.csv", "degradation_slope.csv",
    }


def test_log_candidate_skips_mlflow_when_disabled(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_g", tier1=_tier(0.5, tier=1), tier2=_tier(0.5, tier=2))
    logged = log_candidate(report, **mlflow_paths, log_to_mlflow=False)
    assert logged.mlflow_run_id is None
    assert not mlflow_paths["mlflow_db_path"].exists()


def test_log_candidate_reuses_same_mlflow_experiment_across_calls(mlflow_paths: dict[str, Path]) -> None:
    report = CandidateReport(candidate_id="cand_h", tier1=_tier(0.5, tier=1), tier2=_tier(0.5, tier=2))
    first = log_candidate(report, **mlflow_paths)
    second = log_candidate(report, **mlflow_paths)

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_paths['mlflow_db_path']}")
    client = MlflowClient()
    run1 = client.get_run(first.mlflow_run_id)
    run2 = client.get_run(second.mlflow_run_id)
    assert run1.info.experiment_id == run2.info.experiment_id
