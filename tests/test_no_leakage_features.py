"""Standing guard: no feature module may derive a column from row
position/index within the file (docs/ARCHITECTURE.md §7's leakage-firewall
table). This test currently passes vacuously — no feature modules exist yet
(Project Phase 4) — and is written now so nobody has to remember to add it
later, per docs/PHASE2_EXECUTION_PLAN.md step 2.8.
"""

from pathlib import Path

from tws_forecast.data.loaders import get_repo_root
from tws_forecast.validation.leakage_tests import scan_features_module_for_disallowed_names


def test_current_features_module_has_no_disallowed_names() -> None:
    features_dir = get_repo_root() / "src" / "tws_forecast" / "features"
    violations = scan_features_module_for_disallowed_names(features_dir)
    assert violations == [], (
        f"Disallowed row-position-derived feature name(s) found: {violations}"
    )


def test_scan_returns_empty_for_nonexistent_directory(tmp_path: Path) -> None:
    violations = scan_features_module_for_disallowed_names(tmp_path / "does_not_exist")
    assert violations == []


def test_scan_returns_empty_for_directory_with_no_python_files(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("nothing to see here: row_index")
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert violations == []


def test_scan_catches_disallowed_pattern_in_a_python_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_features.py"
    bad_file.write_text(
        "def compute(df):\n"
        "    df['test_row_index'] = range(len(df))\n"
        "    return df\n"
    )
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert len(violations) == 1
    assert violations[0].file == bad_file
    assert violations[0].line_number == 2
    assert "test_row_index" in violations[0].line


def test_scan_catches_multiple_disallowed_patterns() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        f1 = tmp_path / "a.py"
        f1.write_text("relative_test_position = 1\n")
        f2 = tmp_path / "b.py"
        f2.write_text("file_position = 2\nrow_order = 3\n")

        violations = scan_features_module_for_disallowed_names(tmp_path)
        matched_files = {v.file.name for v in violations}
        assert matched_files == {"a.py", "b.py"}
        assert len(violations) == 3  # relative_test_position, file_position, row_order


def test_scan_does_not_false_positive_on_unrelated_names(tmp_path: Path) -> None:
    ok_file = tmp_path / "ok_features.py"
    ok_file.write_text(
        "def compute_lag(df, origin_time):\n"
        "    # a totally normal, leakage-safe rolling feature\n"
        "    sub = df[df['time'] < origin_time]\n"
        "    return sub['TWS_t'].iloc[-1] if len(sub) else float('nan')\n"
    )
    violations = scan_features_module_for_disallowed_names(tmp_path)
    assert violations == []
