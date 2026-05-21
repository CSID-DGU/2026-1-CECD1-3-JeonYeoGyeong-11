"""Developer gate-check entrypoint for staged cleanup work."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPORT_PATH = Path("docs/maintenance/last_gate_check.json")

GATE0_REQUIRED_FILES = (
    "docs/maintenance/cleanup-status.md",
    "docs/maintenance/rename-inventory.md",
    "docs/maintenance/line-budget-allowlist.txt",
    "MIGRATION.md",
    "docs/removed-materials.md",
    "scripts/dev/run.py",
    "tests/golden/README.md",
)

GATE0_REQUIRED_TEXT = {
    "docs/maintenance/cleanup-status.md": (
        "Existing Unstaged Docs",
        "Existing Plan Mapping",
        "scripts/dev/run.py gate-check <gate>",
        "Gate 4c",
        "workflow_dispatch",
        "real move verification",
        "single Gate branch",
        "docs/framework/claim.md",
        "docs/framework/graph_fl_experimental_design.md",
        "docs/framework/graph_fl_experimental_design_appendix.md",
        "docs/research/prior-work-review.md",
    ),
    "docs/maintenance/rename-inventory.md": (
        "tests/structure/test_boundaries.py",
        "pickle/checkpoint",
        "spectral_fl",
        "graphfl_lab",
    ),
    "docs/maintenance/line-budget-allowlist.txt": (
        "git diff --numstat",
        "added - removed must be <= 0",
    ),
    "MIGRATION.md": (
        "For Users",
        "For Maintainers",
        "pre-graphfl-rename",
    ),
    "docs/removed-materials.md": (
        "pre-graphfl-rename",
        "Tombstones",
    ),
    "tests/golden/README.md": (
        "Volatile Fields",
        "timestamp",
        "run_id",
        "Schema comparison is exact",
    ),
}

GATE1_REQUIRED_TEXT = {
    "docs/maintenance/rename-inventory.md": (
        "Gate 1 Pattern Summary",
        "Serialized Asset Inventory",
        "data/Cora/processed/data.pt",
        "tests/structure/test_boundaries.py",
        "spectral_fl",
        "result_general_",
        "spectral_filter_strength",
    ),
    "docs/maintenance/line-budget-allowlist.txt": (
        "Protected Paths",
        "graphfl_lab/experiments/suites/vision/variants.py",
        "graphfl_lab/strategies/graphfl/strategy.py",
        "graphfl_lab/experiments/vision/suite.py",
        "added - removed <= 0",
    ),
    "docs/removed-materials.md": (
        "pre-graphfl-rename",
        "e647da931bb3a78cc228ac2ad31103537b5ed640",
    ),
}

GATE2_REQUIRED_TEXT = {
    "graphfl_lab/diagnostics/result_schema.py": (
        "RESULT_SCHEMA_VERSION",
        "LEGACY_RESULT_SCHEMA_VERSION",
        "with_result_schema",
        "config_aliases_used",
        "unsupported_components",
    ),
    "graphfl_lab/config_io.py": (
        "_config_aliases_used",
        "ARG_DEST_ALIASES",
        "configs/general->configs/vision",
    ),
    "graphfl_lab/flower_app.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "graphfl_lab/experiments/cora/graph_ablation.py": (
        "with_result_schema",
        "config_aliases_from_args",
        "unsupported_components_from_args",
    ),
    "tests/diagnostics/test_result_schema.py": (
        "test_with_result_schema_adds_required_fields",
        "test_missing_version_reads_as_v0",
        "test_config_aliases_are_recorded",
    ),
}

GATE3_REQUIRED_TEXT = {
    "graphfl_lab/__init__.py": (
        "Canonical package root",
    ),
    "spectral_fl/__init__.py": (
        "Deprecated compatibility shim",
        "DeprecationWarning",
        "GRAPHFL_LAB_SILENCE_DEPRECATION",
        "graphfl_lab",
        "__path__",
        "__getattr__",
    ),
    "pyproject.toml": (
        "graphfl_lab*",
        "graphfl_lab.flower_app:server_app",
        "graphfl_lab.flower_app:client_app",
    ),
    "tests/core/test_package_alias.py": (
        "test_graphfl_lab_imports_flower_app",
        "test_spectral_fl_warns_by_default",
        "test_spectral_fl_warning_can_be_silenced",
        "test_sys_modules_alias_roots_exist",
        "test_pickle_round_trip_for_canonical_import",
        "test_legacy_submodule_import_still_resolves",
        "test_pickle_round_trip_for_legacy_import",
    ),
}

GATE3B_FORBIDDEN_IMPORT_ALLOWLIST = {
    "graphfl_lab/__init__.py",
    "scripts/dev/run.py",
    "spectral_fl/__init__.py",
    "tests/core/test_package_alias.py",
    "tests/dev/test_run_gate_check.py",
}

GATE3B_FORBIDDEN_IMPORTS = (
    "from spectral_fl",
    "import spectral_fl",
    "spectral_fl.",
)

GATE4A_REQUIRED_TEXT = {
    "graphfl_lab/cli/experiment_dispatcher.py": (
        "--track",
        "vision",
        "cora",
        "DeprecationWarning",
        "TRACK_MODULES",
    ),
    "run_experiment.py": (
        "experiment_dispatcher",
        "main = _impl.main",
    ),
    "tests/cli/test_experiment_dispatcher.py": (
        "test_missing_track_defaults_to_cora_with_deprecation_warning",
        "test_track_cora_dispatches_without_track_argument",
        "test_track_vision_dispatches_without_track_argument",
    ),
}

GATE4B_REQUIRED_TEXT = {
    "graphfl_lab/cli/experiment_dispatcher.py": (
        "main_for_track",
        "vision_main",
        "cora_main",
    ),
    "run_vision_experiment.py": (
        "experiment_dispatcher",
        "main = _dispatcher.vision_main",
    ),
    "run_general_experiment.py": (
        "experiment_dispatcher",
        "main = _dispatcher.vision_main",
    ),
    "tests/cli/test_experiment_dispatcher.py": (
        "test_named_track_helpers_use_unified_dispatch",
    ),
}

GATE4C_REQUIRED_TEXT = {
    ".github/workflows/ci.yml": (
        "graphfl_lab",
        "spectral_fl",
        "python -m unittest discover -s tests",
    ),
    ".github/workflows/nightly.yml": (
        "schedule:",
        "workflow_dispatch:",
        "graphfl_lab",
        "python -m unittest discover -s tests",
    ),
    "scripts/dev/golden.py": (
        "VOLATILE_FIELDS",
        "REQUIRED_RESULT_SCHEMA_KEYS",
        "normalize_payload",
        "compare_payloads",
    ),
    "tests/dev/test_golden.py": (
        "test_normalized_compare_ignores_volatile_fields",
        "test_compare_fails_on_schema_shape_change",
        "test_compare_fails_on_normalized_value_change",
    ),
    "tests/golden/README.md": (
        "Gate 4c captures smoke outputs",
        "Volatile Fields",
        "Schema comparison is exact",
    ),
}

GATE5A_PREP_REQUIRED_TEXT = {
    "graphfl_lab/experiments/suites/result_writer.py": (
        "write_json",
        "write_csv_rows",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "write_json(summary_json, suite_summary)",
        "write_json(rows_path, rows)",
        "write_csv_rows(csv_path, summary_rows",
    ),
    "graphfl_lab/experiments/cora/graph_ablation.py": (
        "write_json(summary_json, suite_summary)",
        "write_json(rows_path, rows)",
        "write_csv_rows(csv_path, summary_rows",
    ),
    "graphfl_lab/experiments/vision/stress_grid.py": (
        "write_json(root / \"stress_grid_auto_review.json\"",
        "write_json(root / \"stress_grid_manifest.json\"",
        "write_json(root / \"stress_grid_summary.json\"",
    ),
    "graphfl_lab/experiments/vision/client_count_sweep.py": (
        "write_json(root / \"client_count_sweep_summary.json\"",
    ),
    "graphfl_lab/experiments/suites/vision/reporting.py": (
        "write_json(out_dir / \"vision_suite_summary.json\"",
        "write_csv_rows(",
    ),
    "tests/experiments/test_result_writer.py": (
        "test_write_json_uses_indented_payload",
        "test_write_csv_rows_preserves_field_order",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5a-prep",
        "do not claim full Gate 5a completion",
    ),
}

GATE5B_PREP_REQUIRED_TEXT = {
    "graphfl_lab/experiments/suites/execution.py": (
        "run_cmd",
        "execute_or_reuse_result",
        "reuse_existing",
    ),
    "graphfl_lab/experiments/vision/suite.py": (
        "from graphfl_lab.experiments.suites.execution import execute_or_reuse_result",
        "from graphfl_lab.experiments.suites.vision.features import",
        "cwd=PROJECT_ROOT",
    ),
    "graphfl_lab/experiments/suites/vision/features.py": (
        "collect_run_features",
        "collect_timing_features",
        "load_preloaded_fedavg_accs",
        "rank_key",
    ),
    "graphfl_lab/experiments/suites/vision/summary.py": (
        "build_summary_rows",
        "mean_di_pre",
        "seed{seed}_delta",
    ),
    "graphfl_lab/experiments/suites/vision/metadata.py": (
        "build_suite_meta",
        "record_suite_timing",
        "mean_di_pre/post",
    ),
    "graphfl_lab/experiments/vision/client_count_sweep.py": (
        "from graphfl_lab.experiments.suites.execution import run_cmd",
        "run_cmd(cmd, cwd=PROJECT_ROOT)",
    ),
    "graphfl_lab/experiments/vision/stress_grid.py": (
        "from graphfl_lab.experiments.suites.execution import run_cmd",
        "run_cmd(cmd, cwd=PROJECT_ROOT)",
    ),
    "tests/experiments/test_suite_execution.py": (
        "test_execute_reuses_existing_result_without_running_command",
        "test_execute_runs_when_reuse_disabled",
        "test_run_cmd_uses_cwd_and_check",
    ),
    "tests/experiments/vision/test_suite_features.py": (
        "test_collect_run_features_exports_diagnostic_means_and_aliases",
        "test_load_preloaded_fedavg_accs_reads_latest_supported_names",
    ),
    "tests/experiments/vision/test_suite_summary.py": (
        "test_build_summary_rows_aggregates_all_diagnostic_fields",
    ),
    "tests/experiments/vision/test_suite_metadata.py": (
        "test_build_suite_meta_documents_full_diagnostic_set",
        "test_record_preloaded_and_timing_metadata",
    ),
    "docs/maintenance/cleanup-status.md": (
        "Gate 5b-prep",
        "do not claim full Gate 5b completion",
    ),
}


def repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


def commit_sha(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _missing_files(root: Path, rel_paths: Iterable[str]) -> list[str]:
    return [rel for rel in rel_paths if not (root / rel).is_file()]


def _missing_text(root: Path, expectations: dict[str, Iterable[str]]) -> list[str]:
    failures: list[str] = []
    for rel, needles in expectations.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"{rel}: file missing")
            continue
        haystack = _read_text(path)
        for needle in needles:
            if needle not in haystack:
                failures.append(f"{rel}: missing text {needle!r}")
    return failures


def _tag_exists(root: Path, tag_name: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", tag_name],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0


def _tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _tracked_serialized_assets(root: Path) -> list[str]:
    suffixes = (".pkl", ".pickle", ".pt", ".pth")
    return [path for path in _tracked_files(root) if path.lower().endswith(suffixes)]


def _forbidden_identity_imports(root: Path) -> list[str]:
    failures: list[str] = []
    for rel in _tracked_files(root):
        if not rel.endswith(".py"):
            continue
        if rel in GATE3B_FORBIDDEN_IMPORT_ALLOWLIST:
            continue
        if rel.startswith("scripts/archive/"):
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = _read_text(path)
        for needle in GATE3B_FORBIDDEN_IMPORTS:
            if needle in text:
                failures.append(f"{rel}: forbidden legacy import token {needle!r}")
                break
    return failures


def _unexpected_legacy_package_files(root: Path) -> list[str]:
    allowed = {"spectral_fl/__init__.py"}
    failures: list[str] = []
    for rel in _tracked_files(root):
        if rel.startswith("spectral_fl/") and rel not in allowed:
            failures.append(f"{rel}: legacy package should only contain the shim")
    return failures


def run_gate_check(gate: str, root: Path | None = None) -> dict[str, object]:
    root = repo_root(root)
    failed_checks: list[str] = []

    if gate == "0":
        for rel in _missing_files(root, GATE0_REQUIRED_FILES):
            failed_checks.append(f"missing required Gate 0 file: {rel}")
        failed_checks.extend(_missing_text(root, GATE0_REQUIRED_TEXT))
    elif gate == "1":
        failed_checks.extend(_missing_text(root, GATE1_REQUIRED_TEXT))
        if not _tag_exists(root, "pre-graphfl-rename"):
            failed_checks.append("missing tag: pre-graphfl-rename")
        serialized = _tracked_serialized_assets(root)
        if serialized:
            failed_checks.append(
                "tracked serialized assets must be classified explicitly: "
                + ", ".join(serialized)
            )
    elif gate == "2":
        failed_checks.extend(_missing_text(root, GATE2_REQUIRED_TEXT))
    elif gate == "3a":
        failed_checks.append("Gate 3a alias bridge is superseded by full Gate 3.")
    elif gate == "3b":
        failed_checks.extend(_forbidden_identity_imports(root))
    elif gate == "3":
        failed_checks.extend(_missing_text(root, GATE3_REQUIRED_TEXT))
        failed_checks.extend(_forbidden_identity_imports(root))
        failed_checks.extend(_unexpected_legacy_package_files(root))
    elif gate == "4a":
        failed_checks.extend(_missing_text(root, GATE4A_REQUIRED_TEXT))
    elif gate == "4b":
        failed_checks.extend(_missing_text(root, GATE4A_REQUIRED_TEXT))
        failed_checks.extend(_missing_text(root, GATE4B_REQUIRED_TEXT))
    elif gate == "4c":
        failed_checks.extend(_missing_text(root, GATE4C_REQUIRED_TEXT))
        failed_checks.append(
            "Gate 4c requires one GitHub nightly or manual-nightly green run before completion."
        )
    elif gate == "5a-prep":
        failed_checks.extend(_missing_text(root, GATE5A_PREP_REQUIRED_TEXT))
    elif gate == "5b-prep":
        failed_checks.extend(_missing_text(root, GATE5B_PREP_REQUIRED_TEXT))
    else:
        failed_checks.append(
            f"Gate {gate} check is not implemented yet; add it during that gate."
        )

    return {
        "gate": str(gate),
        "pass": not failed_checks,
        "failed_checks": failed_checks,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha(root),
    }


def write_report(root: Path, report: dict[str, object]) -> Path:
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate-check", help="Run a cleanup gate check.")
    gate.add_argument("gate", help="Gate id, for example 0, 4a, or 5c.")
    args = parser.parse_args(argv)

    root = repo_root()
    if args.command == "gate-check":
        report = run_gate_check(str(args.gate), root)
        report_path = write_report(root, report)
        print(json.dumps(report, indent=2))
        print(f"Wrote {report_path.relative_to(root)}")
        return 0 if report["pass"] else 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
