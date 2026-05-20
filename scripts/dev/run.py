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
        "spectral_fl/experiments/suites/vision/variants.py",
        "spectral_fl/strategies/graphfl/strategy.py",
        "spectral_fl/experiments/vision/suite.py",
        "added - removed <= 0",
    ),
    "docs/removed-materials.md": (
        "pre-graphfl-rename",
        "e647da931bb3a78cc228ac2ad31103537b5ed640",
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
