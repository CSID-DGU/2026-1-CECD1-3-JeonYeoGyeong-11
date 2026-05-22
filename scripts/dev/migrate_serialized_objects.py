"""Scan for serialized assets that still reference legacy ``spectral_fl`` module paths.

Gate 6 hard cleanup requires either migrating these assets or documenting them as
out of post-Gate-6 compatibility guarantees. This script is read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACKED_SUFFIXES = (".pkl", ".pickle", ".pt")
SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", ".venv311", "node_modules"}
LEGACY_MARKERS = (b"spectral_fl.", b"spectral_fl/")


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def scan_serialized(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TRACKED_SUFFIXES:
            continue
        if SKIP_DIR_NAMES.intersection(path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            sample = path.read_bytes()[: 512 * 1024]
        except OSError:
            continue
        if any(marker in sample for marker in LEGACY_MARKERS):
            findings.append(
                {
                    "path": rel,
                    "reason": "legacy_module_path_detected",
                }
            )
        else:
            findings.append({"path": rel, "reason": "no_legacy_module_path_in_sample"})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to auto-detected root).",
    )
    args = parser.parse_args()
    root = args.root or repo_root()
    findings = scan_serialized(root)
    legacy = [row for row in findings if row["reason"] == "legacy_module_path_detected"]
    report = {
        "root": str(root),
        "total_serialized_files": len(findings),
        "legacy_marker_hits": len(legacy),
        "findings": findings,
        "gate6_action": (
            "no migration required"
            if not legacy
            else "migrate or declare out-of-scope before removing spectral_fl shim"
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if not legacy else 2


if __name__ == "__main__":
    raise SystemExit(main())
