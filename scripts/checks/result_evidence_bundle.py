"""Validate that result JSON files contain the GraphFL evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphfl_lab.diagnostics.evidence_bundle import validate_evidence_bundle


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--kind",
        choices=["auto", "single-run", "suite-summary"],
        default="auto",
    )
    args = parser.parse_args(argv)

    failed_checks: list[str] = []
    for path in args.paths:
        failures = validate_evidence_bundle(load_json(path), kind=args.kind)
        failed_checks.extend(f"{path}: {failure}" for failure in failures)

    report = {"pass": not failed_checks, "failed_checks": failed_checks}
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
