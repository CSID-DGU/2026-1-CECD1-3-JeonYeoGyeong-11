"""Generate dashboard mockup markdown from suite summary."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphfl_lab.experiments.suites.vision.artifacts import (
    SUITE_SUMMARY_CSV_FILENAMES,
    resolve_suite_artifact,
)
from graphfl_lab.experiments.suites.vision.reporting import write_dashboard_mockup


def _load_summary_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate suite dashboard mockup.")
    p.add_argument("--suite-dir", type=str, required=True)
    args = p.parse_args()
    suite_dir = Path(args.suite_dir)
    summary_path = resolve_suite_artifact(suite_dir, SUITE_SUMMARY_CSV_FILENAMES)
    rows = _load_summary_rows(summary_path) if summary_path is not None else []
    out = write_dashboard_mockup(
        suite_dir,
        rows,
        diagnostic_csv_path=(suite_dir / "diagnostic_summary.csv"),
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
