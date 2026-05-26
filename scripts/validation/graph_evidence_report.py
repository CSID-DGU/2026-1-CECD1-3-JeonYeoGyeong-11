"""Generate Graph-FL framework evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphfl_lab.validation import generate_evidence_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "poster"], default="smoke")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="experiments_current/graph_evidence_smoke",
    )
    parser.add_argument(
        "--real-suite-dir",
        default="",
        help="Optional directory containing diagnostic_summary.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    real_dir = args.real_suite_dir or None
    if real_dir:
        real_path = Path(real_dir)
        if not real_path.is_absolute():
            real_dir = str((REPO_ROOT / real_path).resolve())

    pack = generate_evidence_pack(
        out_dir,
        profile=args.profile,
        include_external=bool(args.include_external),
        real_summary_dir=real_dir,
    )
    print(f"OUT_DIR {pack.out_dir}")
    print(f"VERDICT {'PASS' if pack.verdict.get('pass') else 'NEEDS_REVIEW'}")
    print(json.dumps(pack.files, indent=2))
    return 0 if pack.verdict.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
