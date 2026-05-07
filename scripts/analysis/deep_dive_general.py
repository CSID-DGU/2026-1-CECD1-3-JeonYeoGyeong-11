"""Deep-dive helper for general FL: merge per-method JSONs, then run ``deep_dive_seed``.

``run_general_experiment.py`` writes separate ``result_general_fedavg_*.json`` and
``result_general_ours_*.json``. This script locates the pair for a given suite tag,
variant label, and seed, merges them (same layout as FGL ``--method both``), and
invokes ``scripts/analysis/deep_dive_seed.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=str, required=True)
    p.add_argument("--suite-tag", type=str, required=True)
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Row label / vlabel, e.g. ours_knn_k3",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sd = Path(args.suite_dir)
    tag = args.suite_tag.strip()
    seed = int(args.seed)
    v = args.variant.strip().lower()

    fed = sd / f"result_general_fedavg_seed{seed}_{tag}_fedavg_seed{seed}.json"
    ours = sd / f"result_general_ours_seed{seed}_{tag}_{v}_seed{seed}.json"
    if not fed.is_file():
        raise SystemExit(f"Missing FedAvg JSON: {fed}")
    if not ours.is_file():
        raise SystemExit(f"Missing Ours JSON: {ours}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = out / "_merged_both_for_deep_dive.json"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "merge_general_fedavg_ours.py"),
            "--fedavg",
            str(fed),
            "--ours",
            str(ours),
            "--out",
            str(merged),
        ],
        check=True,
        cwd=str(root),
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analysis" / "deep_dive_seed.py"),
            "--path",
            str(merged),
            "--out-dir",
            str(out),
        ],
        check=True,
        cwd=str(root),
    )
    print(f"Deep-dive artifacts under {out}")


if __name__ == "__main__":
    main()
