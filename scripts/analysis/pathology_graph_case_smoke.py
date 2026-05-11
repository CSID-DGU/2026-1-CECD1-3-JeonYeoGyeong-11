"""Smoke checks for pathology graph case-study instrumentation.

Checks:
1) graph preset is applied,
2) graph is non-empty unless identity/no-graph,
3) random/shuffled/uniform controls are distinguishable,
4) signed graph contains negative edges,
5) cos_delta and rel_delta_change are logged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _result_path(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return out_dir / f"result_general_{method}_seed{seed}{suffix}.json"


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _load(path: Path, method: str) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"][method]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pathology graph instrumentation smoke checks.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "pathology_graph_case_smoke",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = [
        "--dataset",
        "fashionmnist",
        "--partition",
        "dirichlet",
        "--dirichlet-alpha",
        "0.1",
        "--num-clients",
        "10",
        "--rounds",
        "2",
        "--local-epochs",
        "1",
        "--train-subset-size",
        "1000",
        "--test-subset-size",
        "500",
        "--seed",
        str(args.seed),
        "--out-dir",
        str(out_dir),
    ]

    runs = [
        ("update", "graph_smooth", ["--graph-variant", "update", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "3"]),
        ("random", "graph_smooth", ["--graph-variant", "random", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "3"]),
        ("shuffled", "graph_smooth", ["--graph-variant", "shuffled", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "3"]),
        ("uniform", "graph_smooth", ["--graph-variant", "uniform", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "3"]),
        ("identity", "graph_smooth", ["--graph-variant", "identity", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "3"]),
        ("signed", "graph_smooth", ["--graph-preset", "signed_conflict_knn", "--graph-variant", "update", "--graph-smoothing-operator", "signed_conflict_attenuation"]),
    ]

    traces: Dict[str, Dict] = {}
    for label, method, extras in runs:
        run_tag = f"case_smoke_{label}_seed{args.seed}"
        cmd = [args.python_bin, "run_general_experiment.py", "--method", method, "--run-tag", run_tag] + shared + extras
        _run(cmd)
        result_path = _result_path(out_dir=out_dir, method=method, seed=args.seed, run_tag=run_tag)
        res = _load(result_path, method)
        if not res.get("round_trace"):
            raise RuntimeError(f"missing round_trace: {label}")
        traces[label] = dict(res["round_trace"][-1])

    # 1) Preset applied.
    if str(traces["signed"].get("graph_preset", "")).lower() != "signed_conflict_knn":
        raise RuntimeError("preset not applied for signed run")

    # 2) Non-empty graph except identity.
    if float(traces["update"].get("graph_density", 0.0) or 0.0) <= 0.0:
        raise RuntimeError("update graph unexpectedly empty")
    if float(traces["identity"].get("graph_density", 1.0) or 1.0) != 0.0:
        raise RuntimeError("identity control is not empty")

    # 3) Controls are distinguishable in at least one structural metric.
    signatures = {
        k: (
            float(v.get("graph_density", 0.0) or 0.0),
            float(v.get("mean_degree", v.get("graph_degree_mean", 0.0)) or 0.0),
        )
        for k, v in traces.items()
        if k in {"random", "shuffled", "uniform"}
    }
    if len(set(signatures.values())) < 2:
        raise RuntimeError("random/shuffled/uniform controls are not distinguishable")

    # 4) Signed graph includes negative edges.
    if float(traces["signed"].get("negative_edge_count", 0.0) or 0.0) <= 0.0:
        raise RuntimeError("signed graph did not expose negative edges")

    # 5) Intervention metrics are logged.
    for key in ["cos_delta_corrected_vs_base", "rel_delta_change"]:
        if key not in traces["update"]:
            raise RuntimeError(f"missing intervention metric: {key}")

    print(f"[smoke] passed. output_dir={out_dir}")


if __name__ == "__main__":
    main()

