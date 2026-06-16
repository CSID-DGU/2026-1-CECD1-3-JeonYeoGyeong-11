"""CLI parser for graph ablation."""

from __future__ import annotations

import argparse
import sys

from graphfl_lab.cli.aggregation_targets import (
    AGGREGATION_TARGET_CHOICES,
    AGGREGATION_TARGET_SUITE_HELP,
)
from graphfl_lab.config_io import add_config_argument, parse_args_with_config
import graphfl_lab.experiments.cora.graph_ablation as _experiment
from graphfl_lab.cli.argparse_types import json_object

# Compatibility re-exports for older imports from this CLI module.
globals().update(
    {
        name: getattr(_experiment, name)
        for name in dir(_experiment)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Graph-construction ablation: subprocesses run_experiment.py (Cora + GCN Flower clients, not torchvision). "
            "FedAvg plus ours_* variants sharing the graph-FL diagnostic strategy. "
            "Defaults: partition=dirichlet, dirichlet-alpha=0.2, seeds 42-46, rounds 30, knn-k=2 "
            "for knn/random/threshold-style graphs. "
            "For Fashion-MNIST / MNIST / CIFAR use run_vision_experiment.py / run_vision_suite.py instead."
        ),
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    add_config_argument(p)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="SGD epochs per client each FL round (default 1; forwarded to run_experiment).",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="GCN hidden size for Cora clients (default 64; forwarded to run_experiment).",
    )
    p.add_argument(
        "--compression-dim",
        type=int,
        default=256,
        help="Random projection dimension z_i (default 256; forwarded to run_experiment).",
    )
    p.add_argument(
        "--compression-seed",
        type=int,
        default=0,
        help="Seed for Gaussian projection matrix (default 0; forwarded to run_experiment).",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="EMA on similarity graph W between rounds for ours_* (default 0.8).",
    )
    p.add_argument(
        "--graph-seed",
        type=int,
        default=0,
        help="RNG seed for random graph modes (ours_*; forwarded to run_experiment).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root for Cora data cache (default ./data; forwarded to run_experiment).",
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="FedAvg-only spectral warmup rounds before conflict term (ours_* only; forwarded to run_experiment).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument(
        "--partition",
        type=str,
        default="dirichlet",
        choices=["iid", "dirichlet"],
        help="Client label split: iid or Dirichlet skew (default dirichlet; forwarded to run_experiment).",
    )
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.2,
        help=(
            "Dirichlet alpha for label skew (smaller alpha means stronger imbalance). "
            "Default 0.2 is a middle ground; use 0.1 for stronger skew or 0.5 for milder (matches bare run_experiment default)."
        ),
    )
    p.add_argument(
        "--tau-max",
        type=float,
        default=2.0,
        help="Upper bound for adaptive tau schedule (default 2.0; passed through to run_experiment).",
    )
    p.add_argument(
        "--tau-gain",
        type=float,
        default=2.0,
        help="Scale for the tanh tau schedule (default 2.0; passed through to run_experiment).",
    )
    p.add_argument("--conflict-mix", type=float, default=0.2)
    p.add_argument(
        "--knn-k",
        type=int,
        default=2,
        help="Neighbor count k for graph-mode knn / random (matched edges) / threshold variants.",
    )
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        help="Default graph source forwarded to run_experiment.",
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        help=AGGREGATION_TARGET_SUITE_HELP,
    )
    p.add_argument("--aggregation-params", type=json_object, default={})
    p.add_argument("--graph-plugin", type=str, default="")
    p.add_argument("--graph-preset", type=str, default="none")
    p.add_argument("--edge-threshold", type=float, default=0.0)
    p.add_argument("--fixed-tau", type=float, default=1.0)
    p.add_argument(
        "--diagnostic-only",
        type=str,
        default="false",
        help="Forward to run_experiment: log diagnostics but keep FedAvg aggregation weights.",
    )
    p.add_argument(
        "--e-std-threshold",
        type=float,
        default=0.02,
        help=(
            "If std(e) across clients is below this, skip conflict shaping (0 = off). "
            "Default 0.02 matches run_vision_experiment conservative penalty; ours_* only."
        ),
    )
    p.add_argument(
        "--min-client-weight",
        type=float,
        default=0.05,
        help=(
            "Floor on normalized client weights after conflict shaping (0 = off). "
            "Default 0.05 matches run_vision_experiment; ours_* only."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./outputs_graph_ablation",
        help="Directory for per-run result_*.json and suite_<tag>_summary.json (created if missing).",
    )
    p.add_argument("--suite-tag", type=str, default="graph_ablation")
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[
            "fedavg",
            "ours_dense",
            "ours_knn",
            "ours_threshold",
            "ours_random",
            "ours_uniform",
            "ours_no_ema",
            "ours_fixed_tau",
        ],
        help=(
            "Subset of suite variants to run (see module docstring). "
            "Examples: ours_dense ours_no_ema ours_fixed_tau ours_knn."
        ),
    )
    return parse_args_with_config(p)


def main() -> None:
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
