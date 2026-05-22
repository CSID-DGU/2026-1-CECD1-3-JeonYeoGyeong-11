"""CLI parser for vision FL stress grids."""

from __future__ import annotations

import argparse
import sys

from graphfl_lab.config_io import add_config_argument, parse_args_with_config
import graphfl_lab.experiments.vision.stress_grid as _experiment

# Compatibility re-exports for older imports from this CLI module.
globals().update(
    {
        name: getattr(_experiment, name)
        for name in dir(_experiment)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run vision FL stress suites across num_clients, train/test subset "
            "sizes, Dirichlet alpha, local epochs, and k."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    add_config_argument(p)

    p.add_argument("--client-counts", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--train-subset-sizes", type=int, nargs="+", default=[1000, 3000, 6000])
    p.add_argument("--test-subset-sizes", type=int, nargs="+", default=[1000, 2000])
    p.add_argument("--knn-ks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--dirichlet-alphas", type=float, nargs="+", default=[0.03])

    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--model", type=str, default="mlp")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, nargs="+", default=[2])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])

    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.0)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--graph-source", type=str, default="update")
    p.add_argument("--aggregation-target", type=str, default="graph_filtered_update")
    p.add_argument("--edge-threshold", type=float, default=0.0)
    p.add_argument("--graph-scale-sigma", type=float, default=1.0)
    p.add_argument("--learned-graph-lambda", type=float, default=1.0)
    p.add_argument("--graph-layer-start", type=int, default=0)
    p.add_argument("--graph-layer-end", type=int, default=0)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--use-ema-graph", type=str, default="true")
    p.add_argument("--disable-adaptive-tau", type=str, default="false")
    p.add_argument("--fixed-tau", type=float, default=1.0)
    p.add_argument("--tau-source", type=str, default="h_spec")
    p.add_argument(
        "--graph-filter-strength",
        dest="graph_filter_strength",
        type=float,
        default=1.0,
    )
    p.add_argument("--client-update-ema-alpha", type=float, default=0.8)
    p.add_argument("--diagnostic-only", type=str, default="false")
    p.add_argument("--e-std-threshold", type=float, default=0.0)
    p.add_argument("--min-client-weight", type=float, default=0.0)
    p.add_argument("--server-learning-rate", type=float, default=1.0)
    p.add_argument("--server-momentum", type=float, default=0.9)
    p.add_argument("--ours-server-learning-rate", type=float, default=1.0)
    p.add_argument("--ours-server-momentum", type=float, default=0.0)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--fedopt-eta", type=float, default=0.1)
    p.add_argument("--fedopt-eta-l", type=float, default=0.1)
    p.add_argument("--fedopt-beta1", type=float, default=0.9)
    p.add_argument("--fedopt-beta2", type=float, default=0.99)
    p.add_argument("--fedopt-tau", type=float, default=1e-9)
    p.add_argument("--trimmed-beta", type=float, default=0.1)

    p.add_argument("--data-root", type=str, default="./data/torchvision")
    p.add_argument(
        "--variant-templates",
        type=str,
        nargs="+",
        default=DEFAULT_VARIANT_TEMPLATES,
        help="Variant tokens. Use {k} to expand over --knn-ks.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments_current/vision_stress_grid",
    )
    p.add_argument("--grid-tag", type=str, default="vision_stress_grid")
    p.add_argument(
        "--dry-run",
        type=str,
        default="false",
        help="If true, write the manifest without launching suites.",
    )
    p.add_argument(
        "--skip-existing",
        type=str,
        default="false",
        help="If true, reuse a suite when vision/general suite summary CSV already exists.",
    )
    p.add_argument(
        "--reuse-existing-results",
        type=str,
        default="true",
        help="Forwarded to run_vision_suite.py for partial suite resume.",
    )
    p.add_argument(
        "--max-suites",
        type=int,
        default=0,
        help="Optional execution cap for smoke tests. 0 means no cap.",
    )
    p.add_argument(
        "--fedavg-collapse-acc-threshold",
        type=float,
        default=0.45,
        help="Auto-review threshold: FedAvg mean accuracy at or below this is treated as a collapse/stress condition.",
    )
    p.add_argument(
        "--meaningful-delta",
        type=float,
        default=0.01,
        help="Auto-review threshold for a meaningful mean_delta over FedAvg.",
    )
    p.add_argument(
        "--random-margin",
        type=float,
        default=0.005,
        help="Auto-review margin required to say a variant beats matched random.",
    )
    return parse_args_with_config(p)


def main() -> None:
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
