"""CLI parser for general client count sweep."""

from __future__ import annotations

import argparse
import sys

from spectral_fl.config_io import add_config_argument, parse_args_with_config
import spectral_fl.experiments.general.client_count_sweep as _experiment

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
        description="Sweep General FL suite runs across num_clients values."
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    add_config_argument(p)
    p.add_argument("--client-counts", type=int, nargs="+", default=[5, 10, 20])

    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--model", type=str, default="mlp")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.1)

    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.0)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--knn-k", type=int, default=2)
    p.add_argument("--graph-source", type=str, default="update")
    p.add_argument("--aggregation-target", type=str, default="spectral_filtered_update")
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
    p.add_argument("--spectral-filter-strength", type=float, default=1.0)
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
    p.add_argument("--train-subset-size", type=int, default=1000)
    p.add_argument("--test-subset-size", type=int, default=300)
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[
            "fedavg",
            "fedavgm",
            "ours_spectral_filtered_knn_k2",
            "ours_spectral_filtered_random_matched_k2",
        ],
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments_current/client_count_sweep",
    )
    p.add_argument("--sweep-tag", type=str, default="client_count_sweep")
    return parse_args_with_config(p)


def main() -> None:
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
