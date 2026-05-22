"""CLI parser for vision FL suites."""

from __future__ import annotations

import argparse
import sys

from graphfl_lab.cli.aggregation_targets import (
    AGGREGATION_TARGET_CHOICES,
    AGGREGATION_TARGET_SUITE_HELP,
)
from graphfl_lab.config_io import add_config_argument, parse_args_with_config
import graphfl_lab.experiments.vision.suite as _experiment

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
        description="Vision FL variant-by-seed suite.",
        epilog=VARIANTS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    add_config_argument(p)
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", type=str, default="cnn")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.1)

    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.0)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--knn-k", type=int, default=3, help="Default k for ours_knn / ours_random if unsuffixed")
    p.add_argument(
        "--graph-method",
        type=str,
        default="none",
        help=(
            "Default high-level runnable graph-FL method/profile forwarded to "
            "run_vision_experiment. Variant extras can still override it."
        ),
    )
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        help=(
            "Default graph source forwarded to run_vision_experiment. Built-ins "
            "include update, ema_update, normalized_update, classifier_head_update, "
            "weight and layerwise/classifier variants; custom sources can be "
            "provided by --graph-plugin."
        ),
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        choices=AGGREGATION_TARGET_CHOICES,
        help=AGGREGATION_TARGET_SUITE_HELP,
    )

    p.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Cosine cutoff for variant ours_threshold (graph-mode threshold).",
    )
    p.add_argument(
        "--graph-scale-sigma",
        type=float,
        default=1.0,
        help="Scale for magnitude-log and RBF graph modes. For RBF, <=0 uses a median-distance heuristic.",
    )
    p.add_argument(
        "--learned-graph-lambda",
        type=float,
        default=1.0,
        help="Regularization strength for learned_smooth graph modes.",
    )
    p.add_argument(
        "--graph-layer-start",
        type=int,
        default=0,
        help="Tensor index where layer-slice graph sources begin. Negative values count from the end.",
    )
    p.add_argument(
        "--graph-layer-end",
        type=int,
        default=0,
        help="Tensor index where layer-slice graph sources end. 0 means the end.",
    )

    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument(
        "--graph-plugin",
        type=str,
        default="",
        help=(
            "Comma-separated Python graph plugin modules forwarded to "
            "run_vision_experiment."
        ),
    )
    p.add_argument("--use-ema-graph", type=str, default="true")
    p.add_argument("--disable-adaptive-tau", type=str, default="false")
    p.add_argument("--fixed-tau", type=float, default=1.0)
    p.add_argument(
        "--tau-source",
        type=str,
        default="h_spec",
        choices=[
            "h_spec",
            "h_spec_normalized",
            "e_std",
            "h_spec_normalized_times_e_std",
        ],
        help="Default adaptive-tau signal forwarded to run_vision_experiment.",
    )
    p.add_argument(
        "--graph-filter-strength",
        dest="graph_filter_strength",
        type=float,
        default=1.0,
        help=(
            "Exponent on the graph low-pass gain. 0 disables filtering, "
            "1 is the legacy linear low-pass, >1 is stronger low-pass."
        ),
    )
    p.add_argument("--client-update-ema-alpha", type=float, default=0.8)
    p.add_argument(
        "--diagnostic-only",
        type=str,
        default="false",
        help="Forward to run_vision_experiment: log diagnostics but keep FedAvg aggregation weights.",
    )
    p.add_argument(
        "--correction-family",
        type=str,
        default="real_graph",
        choices=["real_graph", "control_graph", "clustering_only", "graph_free"],
    )
    p.add_argument(
        "--control-graph-mode",
        type=str,
        default="random",
        choices=["random", "shuffled", "uniform", "identity"],
    )
    p.add_argument(
        "--cluster-method",
        type=str,
        default="none",
        choices=["none", "kmeans", "hierarchical", "spectral"],
    )
    p.add_argument("--cluster-k", type=int, default=0)
    p.add_argument("--cluster-auto-k", type=str, default="false")
    p.add_argument(
        "--graph-free-mode",
        type=str,
        default="none",
        choices=["none", "norm_clip", "contribution_cap", "dominance_reweight"],
    )
    p.add_argument("--graph-free-gamma", type=float, default=1.0)
    p.add_argument("--clip-quantile", type=float, default=0.9)
    p.add_argument("--contribution-cap", type=float, default=0.0)
    p.add_argument("--diagnostics-enable", type=str, default="false")
    p.add_argument("--save-round-graphs", type=str, default="false")
    p.add_argument("--graph-snapshot-rounds", type=str, default="")
    p.add_argument("--save-update-arrays", type=str, default="false")
    p.add_argument("--loo-enabled", type=str, default="false")
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
    p.add_argument("--train-subset-size", type=int, default=0)
    p.add_argument("--test-subset-size", type=int, default=0)

    p.add_argument("--out-dir", type=str, default="./experiments_current/vision_suite")
    p.add_argument(
        "--suite-tag",
        type=str,
        default="",
        help="Defaults to last path segment of --out-dir when empty.",
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        metavar="NAME",
        default=[
            "fedavg",
            "ours_dense",
            "ours_knn_k2",
            "ours_knn_k3",
            "ours_knn_k5",
            "ours_random_matched_k3",
            "ours_uniform",
        ],
        help="Suite variant tokens (see module docstring). Example: fedavg ours_knn_k3 ours_random_matched_k3",
    )
    p.add_argument(
        "--preload-fedavg-dir",
        type=str,
        default="",
        help=(
            "Directory containing completed result_vision_fedavg_seed*.json "
            "or result_vision_fedavg_seed*.json runs. "
            "Loads final FedAcc per seed for delta computation when FedAvg is omitted from --variants "
            "(resume after partial suite). Same-dir --out-dir is typical."
        ),
    )
    p.add_argument(
        "--reuse-existing-results",
        type=str,
        default="true",
        help=(
            "If true, load an existing result JSON at the expected path instead "
            "of rerunning that variant/seed. Useful for resuming long suites."
        ),
    )
    return parse_args_with_config(p)


def main() -> None:
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
