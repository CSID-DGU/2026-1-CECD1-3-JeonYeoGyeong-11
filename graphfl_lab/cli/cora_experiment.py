"""CLI parser for Cora spectral FL experiments."""

from __future__ import annotations

import argparse

from graphfl_lab.cli.aggregation_targets import (
    AGGREGATION_TARGET_CHOICES,
    AGGREGATION_TARGET_HELP,
)
from graphfl_lab.cli.argparse_types import str2bool
from graphfl_lab.config_io import add_config_argument, parse_args_with_config
from graphfl_lab.cli.argparse_types import json_object
from graphfl_lab.experiments.cora import single_run as _experiment
GRAPH_MODE_HELP = (
    "Topology/lower-level graph construction knob under the lifecycle design. "
    "Built-ins include dense, knn, mutual_knn, "
    "threshold, uniform, random, magnitude, global_alignment, signed_abs, "
    "negative, rbf, learned_smooth, pfedgraph_qp and *_knn variants. Custom modes can be "
    "provided by --graph-plugin."
)

# Compatibility re-exports for older imports from graphfl_lab.cli.cora_experiment.
globals().update(
    {
        name: getattr(_experiment, name)
        for name in dir(_experiment)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

def parse_args():
    p = argparse.ArgumentParser()
    add_config_argument(p)
    p.add_argument(
        "--engine",
        type=str,
        default="app",
        choices=["app", "flwr-run", "print-flwr-run"],
        help=(
            "Execution backend. 'app' uses Flower ClientApp/ServerApp locally; "
            "'flwr-run' submits the same app through Flower CLI."
        ),
    )
    p.add_argument(
        "--method",
        type=str,
        default="both",
        choices=[
            "both",
            "fedavg",
            "fedavgm",
            "fedadagrad",
            "fedadam",
            "fedyogi",
            "fednova",
            "fedprox",
            "fedmedian",
            "fedtrimmedavg",
            "fedsim",
            "ours",
        ],
    )
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    # graph-FL runtime / aggregation
    p.add_argument("--compression-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.0)
    p.add_argument("--warmup-rounds", type=int, default=2)

    # graph construction
    p.add_argument(
        "--graph-mode",
        type=str,
        default="dense",
        help=GRAPH_MODE_HELP,
    )
    p.add_argument(
        "--graph-plugin",
        type=str,
        default="",
        help=(
            "Comma-separated Python module names imported before graph building. "
            "Each module may call graphfl_lab.graph.register_graph_builder(...) "
            "or register_graph_source(...) to add custom graph modes/sources."
        ),
    )
    p.add_argument(
        "--graph-preset",
        type=str,
        default="none",
        help=(
            "GraphFLDesign preset or compatibility alias. When set, it resolves "
            "client-state/relation/topology/aggregation metadata to legacy "
            "graph source/mode/target knobs."
        ),
    )
    p.add_argument(
        "--graph-method",
        type=str,
        default="none",
        help=(
            "High-level runnable graph-FL method/profile. This resolves to "
            "the current graph source/mode/aggregation knobs, while explicit "
            "lower-level CLI/config values can override individual knobs."
        ),
    )
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        help=(
            "ClientStateExtractor knob used to build the client relation graph. Built-ins "
            "include update, ema_update, normalized_update, layerwise_update, "
            "classifier_head_update, weight and layerwise/classifier variants; "
            "custom sources can be provided by --graph-plugin."
        ),
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        help=AGGREGATION_TARGET_HELP,
    )
    p.add_argument("--aggregation-params", type=json_object, default={})
    p.add_argument("--knn-k", type=int, default=2)
    p.add_argument("--edge-threshold", type=float, default=0.0)
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
        help="Tensor index where layer-slice graph sources begin. Negative values count from the end, e.g. -2 for the last weight/bias pair.",
    )
    p.add_argument(
        "--graph-layer-end",
        type=int,
        default=0,
        help="Tensor index where layer-slice graph sources end. 0 means the end of the parameter list.",
    )
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--use-ema-graph", type=str2bool, default=True,
                   help="Apply EMA on the client graph between rounds (default: true)")

    # tau ablation
    p.add_argument("--disable-adaptive-tau", type=str2bool, default=False,
                   help="If true, use --fixed-tau every round (no tanh schedule).")
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
        help="Signal used by adaptive tau. h_spec is the legacy behavior.",
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
    p.add_argument(
        "--client-update-ema-alpha",
        type=float,
        default=0.8,
        help="EMA coefficient for client update signals used by ema_update graph sources or graph_filtered_ema_update.",
    )
    p.add_argument("--diagnostic-only", type=str2bool, default=False,
                   help="If true, log graph-FL diagnostics but aggregate with FedAvg weights.")

    # conservative penalty
    p.add_argument("--e-std-threshold", type=float, default=0.0,
                   help="Skip conflict penalty when std(e) < this threshold.")
    p.add_argument("--min-client-weight", type=float, default=0.0,
                   help="Lower bound on alpha_i (post-normalization). 0 disables.")

    # baseline controls
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

    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--partition", type=str, default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    return parse_args_with_config(p)


def main():
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
