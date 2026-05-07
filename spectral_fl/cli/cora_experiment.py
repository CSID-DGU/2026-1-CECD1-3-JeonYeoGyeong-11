"""CLI parser for Cora spectral FL experiments."""

from __future__ import annotations

import argparse

from spectral_fl.cli.argparse_types import str2bool
from spectral_fl.config_io import add_config_argument, parse_args_with_config
from spectral_fl.experiments.cora import single_run as _experiment

# Compatibility re-exports for older imports from spectral_fl.cli.cora_experiment.
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

    # spectral / aggregation
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
        choices=[
            "dense",
            "knn",
            "mutual_knn",
            "threshold",
            "uniform",
            "random",
            "magnitude",
            "magnitude_aware",
            "magnitude_knn",
            "global_alignment",
            "signed_abs",
            "signed_abs_knn",
            "negative",
            "negative_knn",
            "rbf",
            "rbf_knn",
            "learned_smooth",
            "learned_smooth_knn",
        ],
    )
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        choices=[
            "update",
            "ema_update",
            "normalized_ema_update",
            "layerwise_ema_update",
            "normalized_update",
            "layer_slice_update",
            "layerwise_slice_update",
            "layerwise_update",
            "classifier_head_update",
            "classifier_head_ema_update",
            "layerwise_classifier_head_update",
            "layerwise_classifier_head_ema_update",
            "weight",
            "classifier_head_weight",
            "layerwise_classifier_head_weight",
            "layer_slice_weight",
            "layerwise_slice_weight",
            "layerwise_weight",
        ],
        help="Representation used to build the client relation graph.",
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        choices=[
            "update",
            "spectral_filtered_update",
            "spectral_filtered_ema_update",
            "weight",
            "spectral_filtered_weight",
        ],
        help="Object averaged with alpha_i to form the next global model.",
    )
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
        "--spectral-filter-strength",
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
        help="EMA coefficient for client update signals used by ema_update graph sources or spectral_filtered_ema_update.",
    )
    p.add_argument("--diagnostic-only", type=str2bool, default=False,
                   help="If true, log spectral diagnostics but aggregate with FedAvg weights.")

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
