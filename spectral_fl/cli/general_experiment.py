"""CLI parser for torchvision general FL experiments."""

from __future__ import annotations

import argparse

from spectral_fl.cli.argparse_types import str2bool
from spectral_fl.config_io import add_config_argument, parse_args_with_config
from spectral_fl.experiments.general import single_run as _experiment
from spectral_fl.graph.presets import graph_preset_names

# Compatibility re-exports for older imports from spectral_fl.cli.general_experiment.
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
            "graph_smooth",
            "dominance_aware",
        ],
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="fashionmnist",
        choices=["fashionmnist", "mnist", "cifar10"],
    )
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"])

    p.add_argument("--compression-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.0)
    p.add_argument("--warmup-rounds", type=int, default=2)

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
        "--graph-preset",
        type=str,
        default="none",
        choices=graph_preset_names(),
        help=(
            "Prior-work-inspired graph design preset. When set, it overrides "
            "graph source/mode and related graph knobs."
        ),
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
    p.add_argument("--use-ema-graph", type=str2bool, default=True)
    p.add_argument(
        "--graph-variant",
        type=str,
        default="update",
        choices=["update", "random", "shuffled", "uniform", "identity"],
        help="Phase-2 graph informativeness variant used by method=graph_smooth.",
    )
    p.add_argument(
        "--graph-smoothing-lambda",
        type=float,
        default=0.05,
        help="Smoothing coefficient lambda for method=graph_smooth.",
    )
    p.add_argument(
        "--graph-smoothing-operator",
        type=str,
        default="laplacian",
        choices=[
            "laplacian",
            "unnormalized_laplacian",
            "normalized_laplacian",
            "random_walk_laplacian",
            "residual",
            "residual_neighbor_mixing",
            "dominance_residual",
            "signed_conflict_attenuation",
            "dominance_aware_attenuation",
        ],
        help="Graph smoothing operator for method=graph_smooth.",
    )
    p.add_argument(
        "--graph-dominance-gamma",
        type=float,
        default=1.0,
        help="Gamma for dominance-aware residual graph mixing.",
    )
    p.add_argument(
        "--graph-dominance-mode",
        type=str,
        default="sample",
        choices=["sample", "uniform", "cap", "soft_reweight"],
        help="Aggregation weighting mode for method=graph_smooth.",
    )
    p.add_argument(
        "--graph-dominance-cap-kappa",
        type=float,
        default=2.0,
        help="kappa for graph_smooth cap mode (cap=kappa*median(q)).",
    )
    p.add_argument(
        "--graph-dominance-soft-tau",
        type=float,
        default=5.0,
        help="tau for graph_smooth soft_reweight mode.",
    )
    p.add_argument(
        "--graph-laplacian-type",
        type=str,
        default="unnormalized",
        choices=["unnormalized", "normalized", "random_walk"],
        help="Laplacian type for method=graph_smooth.",
    )
    p.add_argument(
        "--graph-zero-diagonal",
        type=str2bool,
        default=True,
        help="Whether to force A_ii=0 before building the Laplacian.",
    )
    p.add_argument(
        "--dominance-mode",
        type=str,
        default="fedavgm",
        choices=[
            "fedavgm",
            "uniform",
            "norm_clip",
            "contribution_cap",
            "soft_reweight",
            "n_eff_aware",
            "triggered_soft_reweight",
        ],
        help="Dominance-aware correction mode for method=dominance_aware.",
    )
    p.add_argument(
        "--dominance-tau",
        type=float,
        default=1.0,
        help="Tau value for dominance-aware soft reweighting.",
    )
    p.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.35,
        help="Trigger threshold on DI_t for triggered_soft_reweight.",
    )
    p.add_argument(
        "--dominance-clip-norm",
        type=float,
        default=0.0,
        help="Clip norm c for norm_clip mode. <=0 uses percentile of round update norms.",
    )
    p.add_argument(
        "--dominance-clip-percentile",
        type=float,
        default=0.75,
        help="Percentile used for adaptive clip norm when dominance-clip-norm<=0.",
    )
    p.add_argument(
        "--dominance-contribution-cap",
        type=float,
        default=0.0,
        help="Contribution cap for contribution_cap mode. <=0 uses percentile of q_i.",
    )
    p.add_argument(
        "--dominance-contribution-cap-percentile",
        type=float,
        default=0.75,
        help="Percentile used for adaptive contribution cap when cap<=0.",
    )
    p.add_argument(
        "--dominance-contribution-cap-kappa",
        type=float,
        default=0.0,
        help="If >0, use cap=kappa*median(q_i) for contribution_cap mode.",
    )

    p.add_argument("--disable-adaptive-tau", type=str2bool, default=False)
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
    p.add_argument(
        "--diagnostic-only",
        type=str2bool,
        default=False,
        help="If true, log spectral diagnostics but aggregate with FedAvg weights.",
    )

    p.add_argument(
        "--e-std-threshold",
        type=float,
        default=0.02,
        help="If std(e) across clients is below this, skip conflict penalty (FedAvg-style weights). Set 0 to disable.",
    )
    p.add_argument(
        "--min-client-weight",
        type=float,
        default=0.0,
        help="Floor on normalized aggregation weights after conflict shaping. Set 0 to disable.",
    )

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

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data/torchvision")
    p.add_argument("--out-dir", type=str, default="./outputs_general")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--partition", type=str, default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument(
        "--train-subset-size",
        type=int,
        default=0,
        help="If >0, subsample training set to this many samples. 0 = full train.",
    )
    p.add_argument("--test-subset-size", type=int, default=0)
    p.add_argument("--projection-dim", type=int, default=None)
    return parse_args_with_config(p)


def main():
    _experiment.run(parse_args())


if __name__ == "__main__":
    main()
