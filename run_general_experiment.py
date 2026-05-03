"""Single run: FedAvg and/or spectral Ours on torchvision general FL (Fashion-MNIST, etc.)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from flwr.common import ndarrays_to_parameters

from run_experiment import str2bool
from spectral_fl.config_io import add_config_argument, parse_args_with_config

CODE_VERSION = "general-fl-2026-05"


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
    p.add_argument("--method", type=str, default="both", choices=["both", "fedavg", "ours"])
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
    p.add_argument("--conflict-mix", type=float, default=0.2)
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
            "global_alignment",
        ],
    )
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        choices=["update", "normalized_update", "weight"],
        help="Representation used to build the client relation graph.",
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        choices=["update", "weight"],
        help="Object averaged with alpha_i to form the next global model.",
    )
    p.add_argument("--knn-k", type=int, default=2)
    p.add_argument("--edge-threshold", type=float, default=0.0)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--use-ema-graph", type=str2bool, default=True)

    p.add_argument("--disable-adaptive-tau", type=str2bool, default=False)
    p.add_argument("--fixed-tau", type=float, default=1.0)
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
        default=0.05,
        help="Floor on normalized aggregation weights after conflict shaping. Set 0 to disable.",
    )

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


def make_initial_parameters(model: torch.nn.Module, seed: int):
    saved_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        weights = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    finally:
        torch.random.set_rng_state(saved_state)
    return ndarrays_to_parameters(weights)


def build_general_meta(
    args: argparse.Namespace,
    class_distribution: List[List[int]],
    out_path: Path,
) -> Dict[str, Any]:
    ts = int(args.train_subset_size) if getattr(args, "train_subset_size", 0) else None
    tst = int(args.test_subset_size) if getattr(args, "test_subset_size", 0) else None
    cdim = int(args.projection_dim) if args.projection_dim is not None else int(args.compression_dim)
    cne = [sum(d) for d in class_distribution]
    return {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "run_tag": args.run_tag,
        "track": "general-fl",
        "output_path": str(out_path),
        "experiment": {
            "method": args.method,
            "dataset": args.dataset,
            "model": args.model,
            "lr": float(args.lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "partition": args.partition,
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "train_subset_size": ts,
            "test_subset_size": tst,
            "num_clients": int(args.num_clients),
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "seed": int(args.seed),
        },
        "graph": {
            "graph_mode": args.graph_mode,
            "graph_source": args.graph_source,
            "knn_k": int(args.knn_k),
            "edge_threshold": float(args.edge_threshold),
            "use_ema_graph": bool(args.use_ema_graph),
            "graph_ema_alpha": float(args.ema_alpha) if bool(args.use_ema_graph) else None,
            "graph_seed": int(args.graph_seed),
        },
        "tau": {
            "tau_max": float(args.tau_max),
            "tau_gain": float(args.tau_gain),
            "adaptive_tau_enabled": not bool(args.disable_adaptive_tau),
            "fixed_tau": float(args.fixed_tau),
        },
        "aggregation": {
            "aggregation_target": args.aggregation_target,
            "conflict_mix": float(args.conflict_mix),
            "warmup_rounds": int(args.warmup_rounds),
            "e_std_threshold": float(args.e_std_threshold),
            "min_client_weight": float(args.min_client_weight),
            "diagnostic_only": bool(args.diagnostic_only),
        },
        "compression": {
            "projection_dim": cdim,
            "projection_seed": int(args.compression_seed),
        },
        "seed": int(args.seed),
        "compression_dim": cdim,
        "compression_seed": int(args.compression_seed),
        "graph_mode": args.graph_mode,
        "graph_source": args.graph_source,
        "aggregation_target": args.aggregation_target,
        "knn_k": int(args.knn_k),
        "diagnostic_only": bool(args.diagnostic_only),
        "client_class_distribution": class_distribution,
        "client_num_examples": cne,
        "client_label_hist": class_distribution,
    }


def main():
    args = parse_args()
    from spectral_fl.flower_runner import main_dispatch

    main_dispatch(args, track="general-fl")


if __name__ == "__main__":
    main()
