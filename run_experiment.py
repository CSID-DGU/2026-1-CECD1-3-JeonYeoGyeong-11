"""Single-experiment entrypoint for spectral conflict-aware FL on Cora.

Runs FedAvg, Ours, or both with identical initial parameters and saves a
result JSON containing:

    - meta:        full experiment metadata (graph mode, tau form, ...)
    - results:     per-method history (loss, accuracy, fit metrics) and
                   per-round diagnostic trace (H_spec, alpha, e_z, ...)

The diagnostic trace lives under ``results[method]["round_trace"]`` and is
the primary input for ``scripts/deep_dive_seed.py`` and the analysis tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from flwr.common import ndarrays_to_parameters

from spectral_fl.config_io import add_config_argument, parse_args_with_config
from spectral_fl.model import GCN
from spectral_fl.strategy import SpectralConflictAwareStrategy, TracingFedAvg


CODE_VERSION = "cora-fgl-2026-05"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


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
    p.add_argument("--conflict-mix", type=float, default=0.2)
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
    p.add_argument("--use-ema-graph", type=str2bool, default=True,
                   help="Apply EMA on the client graph between rounds (default: true)")

    # tau ablation
    p.add_argument("--disable-adaptive-tau", type=str2bool, default=False,
                   help="If true, use --fixed-tau every round (no tanh schedule).")
    p.add_argument("--fixed-tau", type=float, default=1.0)
    p.add_argument("--diagnostic-only", type=str2bool, default=False,
                   help="If true, log spectral diagnostics but aggregate with FedAvg weights.")

    # conservative penalty
    p.add_argument("--e-std-threshold", type=float, default=0.0,
                   help="Skip conflict penalty when std(e) < this threshold.")
    p.add_argument("--min-client-weight", type=float, default=0.0,
                   help="Lower bound on alpha_i (post-normalization). 0 disables.")

    # bookkeeping
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--partition", type=str, default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    return parse_args_with_config(p)


# =============================================================================
# History → dict / round_trace
# =============================================================================


def _to_pairs(items):
    return [[int(r), float(v)] for r, v in items]


def history_to_dict(history) -> dict:
    payload = {
        "losses_distributed": _to_pairs(history.losses_distributed),
        "losses_centralized": _to_pairs(history.losses_centralized),
    }
    for name in [
        "metrics_distributed",
        "metrics_centralized",
        "metrics_distributed_fit",
        "metrics_centralized_fit",
    ]:
        if hasattr(history, name):
            metric_obj = getattr(history, name)
            payload[name] = {k: _to_pairs(v) for k, v in metric_obj.items()}
    return payload


def pairs_to_round_map(pairs):
    return {int(r): float(v) for r, v in pairs}


def attach_round_trace(method: str, history_dict: dict, strategy, seed: int):
    loss_map = pairs_to_round_map(history_dict.get("losses_distributed", []))
    acc_map = pairs_to_round_map(
        history_dict.get("metrics_distributed", {}).get("accuracy", [])
    )
    trace = []
    for r in sorted(set(loss_map.keys()) | set(acc_map.keys())):
        trace.append(
            {
                "round": int(r),
                "method": method,
                "seed": int(seed),
                "accuracy": acc_map.get(r),
                "loss": loss_map.get(r),
            }
        )
    if hasattr(strategy, "round_logs"):
        by_round = {int(x["round"]): x for x in strategy.round_logs}
        for row in trace:
            detail = by_round.get(int(row["round"]))
            if detail is not None:
                row.update(detail)
    if hasattr(strategy, "eval_logs"):
        for row in trace:
            row["per_client_eval"] = strategy.eval_logs.get(int(row["round"]), [])
    return trace


# =============================================================================
# Initial parameters (parity between FedAvg and Ours)
# =============================================================================


def make_initial_parameters(in_dim: int, hidden_dim: int, out_dim: int, seed: int):
    """Build initial GCN weights deterministically and return Flower Parameters."""
    saved_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        weights = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    finally:
        torch.random.set_rng_state(saved_state)
    return ndarrays_to_parameters(weights)


def compute_client_class_distribution(client_graphs, out_dim: int) -> List[List[int]]:
    dists = []
    for cg in client_graphs:
        y = cg.data.y[cg.data.train_mask]
        counts = torch.bincount(y, minlength=out_dim).cpu().tolist()
        dists.append([int(v) for v in counts])
    return dists


# =============================================================================
# Strategy / simulation
# =============================================================================


def build_strategy(args, method: str, initial_parameters):
    def weighted_metric_avg(metrics):
        total = float(sum(num_examples for num_examples, _ in metrics))
        out = {}
        if total <= 0:
            return out
        keys = set()
        for _, m in metrics:
            keys.update(m.keys())
        for k in keys:
            num = 0.0
            den = 0.0
            for n, m in metrics:
                if k in m:
                    num += float(n) * float(m[k])
                    den += float(n)
            if den > 0:
                out[k] = num / den
        return out

    seed_for_clients = int(args.seed)

    def on_fit_config_fn(server_round: int):
        return {"server_round": int(server_round), "seed": seed_for_clients}

    def on_evaluate_config_fn(server_round: int):
        return {"server_round": int(server_round), "seed": seed_for_clients}

    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        fit_metrics_aggregation_fn=weighted_metric_avg,
        evaluate_metrics_aggregation_fn=weighted_metric_avg,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
    )
    if method == "fedavg":
        return TracingFedAvg(**common)
    if method == "ours":
        return SpectralConflictAwareStrategy(
            compression_dim=args.compression_dim,
            compression_seed=args.compression_seed,
            ema_alpha=args.ema_alpha,
            tau_gain=args.tau_gain,
            tau_max=args.tau_max,
            conflict_mix=args.conflict_mix,
            warmup_rounds=args.warmup_rounds,
            graph_mode=args.graph_mode,
            graph_source=args.graph_source,
            aggregation_target=args.aggregation_target,
            knn_k=args.knn_k,
            edge_threshold=args.edge_threshold,
            e_std_threshold=args.e_std_threshold,
            graph_seed=args.graph_seed,
            use_ema_graph=bool(args.use_ema_graph),
            adaptive_tau=not bool(args.disable_adaptive_tau),
            fixed_tau=args.fixed_tau,
            diagnostic_only=args.diagnostic_only,
            min_client_weight=args.min_client_weight,
            **common,
        )
    raise ValueError(f"Unknown method: {method}")


def print_final_summary(name: str, history_dict: dict):
    losses = history_dict.get("losses_distributed", [])
    if losses:
        print(f"[{name}] final distributed loss: {losses[-1][1]:.6f}")
    metrics_dist = history_dict.get("metrics_distributed", {})
    if "accuracy" in metrics_dist and metrics_dist["accuracy"]:
        print(f"[{name}] final distributed accuracy: {metrics_dist['accuracy'][-1][1]:.6f}")
    metrics_fit = history_dict.get("metrics_distributed_fit", {})
    for key in ["h_spec", "h_spec_ema", "tau", "e_mean", "e_max"]:
        if key in metrics_fit and metrics_fit[key]:
            print(f"[{name}] final {key}: {metrics_fit[key][-1][1]:.6f}")


# =============================================================================
# Metadata
# =============================================================================


def build_meta(args, client_class_distribution: List[List[int]], out_path: Path) -> Dict[str, Any]:
    """Full structured metadata for downstream analysis."""
    return {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "run_tag": args.run_tag,
        "output_path": str(out_path),

        "experiment": {
            "method": args.method,
            "dataset": "Cora",
            "model": "GCN",
            "hidden_dim": int(args.hidden_dim),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "partition": args.partition,
            "dirichlet_alpha": float(args.dirichlet_alpha),
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

        "init": {
            "initial_parameter_parity": True,
            "initial_seed": int(args.seed),
        },

        "compression": {
            "mode": "gaussian_jl_random_projection",
            "projection_dim": int(args.compression_dim),
            "projection_seed": int(args.compression_seed),
            "projection_type": "seeded_gaussian",
        },

        # legacy flat fields kept for backward-compatible analysis scripts
        "seed": int(args.seed),
        "num_clients": int(args.num_clients),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "tau_max": float(args.tau_max),
        "tau_gain": float(args.tau_gain),
        "conflict_mix": float(args.conflict_mix),
        "warmup_rounds": int(args.warmup_rounds),
        "compression_dim": int(args.compression_dim),
        "compression_seed": int(args.compression_seed),
        "graph_mode": args.graph_mode,
        "graph_source": args.graph_source,
        "aggregation_target": args.aggregation_target,
        "knn_k": int(args.knn_k),
        "edge_threshold": float(args.edge_threshold),
        "e_std_threshold": float(args.e_std_threshold),
        "graph_seed": int(args.graph_seed),
        "use_ema_graph": bool(args.use_ema_graph),
        "adaptive_tau_enabled": not bool(args.disable_adaptive_tau),
        "fixed_tau": float(args.fixed_tau),
        "min_client_weight": float(args.min_client_weight),
        "diagnostic_only": bool(args.diagnostic_only),
        "partition": args.partition,
        "dirichlet_alpha": float(args.dirichlet_alpha),
        "client_class_distribution": client_class_distribution,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    from spectral_fl.flower_runner import main_dispatch

    main_dispatch(args, track="cora-fgl")


if __name__ == "__main__":
    main()
