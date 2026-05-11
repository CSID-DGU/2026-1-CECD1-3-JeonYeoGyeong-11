"""Single-experiment implementation for spectral conflict-aware FL on Cora.

Runs FedAvg, Ours, or both with identical initial parameters and saves a
result JSON containing:

    - meta:        full experiment metadata (graph mode, tau form, ...)
    - results:     per-method history (loss, accuracy, fit metrics) and
                   per-round diagnostic trace (H_spec, alpha, e_z, ...)

The diagnostic trace lives under ``results[method]["round_trace"]`` and is
the primary input for ``scripts/analysis/deep_dive_seed.py`` and the analysis tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from flwr.common import ndarrays_to_parameters

from spectral_fl.models.cora import GCN
from spectral_fl.graph.presets import apply_graph_preset_to_namespace
from spectral_fl.strategies.baselines import (
    TracingDominanceAwareFedAvgM,
    TracingFedAdagrad,
    TracingFedAvg,
    TracingFedAvgM,
    TracingFedAdam,
    TracingGraphSmoothFedAvgM,
    TracingFedMedian,
    TracingFedNova,
    TracingFedProx,
    TracingFedTrimmedAvg,
    TracingFedYogi,
    TracingFedSim,
)
from spectral_fl.strategies.spectral.strategy import SpectralConflictAwareStrategy


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



# =============================================================================
# History ??dict / round_trace
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
    apply_graph_preset_to_namespace(args)

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
    if method == "fedavgm":
        return TracingFedAvgM(
            server_learning_rate=float(args.server_learning_rate),
            server_momentum=float(args.server_momentum),
            **common,
        )
    if method == "fedadagrad":
        return TracingFedAdagrad(
            eta=float(args.fedopt_eta),
            eta_l=float(args.fedopt_eta_l),
            tau=float(args.fedopt_tau),
            **common,
        )
    if method == "fedadam":
        return TracingFedAdam(
            eta=float(args.fedopt_eta),
            eta_l=float(args.fedopt_eta_l),
            beta_1=float(args.fedopt_beta1),
            beta_2=float(args.fedopt_beta2),
            tau=float(args.fedopt_tau),
            **common,
        )
    if method == "fedyogi":
        return TracingFedYogi(
            eta=float(args.fedopt_eta),
            eta_l=float(args.fedopt_eta_l),
            beta_1=float(args.fedopt_beta1),
            beta_2=float(args.fedopt_beta2),
            tau=float(args.fedopt_tau),
            **common,
        )
    if method == "fednova":
        return TracingFedNova(
            server_learning_rate=float(args.server_learning_rate),
            **common,
        )
    if method == "fedprox":
        return TracingFedProx(proximal_mu=float(args.fedprox_mu), **common)
    if method == "fedmedian":
        return TracingFedMedian(**common)
    if method == "fedtrimmedavg":
        return TracingFedTrimmedAvg(beta=float(args.trimmed_beta), **common)
    if method == "fedsim":
        return TracingFedSim(
            compression_dim=args.compression_dim,
            compression_seed=args.compression_seed,
            graph_mode=args.graph_mode,
            graph_source=args.graph_source,
            knn_k=args.knn_k,
            edge_threshold=args.edge_threshold,
            graph_scale_sigma=args.graph_scale_sigma,
            learned_graph_lambda=args.learned_graph_lambda,
            graph_seed=args.graph_seed,
            **common,
        )
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
            graph_scale_sigma=args.graph_scale_sigma,
            learned_graph_lambda=args.learned_graph_lambda,
            graph_layer_start=args.graph_layer_start,
            graph_layer_end=args.graph_layer_end,
            e_std_threshold=args.e_std_threshold,
            graph_seed=args.graph_seed,
            correction_family=getattr(args, "correction_family", "real_graph"),
            control_graph_mode=getattr(args, "control_graph_mode", "random"),
            cluster_method=getattr(args, "cluster_method", "none"),
            cluster_k=int(getattr(args, "cluster_k", 0)),
            cluster_auto_k=bool(getattr(args, "cluster_auto_k", False)),
            use_ema_graph=bool(args.use_ema_graph),
            adaptive_tau=not bool(args.disable_adaptive_tau),
            fixed_tau=args.fixed_tau,
            tau_source=args.tau_source,
            spectral_filter_strength=args.spectral_filter_strength,
            client_update_ema_alpha=args.client_update_ema_alpha,
            diagnostics_enable=bool(getattr(args, "diagnostics_enable", False)),
            loo_enabled=bool(getattr(args, "loo_enabled", False)),
            diagnostics_artifact_dir=str(Path(args.out_dir) / "diagnostics"),
            diagnostics_run_id=f"{method}_{getattr(args, 'run_tag', '') or f'seed{int(args.seed)}'}",
            graph_free_mode=str(getattr(args, "graph_free_mode", "none")),
            graph_free_gamma=float(getattr(args, "graph_free_gamma", 1.0)),
            clip_quantile=float(getattr(args, "clip_quantile", 0.9)),
            contribution_cap=float(getattr(args, "contribution_cap", 0.0)),
            server_learning_rate=args.ours_server_learning_rate,
            server_momentum=args.ours_server_momentum,
            diagnostic_only=args.diagnostic_only,
            min_client_weight=args.min_client_weight,
            **common,
        )
    if method == "graph_smooth":
        return TracingGraphSmoothFedAvgM(
            graph_preset=str(getattr(args, "graph_preset", "none")),
            graph_variant=str(getattr(args, "graph_variant", "update")),
            graph_mode=str(getattr(args, "graph_mode", "dense")),
            graph_source=str(args.graph_source),
            knn_k=int(getattr(args, "knn_k", 2)),
            edge_threshold=float(getattr(args, "edge_threshold", 0.0)),
            graph_scale_sigma=float(getattr(args, "graph_scale_sigma", 1.0)),
            learned_graph_lambda=float(getattr(args, "learned_graph_lambda", 1.0)),
            graph_layer_start=int(getattr(args, "graph_layer_start", 0)),
            graph_layer_end=int(getattr(args, "graph_layer_end", 0)),
            graph_smoothing_operator=str(
                getattr(args, "graph_smoothing_operator", "laplacian")
            ),
            graph_dominance_gamma=float(getattr(args, "graph_dominance_gamma", 1.0)),
            dominance_mode=str(getattr(args, "graph_dominance_mode", "sample")),
            dominance_cap_kappa=float(getattr(args, "graph_dominance_cap_kappa", 2.0)),
            dominance_soft_tau=float(getattr(args, "graph_dominance_soft_tau", 5.0)),
            client_update_ema_alpha=float(
                getattr(args, "client_update_ema_alpha", 0.8)
            ),
            compression_dim=int(args.compression_dim),
            compression_seed=int(args.compression_seed),
            graph_seed=int(args.graph_seed),
            graph_smoothing_lambda=float(getattr(args, "graph_smoothing_lambda", 0.05)),
            graph_laplacian_type=str(getattr(args, "graph_laplacian_type", "unnormalized")),
            graph_zero_diagonal=bool(getattr(args, "graph_zero_diagonal", True)),
            server_learning_rate=float(args.server_learning_rate),
            server_momentum=float(args.server_momentum),
            **common,
        )
    if method == "dominance_aware":
        return TracingDominanceAwareFedAvgM(
            dominance_mode=str(getattr(args, "dominance_mode", "fedavgm")),
            dominance_tau=float(getattr(args, "dominance_tau", 1.0)),
            dominance_threshold=float(getattr(args, "dominance_threshold", 0.35)),
            clip_norm=float(getattr(args, "dominance_clip_norm", 0.0)),
            clip_percentile=float(getattr(args, "dominance_clip_percentile", 0.75)),
            contribution_cap=float(getattr(args, "dominance_contribution_cap", 0.0)),
            contribution_cap_percentile=float(
                getattr(args, "dominance_contribution_cap_percentile", 0.75)
            ),
            contribution_cap_kappa=float(
                getattr(args, "dominance_contribution_cap_kappa", 0.0)
            ),
            server_learning_rate=float(args.server_learning_rate),
            server_momentum=float(args.server_momentum),
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
            "graph_preset": str(getattr(args, "graph_preset", "none")),
            "graph_source": args.graph_source,
            "knn_k": int(args.knn_k),
            "edge_threshold": float(args.edge_threshold),
            "graph_scale_sigma": float(args.graph_scale_sigma),
            "learned_graph_lambda": float(args.learned_graph_lambda),
            "graph_layer_start": int(args.graph_layer_start),
            "graph_layer_end": int(args.graph_layer_end),
            "use_ema_graph": bool(args.use_ema_graph),
            "graph_ema_alpha": float(args.ema_alpha) if bool(args.use_ema_graph) else None,
            "graph_seed": int(args.graph_seed),
        },

        "tau": {
            "tau_max": float(args.tau_max),
            "tau_gain": float(args.tau_gain),
            "adaptive_tau_enabled": not bool(args.disable_adaptive_tau),
            "fixed_tau": float(args.fixed_tau),
            "tau_source": args.tau_source,
            "spectral_filter_strength": float(args.spectral_filter_strength),
            "client_update_ema_alpha": float(args.client_update_ema_alpha),
        },

        "aggregation": {
            "aggregation_target": args.aggregation_target,
            "conflict_mix": float(args.conflict_mix),
            "warmup_rounds": int(args.warmup_rounds),
            "e_std_threshold": float(args.e_std_threshold),
            "min_client_weight": float(args.min_client_weight),
            "diagnostic_only": bool(args.diagnostic_only),
        },

        "baselines": {
            "server_learning_rate": float(args.server_learning_rate),
            "server_momentum": float(args.server_momentum),
            "ours_server_learning_rate": float(args.ours_server_learning_rate),
            "ours_server_momentum": float(args.ours_server_momentum),
            "fedprox_mu": float(args.fedprox_mu),
            "fedopt_eta": float(args.fedopt_eta),
            "fedopt_eta_l": float(args.fedopt_eta_l),
            "fedopt_beta1": float(args.fedopt_beta1),
            "fedopt_beta2": float(args.fedopt_beta2),
            "fedopt_tau": float(args.fedopt_tau),
            "trimmed_beta": float(args.trimmed_beta),
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
        "graph_preset": str(getattr(args, "graph_preset", "none")),
        "graph_source": args.graph_source,
        "aggregation_target": args.aggregation_target,
        "knn_k": int(args.knn_k),
        "edge_threshold": float(args.edge_threshold),
        "graph_scale_sigma": float(args.graph_scale_sigma),
        "learned_graph_lambda": float(args.learned_graph_lambda),
        "graph_layer_start": int(args.graph_layer_start),
        "graph_layer_end": int(args.graph_layer_end),
        "e_std_threshold": float(args.e_std_threshold),
        "graph_seed": int(args.graph_seed),
        "use_ema_graph": bool(args.use_ema_graph),
        "adaptive_tau_enabled": not bool(args.disable_adaptive_tau),
        "fixed_tau": float(args.fixed_tau),
        "tau_source": args.tau_source,
        "spectral_filter_strength": float(args.spectral_filter_strength),
        "min_client_weight": float(args.min_client_weight),
        "diagnostic_only": bool(args.diagnostic_only),
        "partition": args.partition,
        "dirichlet_alpha": float(args.dirichlet_alpha),
        "client_class_distribution": client_class_distribution,
    }


# =============================================================================
# Main
# =============================================================================


def run(args):
    from spectral_fl.flower_runner import main_dispatch

    main_dispatch(args, track="cora-fgl")
