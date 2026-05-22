"""Single-run implementation for torchvision vision FL."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from flwr.common import ndarrays_to_parameters

CODE_VERSION = "vision-fl-2026-05"



def make_initial_parameters(model: torch.nn.Module, seed: int):
    saved_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        weights = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    finally:
        torch.random.set_rng_state(saved_state)
    return ndarrays_to_parameters(weights)


def prepare_artifact_layout(args: argparse.Namespace, out_path: Path) -> Dict[str, str]:
    """Return stable artifact layout paths without forcing directory creation."""
    base_dir = out_path.parent
    diagnostics_dir = base_dir / "diagnostics"
    plots_dir = base_dir / "plots"
    reports_dir = base_dir / "reports"
    snapshots_dir = diagnostics_dir / "snapshots"
    logs_dir = base_dir / "logs"
    return {
        "base_dir": str(base_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "plots_dir": str(plots_dir),
        "reports_dir": str(reports_dir),
        "snapshots_dir": str(snapshots_dir),
        "logs_dir": str(logs_dir),
        "round_metrics_csv": str(diagnostics_dir / "round_metrics.csv"),
        "client_metrics_csv": str(diagnostics_dir / "client_metrics.csv"),
        "graph_stats_csv": str(diagnostics_dir / "graph_stats.csv"),
        "counterfactual_metrics_csv": str(diagnostics_dir / "counterfactual_metrics.csv"),
        "module_traces_jsonl": str(diagnostics_dir / "module_traces.jsonl"),
        "round_trace_jsonl": str(logs_dir / "round_trace.jsonl"),
        "dashboard_mockup_md": str(reports_dir / "dashboard_mockup.md"),
        "dashboard_mockup_html": str(reports_dir / "dashboard_mockup.html"),
        "enabled": bool(getattr(args, "diagnostics_enable", False)),
    }


def build_vision_meta(
    args: argparse.Namespace,
    class_distribution: List[List[int]],
    out_path: Path,
) -> Dict[str, Any]:
    ts = int(args.train_subset_size) if getattr(args, "train_subset_size", 0) else None
    tst = int(args.test_subset_size) if getattr(args, "test_subset_size", 0) else None
    cdim = int(args.projection_dim) if args.projection_dim is not None else int(args.compression_dim)
    cne = [sum(d) for d in class_distribution]
    artifact_layout = prepare_artifact_layout(args, out_path)
    return {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "run_tag": args.run_tag,
        "track": "vision-fl",
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
            "graph_plugin": str(getattr(args, "graph_plugin", "")),
            "graph_preset": str(getattr(args, "graph_preset", "none")),
            "graph_design": str(getattr(args, "graph_design", "none")),
            "graph_method": str(getattr(args, "graph_method", "none")),
            "graph_source": args.graph_source,
            "graph_variant": getattr(args, "graph_variant", "update"),
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
        "graph_smoothing": {
            "lambda": float(getattr(args, "graph_smoothing_lambda", 0.05)),
            "operator": str(getattr(args, "graph_smoothing_operator", "laplacian")),
            "dominance_gamma": float(getattr(args, "graph_dominance_gamma", 1.0)),
            "laplacian_type": str(getattr(args, "graph_laplacian_type", "unnormalized")),
            "zero_diagonal": bool(getattr(args, "graph_zero_diagonal", True)),
        },
        "tau": {
            "tau_max": float(args.tau_max),
            "tau_gain": float(args.tau_gain),
            "adaptive_tau_enabled": not bool(args.disable_adaptive_tau),
            "fixed_tau": float(args.fixed_tau),
            "tau_source": args.tau_source,
            "graph_filter_strength": float(args.graph_filter_strength),
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
        "correction": {
            "family": str(getattr(args, "correction_family", "real_graph")),
            "control_graph_mode": str(getattr(args, "control_graph_mode", "random")),
            "cluster_method": str(getattr(args, "cluster_method", "none")),
            "cluster_k": int(getattr(args, "cluster_k", 0)),
            "cluster_auto_k": bool(getattr(args, "cluster_auto_k", False)),
            "graph_free_mode": str(getattr(args, "graph_free_mode", "none")),
            "graph_free_gamma": float(getattr(args, "graph_free_gamma", 1.0)),
            "clip_quantile": float(getattr(args, "clip_quantile", 0.9)),
            "contribution_cap": float(getattr(args, "contribution_cap", 0.0)),
        },
        "diagnostics": {
            "enabled": bool(getattr(args, "diagnostics_enable", False)),
            "save_round_graphs": bool(getattr(args, "save_round_graphs", False)),
            "graph_snapshot_rounds": str(getattr(args, "graph_snapshot_rounds", "")),
            "save_update_arrays": bool(getattr(args, "save_update_arrays", False)),
            "loo_enabled": bool(getattr(args, "loo_enabled", False)),
        },
        "artifacts": artifact_layout,
        "dominance": {
            "mode": str(getattr(args, "dominance_mode", "fedavgm")),
            "tau": float(getattr(args, "dominance_tau", 1.0)),
            "threshold": float(getattr(args, "dominance_threshold", 0.35)),
            "clip_norm": float(getattr(args, "dominance_clip_norm", 0.0)),
            "clip_percentile": float(getattr(args, "dominance_clip_percentile", 0.75)),
            "contribution_cap": float(getattr(args, "dominance_contribution_cap", 0.0)),
            "contribution_cap_percentile": float(
                getattr(args, "dominance_contribution_cap_percentile", 0.75)
            ),
            "contribution_cap_kappa": float(
                getattr(args, "dominance_contribution_cap_kappa", 0.0)
            ),
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
        "compression": {
            "projection_dim": cdim,
            "projection_seed": int(args.compression_seed),
        },
        "seed": int(args.seed),
        "compression_dim": cdim,
        "compression_seed": int(args.compression_seed),
        "graph_mode": args.graph_mode,
        "graph_plugin": str(getattr(args, "graph_plugin", "")),
        "graph_preset": str(getattr(args, "graph_preset", "none")),
        "graph_design": str(getattr(args, "graph_design", "none")),
        "graph_method": str(getattr(args, "graph_method", "none")),
        "graph_source": args.graph_source,
        "graph_variant": getattr(args, "graph_variant", "update"),
        "aggregation_target": args.aggregation_target,
        "knn_k": int(args.knn_k),
        "edge_threshold": float(args.edge_threshold),
        "graph_scale_sigma": float(args.graph_scale_sigma),
        "learned_graph_lambda": float(args.learned_graph_lambda),
        "graph_layer_start": int(args.graph_layer_start),
        "graph_layer_end": int(args.graph_layer_end),
        "diagnostic_only": bool(args.diagnostic_only),
        "tau_source": args.tau_source,
        "graph_filter_strength": float(args.graph_filter_strength),
        "client_update_ema_alpha": float(args.client_update_ema_alpha),
        "graph_smoothing_lambda": float(getattr(args, "graph_smoothing_lambda", 0.05)),
        "graph_smoothing_operator": str(
            getattr(args, "graph_smoothing_operator", "laplacian")
        ),
        "graph_dominance_gamma": float(getattr(args, "graph_dominance_gamma", 1.0)),
        "graph_laplacian_type": str(getattr(args, "graph_laplacian_type", "unnormalized")),
        "graph_zero_diagonal": bool(getattr(args, "graph_zero_diagonal", True)),
        "dominance_mode": str(getattr(args, "dominance_mode", "fedavgm")),
        "dominance_tau": float(getattr(args, "dominance_tau", 1.0)),
        "dominance_threshold": float(getattr(args, "dominance_threshold", 0.35)),
        "dominance_clip_norm": float(getattr(args, "dominance_clip_norm", 0.0)),
        "dominance_clip_percentile": float(getattr(args, "dominance_clip_percentile", 0.75)),
        "dominance_contribution_cap": float(
            getattr(args, "dominance_contribution_cap", 0.0)
        ),
        "dominance_contribution_cap_percentile": float(
            getattr(args, "dominance_contribution_cap_percentile", 0.75)
        ),
        "dominance_contribution_cap_kappa": float(
            getattr(args, "dominance_contribution_cap_kappa", 0.0)
        ),
        "client_class_distribution": class_distribution,
        "client_num_examples": cne,
        "client_label_hist": class_distribution,
        "correction_family": str(getattr(args, "correction_family", "real_graph")),
        "control_graph_mode": str(getattr(args, "control_graph_mode", "random")),
        "cluster_method": str(getattr(args, "cluster_method", "none")),
        "cluster_k": int(getattr(args, "cluster_k", 0)),
        "cluster_auto_k": bool(getattr(args, "cluster_auto_k", False)),
        "graph_free_mode": str(getattr(args, "graph_free_mode", "none")),
        "graph_free_gamma": float(getattr(args, "graph_free_gamma", 1.0)),
        "clip_quantile": float(getattr(args, "clip_quantile", 0.9)),
        "contribution_cap": float(getattr(args, "contribution_cap", 0.0)),
        "diagnostics_enable": bool(getattr(args, "diagnostics_enable", False)),
        "save_round_graphs": bool(getattr(args, "save_round_graphs", False)),
        "graph_snapshot_rounds": str(getattr(args, "graph_snapshot_rounds", "")),
        "save_update_arrays": bool(getattr(args, "save_update_arrays", False)),
        "loo_enabled": bool(getattr(args, "loo_enabled", False)),
        "artifact_layout": artifact_layout,
    }


def build_general_meta(
    args: argparse.Namespace,
    class_distribution: List[List[int]],
    out_path: Path,
) -> Dict[str, Any]:
    """Compatibility alias for older imports."""
    return build_vision_meta(args, class_distribution, out_path)


def run(args):
    from graphfl_lab.flower_runner import main_dispatch

    main_dispatch(args, track="vision-fl")
