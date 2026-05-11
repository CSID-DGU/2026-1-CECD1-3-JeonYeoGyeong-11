"""Flower run-config defaults and conversion helpers."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict

from flwr.common import Context


DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "track": "general-fl",
    "method": "ours",
    "dataset": "fashionmnist",
    "model": "mlp",
    "num-clients": 5,
    "rounds": 3,
    "local-epochs": 1,
    "batch-size": 64,
    "hidden-dim": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight-decay": 5e-4,
    "compression-dim": 256,
    "compression-seed": 0,
    "ema-alpha": 0.8,
    "tau-gain": 2.0,
    "tau-max": 2.0,
    "conflict-mix": 0.0,
    "warmup-rounds": 2,
    "graph-mode": "dense",
    "graph-preset": "none",
    "graph-source": "update",
    "aggregation-target": "spectral_filtered_update",
    "knn-k": 2,
    "edge-threshold": 0.0,
    "graph-scale-sigma": 1.0,
    "learned-graph-lambda": 1.0,
    "graph-layer-start": 0,
    "graph-layer-end": 0,
    "graph-seed": 0,
    "graph-variant": "update",
    "graph-smoothing-lambda": 0.05,
    "graph-smoothing-operator": "laplacian",
    "graph-dominance-gamma": 1.0,
    "graph-dominance-mode": "sample",
    "graph-dominance-cap-kappa": 2.0,
    "graph-dominance-soft-tau": 5.0,
    "graph-laplacian-type": "unnormalized",
    "graph-zero-diagonal": True,
    "dominance-mode": "fedavgm",
    "dominance-tau": 1.0,
    "dominance-threshold": 0.35,
    "dominance-clip-norm": 0.0,
    "dominance-clip-percentile": 0.75,
    "dominance-contribution-cap": 0.0,
    "dominance-contribution-cap-percentile": 0.75,
    "dominance-contribution-cap-kappa": 0.0,
    "use-ema-graph": True,
    "disable-adaptive-tau": False,
    "fixed-tau": 1.0,
    "tau-source": "h_spec",
    "spectral-filter-strength": 1.0,
    "client-update-ema-alpha": 0.8,
    "diagnostic-only": False,
    "e-std-threshold": 0.0,
    "min-client-weight": 0.0,
    "server-learning-rate": 1.0,
    "server-momentum": 0.9,
    "ours-server-learning-rate": 1.0,
    "ours-server-momentum": 0.0,
    "fedprox-mu": 0.01,
    "fedopt-eta": 0.1,
    "fedopt-eta-l": 0.1,
    "fedopt-beta1": 0.9,
    "fedopt-beta2": 0.99,
    "fedopt-tau": 1e-9,
    "trimmed-beta": 0.1,
    "seed": 42,
    "data-root": "./data/torchvision",
    "out-dir": "./experiments_current/app_smoke",
    "run-tag": "",
    "partition": "dirichlet",
    "dirichlet-alpha": 0.1,
    "train-subset-size": 0,
    "test-subset-size": 0,
    "projection-dim": 0,
}


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_s = str(value).strip().lower()
    if value_s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def merged_run_config(context: Context) -> Dict[str, Any]:
    merged = dict(DEFAULT_RUN_CONFIG)
    merged.update(dict(context.run_config or {}))
    return merged


def args_from_context(context: Context) -> Namespace:
    cfg = merged_run_config(context)
    projection_dim_raw = int(cfg.get("projection-dim", 0) or 0)
    projection_dim = projection_dim_raw if projection_dim_raw > 0 else None
    compression_dim = (
        projection_dim if projection_dim is not None else int(cfg["compression-dim"])
    )

    return Namespace(
        track=str(cfg["track"]),
        method=str(cfg["method"]),
        dataset=str(cfg["dataset"]),
        model=str(cfg["model"]),
        num_clients=int(cfg["num-clients"]),
        rounds=int(cfg["rounds"]),
        local_epochs=int(cfg["local-epochs"]),
        batch_size=int(cfg["batch-size"]),
        hidden_dim=int(cfg["hidden-dim"]),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        compression_dim=int(compression_dim),
        compression_seed=int(cfg["compression-seed"]),
        ema_alpha=float(cfg["ema-alpha"]),
        tau_gain=float(cfg["tau-gain"]),
        tau_max=float(cfg["tau-max"]),
        conflict_mix=float(cfg["conflict-mix"]),
        warmup_rounds=int(cfg["warmup-rounds"]),
        graph_mode=str(cfg["graph-mode"]),
        graph_preset=str(cfg.get("graph-preset", "none")),
        graph_source=str(cfg["graph-source"]),
        aggregation_target=str(cfg["aggregation-target"]),
        knn_k=int(cfg["knn-k"]),
        edge_threshold=float(cfg["edge-threshold"]),
        graph_scale_sigma=float(cfg["graph-scale-sigma"]),
        learned_graph_lambda=float(cfg["learned-graph-lambda"]),
        graph_layer_start=int(cfg["graph-layer-start"]),
        graph_layer_end=int(cfg["graph-layer-end"]),
        graph_seed=int(cfg["graph-seed"]),
        graph_variant=str(cfg["graph-variant"]),
        graph_smoothing_lambda=float(cfg["graph-smoothing-lambda"]),
        graph_smoothing_operator=str(cfg["graph-smoothing-operator"]),
        graph_dominance_gamma=float(cfg["graph-dominance-gamma"]),
        graph_dominance_mode=str(cfg["graph-dominance-mode"]),
        graph_dominance_cap_kappa=float(cfg["graph-dominance-cap-kappa"]),
        graph_dominance_soft_tau=float(cfg["graph-dominance-soft-tau"]),
        graph_laplacian_type=str(cfg["graph-laplacian-type"]),
        graph_zero_diagonal=bool_value(cfg["graph-zero-diagonal"]),
        dominance_mode=str(cfg["dominance-mode"]),
        dominance_tau=float(cfg["dominance-tau"]),
        dominance_threshold=float(cfg["dominance-threshold"]),
        dominance_clip_norm=float(cfg["dominance-clip-norm"]),
        dominance_clip_percentile=float(cfg["dominance-clip-percentile"]),
        dominance_contribution_cap=float(cfg["dominance-contribution-cap"]),
        dominance_contribution_cap_percentile=float(
            cfg["dominance-contribution-cap-percentile"]
        ),
        dominance_contribution_cap_kappa=float(cfg["dominance-contribution-cap-kappa"]),
        use_ema_graph=bool_value(cfg["use-ema-graph"]),
        disable_adaptive_tau=bool_value(cfg["disable-adaptive-tau"]),
        fixed_tau=float(cfg["fixed-tau"]),
        tau_source=str(cfg["tau-source"]),
        spectral_filter_strength=float(cfg["spectral-filter-strength"]),
        client_update_ema_alpha=float(cfg["client-update-ema-alpha"]),
        diagnostic_only=bool_value(cfg["diagnostic-only"]),
        e_std_threshold=float(cfg["e-std-threshold"]),
        min_client_weight=float(cfg["min-client-weight"]),
        server_learning_rate=float(cfg["server-learning-rate"]),
        server_momentum=float(cfg["server-momentum"]),
        ours_server_learning_rate=float(cfg["ours-server-learning-rate"]),
        ours_server_momentum=float(cfg["ours-server-momentum"]),
        fedprox_mu=float(cfg["fedprox-mu"]),
        fedopt_eta=float(cfg["fedopt-eta"]),
        fedopt_eta_l=float(cfg["fedopt-eta-l"]),
        fedopt_beta1=float(cfg["fedopt-beta1"]),
        fedopt_beta2=float(cfg["fedopt-beta2"]),
        fedopt_tau=float(cfg["fedopt-tau"]),
        trimmed_beta=float(cfg["trimmed-beta"]),
        seed=int(cfg["seed"]),
        data_root=str(cfg["data-root"]),
        out_dir=str(cfg["out-dir"]),
        run_tag=str(cfg["run-tag"]),
        partition=str(cfg["partition"]),
        dirichlet_alpha=float(cfg["dirichlet-alpha"]),
        train_subset_size=int(cfg["train-subset-size"]),
        test_subset_size=int(cfg["test-subset-size"]),
        projection_dim=projection_dim,
    )
