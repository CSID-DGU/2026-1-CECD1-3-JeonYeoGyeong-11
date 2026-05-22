"""Config context projection for GraphFL round logs and metrics."""

from __future__ import annotations

from typing import Any, Dict


def build_config_context(strategy: Any) -> Dict[str, Any]:
    return {
        "adaptive_tau": strategy.adaptive_tau,
        "aggregation_target": strategy.aggregation_target,
        "client_update_ema_alpha": strategy.client_update_ema_alpha,
        "conflict_mix": strategy.conflict_mix,
        "diagnostic_only": strategy.diagnostic_only,
        "diagnostics_enable": strategy.diagnostics_enable,
        "loo_enabled": strategy.loo_enabled,
        "edge_threshold": strategy.edge_threshold,
        "e_std_threshold": strategy.e_std_threshold,
        "fixed_tau": strategy.fixed_tau,
        "graph_layer_end": strategy.graph_layer_end,
        "graph_layer_start": strategy.graph_layer_start,
        "graph_mode": strategy.graph_mode,
        "graph_method": strategy.graph_method,
        "graph_scale_sigma": strategy.graph_scale_sigma,
        "graph_source": strategy.graph_source,
        "correction_family": strategy.correction_family,
        "control_graph_mode": strategy.control_graph_mode,
        "cluster_method": strategy.cluster_method,
        "cluster_k": strategy.cluster_k,
        "cluster_auto_k": strategy.cluster_auto_k,
        "graph_free_mode": strategy.graph_free_mode,
        "graph_free_gamma": strategy.graph_free_gamma,
        "clip_quantile": strategy.clip_quantile,
        "contribution_cap": strategy.contribution_cap,
        "knn_k": strategy.knn_k,
        "learned_graph_lambda": strategy.learned_graph_lambda,
        "min_client_weight": strategy.min_client_weight,
        "server_learning_rate": strategy.server_learning_rate,
        "server_momentum": strategy.server_momentum,
        "graph_filter_strength": strategy.graph_filter_strength,
        "tau_source": strategy.tau_source,
        "use_ema_graph": strategy.use_ema_graph,
        "warmup_rounds": strategy.warmup_rounds,
    }


__all__ = ["build_config_context"]
