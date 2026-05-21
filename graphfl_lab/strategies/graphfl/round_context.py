"""Round log and metric context builders for GraphFL strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def build_spectral_context(
    *,
    spectral_metrics: Any,
    h_spec_ema: float,
    in_warmup: bool,
    conflict_metrics: Any,
    target_filter_diag: Dict[str, Any],
    diagnostic_filter_diag: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "h_spec": spectral_metrics.h_spec,
        "h_spec_current": spectral_metrics.h_spec_current,
        "h_spec_raw_current": spectral_metrics.h_spec_raw_current,
        "h_spec_normalized": spectral_metrics.h_spec_normalized,
        "h_spec_current_normalized": spectral_metrics.h_spec_current_normalized,
        "h_spec_raw_current_normalized": spectral_metrics.h_spec_raw_current_normalized,
        "metric_lambda_max": spectral_metrics.metric_lambda_max,
        "h_spec_ema": h_spec_ema,
        "h_spec_ema_candidate": spectral_metrics.h_spec_ema_candidate,
        "metric_graph_source": spectral_metrics.metric_graph_source,
        "in_warmup": in_warmup,
        "tau": conflict_metrics.tau,
        "tau_source_used": conflict_metrics.tau_source.source_used,
        "tau_source_signal": conflict_metrics.tau_source.source_signal,
        "tau_source_ema": conflict_metrics.tau_source.ema_value,
        "tau_source_ema_candidate": conflict_metrics.tau_source.ema_candidate,
        "spectral_diag": spectral_metrics.spectral_diag,
        "filter_diag": conflict_metrics.filter_diag,
        "target_filter_diag": target_filter_diag,
        "diagnostic_filter_diag": diagnostic_filter_diag,
    }


def build_conflict_context(
    *,
    conflict_metrics: Any,
    conflict_weight: np.ndarray,
    graph_fallback_used: bool,
) -> Dict[str, Any]:
    return {
        "e": conflict_metrics.e,
        "e_z": conflict_metrics.e_z,
        "conflict_weight": conflict_weight,
        "raw_cw": conflict_metrics.raw_cw,
        "e_mean": conflict_metrics.e_mean,
        "e_std": conflict_metrics.e_std_raw,
        "estd_disabled": conflict_metrics.estd_disabled,
        "graph_fallback_used": graph_fallback_used,
    }


def build_update_context(
    *,
    z_norms: np.ndarray,
    update_space: Any,
    graph_source_norms: np.ndarray,
    ema_update_source: str,
) -> Dict[str, Any]:
    return {
        "z_norms": z_norms,
        "delta_norms": update_space.delta_norms,
        "ema_delta_norms": update_space.ema_delta_norms,
        "weight_norms": update_space.weight_norms,
        "graph_source_norms": graph_source_norms,
        "ema_update_source": ema_update_source,
    }


def build_graph_context(
    *,
    graph_source_used: str,
    graph_used_source: str,
    graph_meta: Dict[str, Any],
    graph_diag_current: Dict[str, Any],
    graph_diag: Dict[str, Any],
    w_matrix_log: Optional[List[List[float]]],
) -> Dict[str, Any]:
    return {
        "graph_source_used": graph_source_used,
        "graph_used_source": graph_used_source,
        "graph_meta": graph_meta,
        "graph_diag_current": graph_diag_current,
        "graph_diag": graph_diag,
        "w_matrix_log": w_matrix_log,
    }


def build_alpha_context(
    *,
    alpha_raw: np.ndarray,
    alpha_norm: np.ndarray,
    alpha_mode: str,
    active_client_mask: np.ndarray,
    aggregation_target_used: str,
    diagnostic_target_used: str,
    server_opt_diag: Dict[str, Any],
    pre_post: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "alpha_raw": alpha_raw,
        "alpha_norm": alpha_norm,
        "alpha_mode": alpha_mode,
        "active_client_mask": active_client_mask,
        "aggregation_target_used": aggregation_target_used,
        "diagnostic_target_used": diagnostic_target_used,
        "server_opt_diag": server_opt_diag,
        "pre_post_round": pre_post["round"],
        "q_pre": pre_post["q_pre"],
        "q_post": pre_post["q_post"],
        "align_pre": pre_post["align_pre"],
        "align_post": pre_post["align_post"],
        "loo_pre": pre_post["loo_pre"],
        "loo_post": pre_post["loo_post"],
    }


def build_client_context(
    *,
    n_examples_arr: np.ndarray,
    client_train_acc: Any,
    client_train_loss: Any,
) -> Dict[str, Any]:
    return {
        "n_examples_arr": n_examples_arr,
        "client_train_acc": client_train_acc,
        "client_train_loss": client_train_loss,
    }


__all__ = [
    "build_alpha_context",
    "build_client_context",
    "build_conflict_context",
    "build_graph_context",
    "build_spectral_context",
    "build_update_context",
]
