"""Round and graph-frequency diagnostics for graph-FL strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from spectral_fl.strategies.graphfl.aggregation import (
    compute_effective_clients,
    compute_entropy,
)


def heterogeneity(z_mat: np.ndarray, l_mat: np.ndarray, eps: float = 1e-12) -> float:
    numerator = float(np.trace(z_mat.T @ l_mat @ z_mat))
    denominator = float(np.linalg.norm(z_mat, ord="fro") ** 2 + eps)
    return numerator / denominator


def spectral_energy_diagnostics(
    z_mat: np.ndarray,
    l_mat: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Return graph Fourier decomposition diagnostics for update signal Z.

    Global band energy ratios are exact partitions in the Laplacian eigenbasis.
    Per-client band values are node-domain component norm ratios after
    reconstructing low/mid/high components, so they should be interpreted as
    client-local component strength rather than a strict per-client partition.
    """
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    z_hat = eigvecs.T @ z_mat
    energy = np.sum(z_hat * z_hat, axis=1).astype(np.float64)
    total_energy = float(np.sum(energy) + eps)
    ratios = energy / total_energy

    n_modes = int(eigvals.size)
    band_k = max(1, n_modes // 3) if n_modes > 0 else 0
    low_idx = list(range(0, band_k))
    high_start = max(n_modes - band_k, 0)
    high_idx = list(range(high_start, n_modes))
    mid_idx = list(range(band_k, high_start))

    low_ratio = float(np.sum(ratios[low_idx])) if low_idx else 0.0
    mid_ratio = float(np.sum(ratios[mid_idx])) if mid_idx else 0.0
    high_ratio = float(np.sum(ratios[high_idx])) if high_idx else 0.0
    dc_ratio = float(ratios[0]) if n_modes else 0.0
    highest_ratio = float(ratios[-1]) if n_modes else 0.0
    dominant_mode_idx = int(np.argmax(ratios)) if n_modes else -1

    def reconstruct(mode_indices: list[int]) -> np.ndarray:
        if not mode_indices:
            return np.zeros_like(z_mat, dtype=np.float64)
        u_band = eigvecs[:, mode_indices]
        return u_band @ z_hat[mode_indices, :]

    z_low = reconstruct(low_idx)
    z_mid = reconstruct(mid_idx)
    z_high = reconstruct(high_idx)
    client_norm = np.linalg.norm(z_mat, axis=1) + eps
    low_client_ratio = np.linalg.norm(z_low, axis=1) / client_norm
    mid_client_ratio = np.linalg.norm(z_mid, axis=1) / client_norm
    high_client_ratio = np.linalg.norm(z_high, axis=1) / client_norm

    if n_modes > 1:
        gaps = np.diff(eigvals)
        eigengap_max = float(np.max(gaps))
        eigengap_argmax = int(np.argmax(gaps))
        lambda_2 = float(eigvals[1])
    else:
        eigengap_max = 0.0
        eigengap_argmax = -1
        lambda_2 = 0.0

    entropy = 0.0
    if n_modes > 1:
        entropy = float(-np.sum(ratios * np.log(ratios + eps)) / np.log(n_modes))

    return {
        "laplacian_eigenvalues": [float(x) for x in eigvals.tolist()],
        "spectral_mode_energy_list": [float(x) for x in energy.tolist()],
        "spectral_energy_ratio_list": [float(x) for x in ratios.tolist()],
        "spectral_entropy": float(entropy),
        "frequency_band_indices": {
            "low": [int(x) for x in low_idx],
            "mid": [int(x) for x in mid_idx],
            "high": [int(x) for x in high_idx],
        },
        "low_frequency_energy_ratio": low_ratio,
        "mid_frequency_energy_ratio": mid_ratio,
        "high_frequency_energy_ratio": high_ratio,
        "low_frequency_band_k": int(band_k),
        "mid_frequency_band_k": int(len(mid_idx)),
        "high_frequency_band_k": int(band_k),
        "low_frequency_component_norm_ratio_list": [
            float(x) for x in low_client_ratio.tolist()
        ],
        "mid_frequency_component_norm_ratio_list": [
            float(x) for x in mid_client_ratio.tolist()
        ],
        "high_frequency_component_norm_ratio_list": [
            float(x) for x in high_client_ratio.tolist()
        ],
        "dc_energy_ratio": dc_ratio,
        "highest_frequency_energy_ratio": highest_ratio,
        "dominant_frequency_mode_index": int(dominant_mode_idx),
        "dominant_frequency_mode_lambda": (
            float(eigvals[dominant_mode_idx]) if dominant_mode_idx >= 0 else 0.0
        ),
        "dominant_frequency_energy_ratio": (
            float(ratios[dominant_mode_idx]) if dominant_mode_idx >= 0 else 0.0
        ),
        "high_to_low_energy_ratio": float(high_ratio / (low_ratio + eps)),
        "lambda_max": float(eigvals[-1]) if n_modes else 0.0,
        "lambda_2": lambda_2,
        "eigengap_max": eigengap_max,
        "eigengap_argmax": eigengap_argmax,
    }


def build_round_log(
    *,
    server_round: int,
    cids: List[str],
    spectral: Dict[str, Any],
    conflict: Dict[str, Any],
    update_stats: Dict[str, Any],
    graph: Dict[str, Any],
    alpha: Dict[str, Any],
    client: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full per-round diagnostic row emitted by the graph-FL strategy."""
    e = conflict["e"]
    e_z = conflict["e_z"]
    conflict_weight = conflict["conflict_weight"]
    raw_cw = conflict["raw_cw"]
    alpha_raw = alpha["alpha_raw"]
    alpha_norm = alpha["alpha_norm"]
    n_examples_arr = client["n_examples_arr"]
    active_client_mask = alpha["active_client_mask"]
    graph_diag_current = graph["graph_diag_current"]
    graph_diag = graph["graph_diag"]
    graph_meta = dict(graph.get("graph_meta") or {})
    graph_filter_strength = float(
        config.get("graph_filter_strength", config.get("spectral_filter_strength", 1.0))
    )

    pre_post_round = alpha.get("pre_post_round", {})
    di_pre = float(pre_post_round.get("di_pre", np.nan))
    di_post = float(pre_post_round.get("di_post", np.nan))
    neff_pre = float(pre_post_round.get("neff_pre", np.nan))
    neff_post = float(pre_post_round.get("neff_post", np.nan))
    align_mean_pre = float(pre_post_round.get("align_mean_pre", np.nan))
    align_mean_post = float(pre_post_round.get("align_mean_post", np.nan))
    loo_mean_pre = float(pre_post_round.get("loo_mean_pre", np.nan))
    loo_mean_post = float(pre_post_round.get("loo_mean_post", np.nan))

    q_pre = np.asarray(alpha.get("q_pre", []), dtype=np.float64)
    q_post = np.asarray(alpha.get("q_post", []), dtype=np.float64)
    align_pre = np.asarray(alpha.get("align_pre", []), dtype=np.float64)
    align_post = np.asarray(alpha.get("align_post", []), dtype=np.float64)
    loo_pre = np.asarray(alpha.get("loo_pre", []), dtype=np.float64)
    loo_post = np.asarray(alpha.get("loo_post", []), dtype=np.float64)

    return {
        "round": int(server_round),
        "cids": list(cids),

        # spectral signals
        "h_spec": float(spectral["h_spec"]),
        "h_spec_metric": float(spectral["h_spec"]),
        "h_spec_current": float(spectral["h_spec_current"]),
        "h_spec_used_graph": float(spectral["h_spec_current"]),
        "h_spec_raw_current_graph": float(spectral["h_spec_raw_current"]),
        "h_spec_normalized": float(spectral["h_spec_normalized"]),
        "h_spec_current_normalized": float(
            spectral["h_spec_current_normalized"]
        ),
        "h_spec_raw_current_graph_normalized": float(
            spectral["h_spec_raw_current_normalized"]
        ),
        "h_spec_metric_lambda_max": float(spectral["metric_lambda_max"]),
        "h_spec_ema": float(spectral["h_spec_ema"]),
        "h_spec_ema_candidate": float(spectral["h_spec_ema_candidate"]),
        "h_spec_metric_graph_source": spectral["metric_graph_source"],
        "h_spec_graph_uses_ema": bool(config["use_ema_graph"]),
        "graph_used_source": graph["graph_used_source"],
        "graph_state_active": bool(not spectral["in_warmup"]),
        "graph_state_skipped_for_warmup": bool(spectral["in_warmup"]),
        "tau": float(spectral["tau"]),
        "tau_source": str(config["tau_source"]),
        "tau_source_used": str(spectral["tau_source_used"]),
        "tau_source_signal": float(spectral["tau_source_signal"]),
        "tau_source_ema": float(spectral["tau_source_ema"]),
        "tau_source_ema_candidate": float(spectral["tau_source_ema_candidate"]),
        "adaptive_tau_enabled": bool(config["adaptive_tau"]),
        "fixed_tau": (
            float(config["fixed_tau"]) if not bool(config["adaptive_tau"]) else None
        ),
        **spectral["spectral_diag"],
        **spectral["filter_diag"],
        **spectral["target_filter_diag"],

        # conflict
        "e_list": [float(x) for x in e.tolist()],
        "e_z_list": [float(x) for x in e_z.tolist()],
        "conflict_weight_list": [float(x) for x in conflict_weight.tolist()],
        "raw_conflict_weight_list": [float(x) for x in raw_cw.tolist()],
        "e_mean": float(conflict["e_mean"]),
        "e_std": float(conflict["e_std"]),
        "min_e": float(np.min(e)),
        "max_e": float(np.max(e)),
        # legacy keys retained for older analysis scripts
        "mean_e": float(conflict["e_mean"]),
        "std_e": float(conflict["e_std"]),
        "conflict_penalty_disabled_due_to_estd": bool(conflict["estd_disabled"]),
        "graph_fallback_used": bool(conflict["graph_fallback_used"]),

        # update-space stats
        "z_norm_list": [float(x) for x in update_stats["z_norms"].tolist()],
        "delta_norm_list": [
            float(x) for x in update_stats["delta_norms"].tolist()
        ],
        "ema_delta_norm_list": [
            float(x) for x in update_stats["ema_delta_norms"].tolist()
        ],
        "ema_delta_norm_mean": float(np.mean(update_stats["ema_delta_norms"])),
        "weight_norm_list": [
            float(x) for x in update_stats["weight_norms"].tolist()
        ],
        "graph_source_norm_list": [
            float(x) for x in update_stats["graph_source_norms"].tolist()
        ],
        "client_update_ema_source": str(update_stats["ema_update_source"]),

        # graph
        "graph_method": str(config.get("graph_method", "none")),
        "graph_mode": str(config["graph_mode"]),
        "graph_source": str(config["graph_source"]),
        "graph_source_used": graph["graph_source_used"],
        "correction_family": str(config.get("correction_family", "")),
        "control_graph_mode": str(config.get("control_graph_mode", "")),
        "cluster_method": str(config.get("cluster_method", "")),
        "cluster_k_config": int(config.get("cluster_k", 0)),
        "cluster_auto_k": bool(config.get("cluster_auto_k", False)),
        "graph_kind": str(graph_meta.get("graph_kind", "real_graph")),
        "graph_meta": graph_meta,
        "cluster_ids": list(graph_meta.get("cluster_ids", [])),
        "knn_k": int(config["knn_k"]),
        "edge_threshold": float(config["edge_threshold"]),
        "graph_scale_sigma": float(config["graph_scale_sigma"]),
        "learned_graph_lambda": float(config["learned_graph_lambda"]),
        "graph_layer_start": int(config["graph_layer_start"]),
        "graph_layer_end": int(config["graph_layer_end"]),
        "use_ema_graph": bool(config["use_ema_graph"]),
        "raw_current_graph_density": float(graph_diag_current["graph_density"]),
        "raw_current_graph_degree_list": list(
            graph_diag_current["graph_degree_list"]
        ),
        "raw_current_number_of_edges": int(graph_diag_current["number_of_edges"]),
        "raw_current_graph_empty": bool(graph_diag_current["graph_empty"]),
        "graph_density": float(graph_diag["graph_density"]),
        "graph_degree_list": list(graph_diag["graph_degree_list"]),
        "number_of_edges": int(graph_diag["number_of_edges"]),
        "graph_empty": bool(graph_diag["graph_empty"]),
        "W_matrix": graph["w_matrix_log"],

        # alpha
        "alpha_mode": alpha["alpha_mode"],
        "aggregation_target": str(config["aggregation_target"]),
        "aggregation_target_used": alpha["aggregation_target_used"],
        **alpha["server_opt_diag"],
        "alpha_raw_list": [float(x) for x in alpha_raw.tolist()],
        "alpha_norm_list": [float(x) for x in alpha_norm.tolist()],
        # legacy alias used by some older analysis scripts
        "alpha_list": [float(x) for x in alpha_norm.tolist()],
        "min_alpha": float(np.min(alpha_norm)),
        "max_alpha": float(np.max(alpha_norm)),
        "entropy_alpha": float(compute_entropy(alpha_norm)),
        "effective_clients": float(compute_effective_clients(alpha_norm)),
        "di_pre": di_pre,
        "di_post": di_post,
        "neff_pre": neff_pre,
        "neff_post": neff_post,
        "alignment_mean_pre": align_mean_pre,
        "alignment_mean_post": align_mean_post,
        "loo_mean_pre": loo_mean_pre,
        "loo_mean_post": loo_mean_post,
        "q_pre_list": [float(x) for x in q_pre.tolist()],
        "q_post_list": [float(x) for x in q_post.tolist()],
        "alignment_pre_list": [float(x) for x in align_pre.tolist()],
        "alignment_post_list": [float(x) for x in align_post.tolist()],
        "loo_pre_list": [float(x) for x in loo_pre.tolist()],
        "loo_post_list": [float(x) for x in loo_post.tolist()],

        # client info
        "client_num_examples": [int(x) for x in n_examples_arr.tolist()],
        "active_client_mask": [bool(x) for x in active_client_mask.tolist()],
        "zero_example_client_count": int(np.sum(~active_client_mask)),
        "client_train_accuracy_list": client["client_train_acc"],
        "client_train_loss_list": client["client_train_loss"],

        # config flags
        "conflict_mix": float(config["conflict_mix"]),
        "warmup_rounds": int(config["warmup_rounds"]),
        "e_std_threshold": float(config["e_std_threshold"]),
        "min_client_weight": float(config["min_client_weight"]),
        "diagnostic_only": bool(config["diagnostic_only"]),
        "tau_source_config": str(config["tau_source"]),
        "graph_filter_strength": graph_filter_strength,
        "graph_filter_strength_config": graph_filter_strength,
        "spectral_filter_strength": graph_filter_strength,
        "spectral_filter_strength_config": graph_filter_strength,
        "client_update_ema_alpha": float(config["client_update_ema_alpha"]),
        "client_update_ema_alpha_config": float(
            config["client_update_ema_alpha"]
        ),
        "server_learning_rate_config": float(config["server_learning_rate"]),
        "server_momentum_config": float(config["server_momentum"]),
        "graph_source_config": str(config["graph_source"]),
        "graph_layer_start_config": int(config["graph_layer_start"]),
        "graph_layer_end_config": int(config["graph_layer_end"]),
        "aggregation_target_config": str(config["aggregation_target"]),
    }


def build_fit_metrics(
    *,
    spectral: Dict[str, Any],
    conflict: Dict[str, Any],
    alpha_norm: np.ndarray,
    graph_diag_current: Dict[str, Any],
    graph_diag: Dict[str, Any],
    filter_diag: Dict[str, Any],
    config: Dict[str, Any],
    pre_post_round: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Build scalar Flower fit metrics for the current graph-FL round."""
    pp = pre_post_round or {}
    graph_filter_strength = float(
        config.get("graph_filter_strength", config.get("spectral_filter_strength", 1.0))
    )
    return {
        "h_spec": float(spectral["h_spec"]),
        "h_spec_metric": float(spectral["h_spec"]),
        "h_spec_current": float(spectral["h_spec_current"]),
        "h_spec_used_graph": float(spectral["h_spec_current"]),
        "h_spec_raw_current_graph": float(spectral["h_spec_raw_current"]),
        "h_spec_ema": float(spectral["h_spec_ema"]),
        "tau": float(spectral["tau"]),
        "low_frequency_energy_ratio": float(
            spectral["spectral_diag"]["low_frequency_energy_ratio"]
        ),
        "mid_frequency_energy_ratio": float(
            spectral["spectral_diag"]["mid_frequency_energy_ratio"]
        ),
        "high_frequency_energy_ratio": float(
            spectral["spectral_diag"]["high_frequency_energy_ratio"]
        ),
        "high_to_low_energy_ratio": float(
            spectral["spectral_diag"]["high_to_low_energy_ratio"]
        ),
        "dominant_frequency_mode_index": float(
            spectral["spectral_diag"]["dominant_frequency_mode_index"]
        ),
        "dominant_frequency_mode_lambda": float(
            spectral["spectral_diag"]["dominant_frequency_mode_lambda"]
        ),
        "dominant_frequency_energy_ratio": float(
            spectral["spectral_diag"]["dominant_frequency_energy_ratio"]
        ),
        "spectral_entropy": float(spectral["spectral_diag"]["spectral_entropy"]),
        "eigengap_max": float(spectral["spectral_diag"]["eigengap_max"]),
        "graph_filter_strength": graph_filter_strength,
        "spectral_filter_strength": graph_filter_strength,
        "spectral_filter_gain_mean": float(
            filter_diag["spectral_filter_gain_mean"]
        ),
        "spectral_filter_output_energy_ratio": float(
            filter_diag["spectral_filter_output_energy_ratio"]
        ),
        "spectral_filter_residual_energy_ratio": float(
            filter_diag["spectral_filter_residual_energy_ratio"]
        ),
        "spectral_filter_suppressed_energy_ratio": float(
            filter_diag["spectral_filter_suppressed_energy_ratio"]
        ),
        "client_update_ema_alpha": float(config["client_update_ema_alpha"]),
        "server_learning_rate": float(config["server_learning_rate"]),
        "server_momentum": float(config["server_momentum"]),
        "server_momentum_active": float(bool(config["server_momentum"] > 0.0)),
        "e_mean": float(conflict["e_mean"]),
        "e_std": float(conflict["e_std"]),
        "e_min": float(np.min(conflict["e"])),
        "e_max": float(np.max(conflict["e"])),
        "alpha_min": float(np.min(alpha_norm)),
        "alpha_max": float(np.max(alpha_norm)),
        "alpha_entropy": float(compute_entropy(alpha_norm)),
        "alpha_effective_clients": float(compute_effective_clients(alpha_norm)),
        "di_pre": float(pp.get("di_pre", np.nan)),
        "di_post": float(pp.get("di_post", np.nan)),
        "neff_pre": float(pp.get("neff_pre", np.nan)),
        "neff_post": float(pp.get("neff_post", np.nan)),
        "alignment_mean_pre": float(pp.get("align_mean_pre", np.nan)),
        "alignment_mean_post": float(pp.get("align_mean_post", np.nan)),
        "loo_mean_pre": float(pp.get("loo_mean_pre", np.nan)),
        "loo_mean_post": float(pp.get("loo_mean_post", np.nan)),
        "raw_current_graph_density": float(graph_diag_current["graph_density"]),
        "raw_current_graph_empty": float(graph_diag_current["graph_empty"]),
        "raw_current_number_of_edges": float(graph_diag_current["number_of_edges"]),
        "graph_density": float(graph_diag["graph_density"]),
        "graph_empty": float(graph_diag["graph_empty"]),
        "number_of_edges": float(graph_diag["number_of_edges"]),
    }


__all__ = [
    "build_fit_metrics",
    "build_round_log",
    "heterogeneity",
    "spectral_energy_diagnostics",
]
