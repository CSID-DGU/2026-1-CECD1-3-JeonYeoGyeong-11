"""Diagnostic target flattening for GraphFL counterfactual metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources import normalize_key
from graphfl_lab.projection import flatten_weights
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
)


def flatten_diagnostic_post_updates(
    *,
    current_global: NDArrays,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: List[NDArrays],
    l_mat: np.ndarray,
    aggregation_target: str,
    filter_strength: float,
    target_override: Optional[str] = None,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    target = normalize_key(target_override or aggregation_target)
    global_flat = flatten_weights(current_global).astype(np.float64, copy=False)

    if target in {"update", "delta", "update_delta"}:
        mat = _flat_matrix(local_updates)
        return mat, "update_delta", {}

    if target in {
        "spectral_filtered_update",
        "filtered_update",
        "graph_filtered_update",
        "lowpass_update",
        "low_pass_update",
        "spectral_update",
    }:
        mat = _flat_matrix(local_updates)
        filtered, diag = apply_spectral_filter_with_diagnostics(
            z_mat=mat,
            l_mat=l_mat,
            filter_strength=filter_strength,
        )
        return filtered, "spectral_filtered_update_delta", diag

    if target in {
        "spectral_filtered_ema_update",
        "filtered_ema_update",
        "graph_filtered_ema_update",
        "lowpass_ema_update",
        "low_pass_ema_update",
        "spectral_ema_update",
        "client_ema_spectral_filtered_update",
    }:
        mat = _flat_matrix(ema_updates)
        filtered, diag = apply_spectral_filter_with_diagnostics(
            z_mat=mat,
            l_mat=l_mat,
            filter_strength=filter_strength,
        )
        return filtered, "spectral_filtered_client_ema_update_delta", diag

    if target in {"weight", "weights", "model_weight", "model_weights", "state"}:
        mat = _flat_matrix(local_weights) - global_flat[None, :]
        return mat, "local_weight_delta", {}

    if target in {
        "spectral_filtered_weight",
        "filtered_weight",
        "graph_filtered_weight",
        "lowpass_weight",
        "low_pass_weight",
        "spectral_weight",
        "spectral_filtered_model_weight",
    }:
        mat = _flat_matrix(local_weights)
        filtered, diag = apply_spectral_filter_with_diagnostics(
            z_mat=mat,
            l_mat=l_mat,
            filter_strength=filter_strength,
        )
        return filtered - global_flat[None, :], "spectral_filtered_local_weight_delta", diag

    raise ValueError(f"Unknown diagnostic aggregation_target {target!r}")


def _flat_matrix(arrays: List[NDArrays]) -> np.ndarray:
    return np.stack(
        [flatten_weights(arr).astype(np.float64, copy=False) for arr in arrays],
        axis=0,
    )


__all__ = ["flatten_diagnostic_post_updates"]
