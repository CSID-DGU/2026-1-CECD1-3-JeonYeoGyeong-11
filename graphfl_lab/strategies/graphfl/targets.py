"""Aggregation target resolution for the graph-FL strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources import normalize_key
from graphfl_lab.projection import flatten_weights, unflatten_like
from graphfl_lab.strategies.graphfl.aggregation import weighted_average_by_alpha
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
)


@dataclass(frozen=True)
class AggregationTargetConfig:
    target: str = "update"
    filter_strength: float = 1.0


def _add_delta_to_global(current_global: NDArrays, delta: NDArrays) -> NDArrays:
    return [gp + gd for gp, gd in zip(current_global, delta)]


def _filter_client_arrays(
    *,
    arrays: List[NDArrays],
    l_mat: Optional[np.ndarray],
    filter_strength: float,
    target_name: str,
) -> Tuple[List[NDArrays], Dict[str, Any]]:
    if l_mat is None:
        raise ValueError(f"{target_name} requires a client Laplacian")
    flat_mat = np.stack(
        [flatten_weights(arr).astype(np.float64, copy=False) for arr in arrays],
        axis=0,
    )
    filtered_mat, filter_diag = apply_spectral_filter_with_diagnostics(
        z_mat=flat_mat,
        l_mat=l_mat,
        filter_strength=filter_strength,
    )
    filtered_arrays = [
        unflatten_like(filtered_mat[i], arrays[i]) for i in range(filtered_mat.shape[0])
    ]
    return filtered_arrays, filter_diag


def aggregate_target(
    *,
    current_global: NDArrays,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    alpha_norm: np.ndarray,
    config: AggregationTargetConfig,
    l_mat: Optional[np.ndarray] = None,
    ema_updates: Optional[List[NDArrays]] = None,
) -> Tuple[NDArrays, str, Dict[str, Any]]:
    target = normalize_key(config.target)
    ema_source = ema_updates if ema_updates is not None else local_updates
    if target in {"update", "delta", "update_delta"}:
        agg_delta = weighted_average_by_alpha(
            local_updates=local_updates, alphas=alpha_norm
        )
        return _add_delta_to_global(current_global, agg_delta), "update_delta", {}

    if target in {
        "spectral_filtered_update",
        "filtered_update",
        "graph_filtered_update",
        "lowpass_update",
        "low_pass_update",
        "spectral_update",
    }:
        filtered_updates, filter_diag = _filter_client_arrays(
            arrays=local_updates,
            l_mat=l_mat,
            filter_strength=float(config.filter_strength),
            target_name="graph_filtered_update",
        )
        agg_delta = weighted_average_by_alpha(
            local_updates=filtered_updates, alphas=alpha_norm
        )
        prefixed_filter_diag = {
            f"update_{key}": value for key, value in filter_diag.items()
        }
        return (
            _add_delta_to_global(current_global, agg_delta),
            "graph_filtered_update_delta",
            prefixed_filter_diag,
        )

    if target in {
        "spectral_filtered_ema_update",
        "filtered_ema_update",
        "graph_filtered_ema_update",
        "lowpass_ema_update",
        "low_pass_ema_update",
        "spectral_ema_update",
        "client_ema_spectral_filtered_update",
    }:
        filtered_updates, filter_diag = _filter_client_arrays(
            arrays=ema_source,
            l_mat=l_mat,
            filter_strength=float(config.filter_strength),
            target_name="graph_filtered_ema_update",
        )
        agg_delta = weighted_average_by_alpha(
            local_updates=filtered_updates, alphas=alpha_norm
        )
        prefixed_filter_diag = {
            f"ema_update_{key}": value for key, value in filter_diag.items()
        }
        return (
            _add_delta_to_global(current_global, agg_delta),
            "graph_filtered_client_ema_update_delta",
            prefixed_filter_diag,
        )

    if target in {"weight", "weights", "model_weight", "model_weights", "state"}:
        return (
            weighted_average_by_alpha(local_updates=local_weights, alphas=alpha_norm),
            "local_weight",
            {},
        )

    if target in {
        "spectral_filtered_weight",
        "filtered_weight",
        "graph_filtered_weight",
        "lowpass_weight",
        "low_pass_weight",
        "spectral_weight",
        "spectral_filtered_model_weight",
    }:
        filtered_weights, filter_diag = _filter_client_arrays(
            arrays=local_weights,
            l_mat=l_mat,
            filter_strength=float(config.filter_strength),
            target_name="graph_filtered_weight",
        )
        prefixed_filter_diag = {
            f"weight_{key}": value for key, value in filter_diag.items()
        }
        return (
            weighted_average_by_alpha(
                local_updates=filtered_weights, alphas=alpha_norm
            ),
            "graph_filtered_local_weight",
            prefixed_filter_diag,
        )

    raise ValueError(
        "Unknown aggregation_target "
        f"{config.target!r}; expected update, graph_filtered_update, "
        "graph_filtered_ema_update, weight, graph_filtered_weight, "
        "or their spectral_filtered_* compatibility aliases"
    )
