"""Diagnostic target flattening for GraphFL counterfactual metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregation_target_names,
    canonical_aggregation_target,
    evaluate_aggregation_target,
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
    num_clients = len(local_updates)
    alpha = np.full(
        num_clients,
        1.0 / max(float(num_clients), 1.0),
        dtype=np.float64,
    )
    requested = target_override or aggregation_target
    canonical = canonical_aggregation_target(requested)
    try:
        evaluation = evaluate_aggregation_target(
            current_global=current_global,
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            alpha_norm=alpha,
            l_mat=l_mat,
            config=AggregationTargetConfig(
                target=requested,
                filter_strength=filter_strength,
            ),
        )
    except ValueError as exc:
        if str(exc).startswith("Unknown aggregation_target"):
            raise ValueError(
                f"Unknown diagnostic aggregation_target {requested!r}"
            ) from exc
        raise

    metadata = dict(evaluation.metadata)
    if canonical not in set(aggregation_target_names()):
        for key in (
            "component_kind",
            "component_name",
            "plugin_module",
            "parameters",
            "target_used",
            "output_kind",
            "num_clients",
            "input_shape",
            "output_shape",
            "output_shapes",
        ):
            metadata.pop(key, None)
        if canonical in {"update", "delta", "update_delta"}:
            return evaluation.post_flat_updates, "update_delta", {}
        if canonical in {"weight", "weights", "model_weight", "model_weights", "state"}:
            return evaluation.post_flat_updates, "local_weight_delta", {}
        prefixes = ("update_", "ema_update_", "weight_")
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in metadata):
                metadata = {
                    key[len(prefix):] if key.startswith(prefix) else key: value
                    for key, value in metadata.items()
                }
                break
    return evaluation.post_flat_updates, evaluation.target_used, metadata


__all__ = ["flatten_diagnostic_post_updates"]
