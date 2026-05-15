"""Graph source selection for the graph-FL diagnostic strategy."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from spectral_fl.graph.sources.config import GraphSourceConfig, normalize_key
from spectral_fl.graph.sources.registry import (
    GraphSourceContext,
    build_registered_graph_source,
)
from spectral_fl.graph.sources.selection import (
    flatten_layerwise,
    normalize_vector,
    select_classifier_head,
    select_graph_layers,
)
from spectral_fl.projection import flatten_weights


def graph_vectors_for_spectral(
    *,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: Optional[List[NDArrays]] = None,
    config: GraphSourceConfig,
) -> Tuple[List[np.ndarray], str]:
    source = normalize_key(config.source)
    registered = build_registered_graph_source(
        GraphSourceContext(
            local_weights=local_weights,
            local_updates=local_updates,
            ema_updates=ema_updates,
            config=config,
        )
    )
    if registered is not None:
        return registered.vectors, registered.source_used

    ema_source = ema_updates if ema_updates is not None else local_updates
    if source in {"update", "delta", "update_delta", "pseudo_gradient", "pseudo_grad"}:
        return [flatten_weights(g_i) for g_i in local_updates], "update_delta"
    if source in {
        "ema_update",
        "client_ema_update",
        "momentum_update",
        "momentum_smoothed_update",
        "temporal_update",
    }:
        return [flatten_weights(g_i) for g_i in ema_source], "client_ema_update_delta"
    if source in {"normalized_update", "normalized_delta"}:
        return (
            [normalize_vector(flatten_weights(g_i)) for g_i in local_updates],
            "normalized_update_delta",
        )
    if source in {
        "normalized_ema_update",
        "ema_normalized_update",
        "normalized_client_ema_update",
        "client_ema_normalized_update",
    }:
        return (
            [normalize_vector(flatten_weights(g_i)) for g_i in ema_source],
            "normalized_client_ema_update_delta",
        )
    if source in {
        "layer_slice_update",
        "slice_update",
        "tail_update",
        "partial_update",
        "sliced_update",
    }:
        selected = [
            select_graph_layers(g_i, config.layer_start, config.layer_end)
            for g_i in local_updates
        ]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_weights(arrays) for arrays, _, _ in selected],
            f"update_delta_layers_{start}:{end}",
        )
    if source in {
        "layerwise_slice_update",
        "layerwise_tail_update",
        "layer_slice_normalized_update",
        "tail_layerwise_update",
    }:
        selected = [
            select_graph_layers(g_i, config.layer_start, config.layer_end)
            for g_i in local_updates
        ]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_layerwise(arrays, normalize_layers=True) for arrays, _, _ in selected],
            f"layerwise_normalized_update_delta_layers_{start}:{end}",
        )
    if source in {
        "layerwise_update",
        "layerwise_delta",
        "layerwise_normalized_update",
        "layer_normalized_update",
        "layerwise_normalized_delta",
        "layer_normalized_delta",
    }:
        return (
            [flatten_layerwise(g_i, normalize_layers=True) for g_i in local_updates],
            "layerwise_normalized_update_delta",
        )
    if source in {
        "layerwise_ema_update",
        "ema_layerwise_update",
        "layerwise_client_ema_update",
        "client_ema_layerwise_update",
    }:
        return (
            [flatten_layerwise(g_i, normalize_layers=True) for g_i in ema_source],
            "layerwise_normalized_client_ema_update_delta",
        )
    if source in {
        "classifier_head_update",
        "classifier_head_delta",
        "head_update",
        "head_delta",
        "head",
        "classifier_head",
    }:
        selected = [select_classifier_head(g_i) for g_i in local_updates]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_weights(arrays) for arrays, _, _ in selected],
            f"classifier_head_update_delta_layers_{start}:{end}",
        )
    if source in {
        "classifier_head_ema_update",
        "ema_classifier_head_update",
        "head_ema_update",
        "ema_head_update",
        "client_ema_head_update",
    }:
        selected = [select_classifier_head(g_i) for g_i in ema_source]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_weights(arrays) for arrays, _, _ in selected],
            f"classifier_head_client_ema_update_delta_layers_{start}:{end}",
        )
    if source in {
        "layerwise_classifier_head_update",
        "classifier_head_layerwise_update",
        "normalized_classifier_head_update",
        "layerwise_head_update",
        "head_layerwise_update",
        "normalized_head_update",
    }:
        selected = [select_classifier_head(g_i) for g_i in local_updates]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_layerwise(arrays, normalize_layers=True) for arrays, _, _ in selected],
            f"layerwise_normalized_classifier_head_update_delta_layers_{start}:{end}",
        )
    if source in {
        "layerwise_classifier_head_ema_update",
        "classifier_head_layerwise_ema_update",
        "normalized_classifier_head_ema_update",
        "layerwise_head_ema_update",
        "ema_layerwise_head_update",
        "normalized_head_ema_update",
    }:
        selected = [select_classifier_head(g_i) for g_i in ema_source]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_layerwise(arrays, normalize_layers=True) for arrays, _, _ in selected],
            f"layerwise_normalized_classifier_head_client_ema_update_delta_layers_{start}:{end}",
        )
    if source in {"weight", "weights", "model_weight", "model_weights", "state"}:
        return [flatten_weights(w_i) for w_i in local_weights], "local_weight"
    if source in {
        "layer_slice_weight",
        "slice_weight",
        "tail_weight",
        "partial_weight",
        "sliced_weight",
    }:
        selected = [
            select_graph_layers(w_i, config.layer_start, config.layer_end)
            for w_i in local_weights
        ]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_weights(arrays) for arrays, _, _ in selected],
            f"local_weight_layers_{start}:{end}",
        )
    if source in {
        "layerwise_slice_weight",
        "layerwise_tail_weight",
        "layer_slice_normalized_weight",
        "tail_layerwise_weight",
    }:
        selected = [
            select_graph_layers(w_i, config.layer_start, config.layer_end)
            for w_i in local_weights
        ]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_layerwise(arrays, normalize_layers=True) for arrays, _, _ in selected],
            f"layerwise_normalized_local_weight_layers_{start}:{end}",
        )
    if source in {
        "layerwise_weight",
        "layerwise_weights",
        "layerwise_normalized_weight",
        "layer_normalized_weight",
    }:
        return (
            [flatten_layerwise(w_i, normalize_layers=True) for w_i in local_weights],
            "layerwise_normalized_local_weight",
        )
    if source in {
        "classifier_head_weight",
        "classifier_head_weights",
        "head_weight",
        "head_weights",
        "classifier_weight",
        "classifier_weights",
    }:
        selected = [select_classifier_head(w_i) for w_i in local_weights]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_weights(arrays) for arrays, _, _ in selected],
            f"classifier_head_local_weight_layers_{start}:{end}",
        )
    if source in {
        "layerwise_classifier_head_weight",
        "classifier_head_layerwise_weight",
        "normalized_classifier_head_weight",
        "layerwise_head_weight",
        "head_layerwise_weight",
        "normalized_head_weight",
    }:
        selected = [select_classifier_head(w_i) for w_i in local_weights]
        start, end = selected[0][1], selected[0][2]
        return (
            [flatten_layerwise(arrays, normalize_layers=True) for arrays, _, _ in selected],
            f"layerwise_normalized_classifier_head_local_weight_layers_{start}:{end}",
        )
    raise ValueError(
        "Unknown graph_source "
        f"{config.source!r}; expected update, normalized_update, "
        "ema_update, normalized_ema_update, classifier_head_update, "
        "layer_slice_update, layerwise_update, weight, classifier_head_weight, "
        "layer_slice_weight, or layerwise_weight"
    )
