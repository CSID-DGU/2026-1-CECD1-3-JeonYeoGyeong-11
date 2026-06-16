"""Aggregation target resolution and extension registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources import normalize_key
from graphfl_lab.projection import flatten_weights, unflatten_like
from graphfl_lab.strategies.graphfl.aggregation import weighted_average_by_alpha
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
)

_AGGREGATION_TARGET_LEGACY_ALIASES = {
    "spectral_filtered_update": "graph_filtered_update",
    "spectral_filtered_ema_update": "graph_filtered_ema_update",
    "spectral_filtered_weight": "graph_filtered_weight",
    "spectral_filtered_update_delta": "graph_filtered_update",
    "spectral_filtered_client_ema_update_delta": "graph_filtered_ema_update",
    "spectral_filtered_local_weight_delta": "graph_filtered_weight",
    "spectral_filtered_model_weight": "graph_filtered_weight",
    "client_ema_spectral_filtered_update": "graph_filtered_ema_update",
    "spectral_update": "graph_filtered_update",
    "spectral_ema_update": "graph_filtered_ema_update",
    "spectral_weight": "graph_filtered_weight",
}


def canonical_aggregation_target(target: str) -> str:
    key = normalize_key(target)
    return _AGGREGATION_TARGET_LEGACY_ALIASES.get(key, key)


@dataclass(frozen=True)
class AggregationTargetConfig:
    target: str = "update"
    filter_strength: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationTargetContext:
    current_global: NDArrays
    local_weights: List[NDArrays]
    local_updates: List[NDArrays]
    alpha_norm: np.ndarray
    config: AggregationTargetConfig
    l_mat: Optional[np.ndarray] = None
    ema_updates: Optional[List[NDArrays]] = None


@dataclass(frozen=True)
class AggregationTargetResult:
    """Per-client arrays consumed by both aggregation and diagnostics."""

    post_local_updates: List[NDArrays]
    target_used: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    output_kind: str = "update_delta"


@dataclass(frozen=True)
class AggregationTargetEvaluation:
    candidate_global: NDArrays
    post_flat_updates: np.ndarray
    target_used: str
    metadata: Mapping[str, Any]
    output_kind: str


AggregationTargetBuilder = Callable[
    [AggregationTargetContext],
    AggregationTargetResult,
]

_AGGREGATION_TARGET_BUILDERS: dict[str, AggregationTargetBuilder] = {}


def register_aggregation_target(
    *names: str,
    override: bool = False,
) -> Callable[[AggregationTargetBuilder], AggregationTargetBuilder]:
    """Register a custom ``--aggregation-target`` implementation."""
    if not names:
        raise ValueError("register_aggregation_target requires at least one target name")

    def decorator(func: AggregationTargetBuilder) -> AggregationTargetBuilder:
        for name in names:
            key = canonical_aggregation_target(name)
            if not key:
                raise ValueError("aggregation target name cannot be empty")
            if key in _AGGREGATION_TARGET_BUILDERS and not bool(override):
                raise ValueError(
                    f"Aggregation target {name!r} is already registered; "
                    "pass override=True to replace it"
                )
            _AGGREGATION_TARGET_BUILDERS[key] = func
        return func

    return decorator


def unregister_aggregation_target(name: str) -> None:
    _AGGREGATION_TARGET_BUILDERS.pop(canonical_aggregation_target(name), None)


def aggregation_target_names() -> list[str]:
    return sorted(_AGGREGATION_TARGET_BUILDERS)


def _add_delta_to_global(current_global: NDArrays, delta: NDArrays) -> NDArrays:
    return [gp + gd for gp, gd in zip(current_global, delta)]


def add_weighted_delta(
    current_global: NDArrays,
    local_updates: List[NDArrays],
    alpha_norm: np.ndarray,
) -> NDArrays:
    delta = weighted_average_by_alpha(local_updates=local_updates, alphas=alpha_norm)
    return _add_delta_to_global(current_global, delta)


def graph_filter_client_arrays(
    arrays: List[NDArrays],
    l_mat: Optional[np.ndarray],
    filter_strength: float,
    *,
    target_name: str = "custom_aggregation",
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


def mix_client_arrays(
    left: List[NDArrays],
    right: List[NDArrays],
    ratio: float,
) -> List[NDArrays]:
    if len(left) != len(right):
        raise ValueError("client array collections must have the same length")
    beta = float(ratio)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("mix ratio must be in [0, 1]")
    mixed: List[NDArrays] = []
    for left_client, right_client in zip(left, right):
        if len(left_client) != len(right_client):
            raise ValueError("client array layer counts must match")
        mixed.append(
            [
                ((1.0 - beta) * a + beta * b).astype(a.dtype, copy=False)
                for a, b in zip(left_client, right_client)
            ]
        )
    return mixed


def _filter_client_arrays(
    *,
    arrays: List[NDArrays],
    l_mat: Optional[np.ndarray],
    filter_strength: float,
    target_name: str,
) -> Tuple[List[NDArrays], Dict[str, Any]]:
    return graph_filter_client_arrays(
        arrays,
        l_mat,
        filter_strength,
        target_name=target_name,
    )


def _builtin_target_result(
    context: AggregationTargetContext,
    target: str,
) -> AggregationTargetResult:
    ema_source = (
        context.ema_updates
        if context.ema_updates is not None
        else context.local_updates
    )
    if target in {"update", "delta", "update_delta"}:
        return AggregationTargetResult(
            post_local_updates=context.local_updates,
            target_used="update_delta",
        )

    if target in {
        "filtered_update",
        "graph_filtered_update",
        "lowpass_update",
        "low_pass_update",
    }:
        filtered, diag = _filter_client_arrays(
            arrays=context.local_updates,
            l_mat=context.l_mat,
            filter_strength=float(context.config.filter_strength),
            target_name="graph_filtered_update",
        )
        return AggregationTargetResult(
            post_local_updates=filtered,
            target_used="graph_filtered_update_delta",
            metadata={f"update_{key}": value for key, value in diag.items()},
        )

    if target in {
        "filtered_ema_update",
        "graph_filtered_ema_update",
        "lowpass_ema_update",
        "low_pass_ema_update",
    }:
        filtered, diag = _filter_client_arrays(
            arrays=ema_source,
            l_mat=context.l_mat,
            filter_strength=float(context.config.filter_strength),
            target_name="graph_filtered_ema_update",
        )
        return AggregationTargetResult(
            post_local_updates=filtered,
            target_used="graph_filtered_client_ema_update_delta",
            metadata={f"ema_update_{key}": value for key, value in diag.items()},
        )

    if target in {"weight", "weights", "model_weight", "model_weights", "state"}:
        return AggregationTargetResult(
            post_local_updates=context.local_weights,
            target_used="local_weight",
            output_kind="local_weight",
        )

    if target in {
        "filtered_weight",
        "graph_filtered_weight",
        "lowpass_weight",
        "low_pass_weight",
    }:
        filtered, diag = _filter_client_arrays(
            arrays=context.local_weights,
            l_mat=context.l_mat,
            filter_strength=float(context.config.filter_strength),
            target_name="graph_filtered_weight",
        )
        return AggregationTargetResult(
            post_local_updates=filtered,
            target_used="graph_filtered_local_weight",
            metadata={f"weight_{key}": value for key, value in diag.items()},
            output_kind="local_weight",
        )

    raise ValueError(
        "Unknown aggregation_target "
        f"{context.config.target!r}; expected a built-in target or a target "
        "registered by --graph-plugin"
    )


def _validate_post_local_updates(
    result: AggregationTargetResult,
    context: AggregationTargetContext,
    target: str,
) -> AggregationTargetResult:
    arrays = list(result.post_local_updates)
    if len(arrays) != len(context.local_updates):
        raise ValueError(
            f"Aggregation target {target!r} returned {len(arrays)} clients; "
            f"expected {len(context.local_updates)}"
        )
    template = (
        context.local_weights
        if result.output_kind == "local_weight"
        else context.local_updates
    )
    normalized: List[NDArrays] = []
    for client_index, (client_arrays, client_template) in enumerate(
        zip(arrays, template)
    ):
        if len(client_arrays) != len(client_template):
            raise ValueError(
                f"Aggregation target {target!r} returned an invalid layer count "
                f"for client {client_index}"
            )
        client_out: NDArrays = []
        for layer_index, (array, expected) in enumerate(
            zip(client_arrays, client_template)
        ):
            value = np.asarray(array)
            if value.shape != expected.shape:
                raise ValueError(
                    f"Aggregation target {target!r} returned shape {value.shape} "
                    f"for client {client_index} layer {layer_index}; "
                    f"expected {expected.shape}"
                )
            if not bool(np.all(np.isfinite(value))):
                raise ValueError(
                    f"Aggregation target {target!r} returned non-finite values"
                )
            client_out.append(value.astype(expected.dtype, copy=False))
        normalized.append(client_out)
    metadata = dict(result.metadata)
    metadata.setdefault("component_kind", "AggregationOperator")
    metadata.setdefault("component_name", target)
    metadata.setdefault("plugin_module", __name__)
    metadata.setdefault("parameters", dict(context.config.parameters))
    metadata.setdefault("target_used", str(result.target_used))
    metadata.setdefault("output_kind", result.output_kind)
    metadata.setdefault("num_clients", len(normalized))
    metadata.setdefault(
        "input_shape",
        [
            [list(layer.shape) for layer in client]
            for client in context.local_updates
        ],
    )
    metadata.setdefault(
        "output_shape",
        [[list(layer.shape) for layer in client] for client in normalized],
    )
    return AggregationTargetResult(
        post_local_updates=normalized,
        target_used=str(result.target_used),
        metadata=metadata,
        output_kind=str(result.output_kind),
    )


def evaluate_aggregation_target(
    *,
    current_global: NDArrays,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    alpha_norm: np.ndarray,
    config: AggregationTargetConfig,
    l_mat: Optional[np.ndarray] = None,
    ema_updates: Optional[List[NDArrays]] = None,
) -> AggregationTargetEvaluation:
    target = canonical_aggregation_target(config.target)
    context = AggregationTargetContext(
        current_global=current_global,
        local_weights=local_weights,
        local_updates=local_updates,
        alpha_norm=np.asarray(alpha_norm, dtype=np.float64),
        config=config,
        l_mat=l_mat,
        ema_updates=ema_updates,
    )
    builder = _AGGREGATION_TARGET_BUILDERS.get(target)
    if builder is None:
        raw_result = _builtin_target_result(context, target)
    else:
        raw_result = builder(context)
        if not isinstance(raw_result, AggregationTargetResult):
            raise TypeError(
                f"Aggregation target {target!r} must return AggregationTargetResult"
            )
        metadata = dict(raw_result.metadata)
        metadata.setdefault("plugin_module", builder.__module__)
        raw_result = AggregationTargetResult(
            post_local_updates=raw_result.post_local_updates,
            target_used=raw_result.target_used,
            metadata=metadata,
            output_kind=raw_result.output_kind,
        )
    result = _validate_post_local_updates(raw_result, context, target)
    if result.output_kind == "local_weight":
        candidate_global = weighted_average_by_alpha(
            local_updates=result.post_local_updates,
            alphas=context.alpha_norm,
        )
        global_flat = flatten_weights(current_global).astype(np.float64, copy=False)
        post_flat = np.stack(
            [
                flatten_weights(arrays).astype(np.float64, copy=False)
                - global_flat
                for arrays in result.post_local_updates
            ],
            axis=0,
        )
    elif result.output_kind == "update_delta":
        candidate_global = add_weighted_delta(
            current_global,
            result.post_local_updates,
            context.alpha_norm,
        )
        post_flat = np.stack(
            [
                flatten_weights(arrays).astype(np.float64, copy=False)
                for arrays in result.post_local_updates
            ],
            axis=0,
        )
    else:
        raise ValueError(
            f"Aggregation target {target!r} returned unknown output_kind "
            f"{result.output_kind!r}"
        )
    return AggregationTargetEvaluation(
        candidate_global=candidate_global,
        post_flat_updates=post_flat,
        target_used=result.target_used,
        metadata=dict(result.metadata),
        output_kind=result.output_kind,
    )


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
    """Backward-compatible wrapper returning the candidate global model."""
    evaluation = evaluate_aggregation_target(
        current_global=current_global,
        local_weights=local_weights,
        local_updates=local_updates,
        alpha_norm=alpha_norm,
        config=config,
        l_mat=l_mat,
        ema_updates=ema_updates,
    )
    metadata = dict(evaluation.metadata)
    if canonical_aggregation_target(config.target) not in _AGGREGATION_TARGET_BUILDERS:
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
    return (
        evaluation.candidate_global,
        evaluation.target_used,
        metadata,
    )


__all__ = [
    "AggregationTargetBuilder",
    "AggregationTargetConfig",
    "AggregationTargetContext",
    "AggregationTargetEvaluation",
    "AggregationTargetResult",
    "add_weighted_delta",
    "aggregate_target",
    "aggregation_target_names",
    "canonical_aggregation_target",
    "evaluate_aggregation_target",
    "graph_filter_client_arrays",
    "mix_client_arrays",
    "register_aggregation_target",
    "unregister_aggregation_target",
]
