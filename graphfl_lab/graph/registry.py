"""Registry for pluggable client graph construction algorithms."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


def normalize_graph_mode(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


@dataclass(frozen=True)
class GraphBuildContext:
    """Inputs exposed to graph builder plugins.

    Builders should use only this context and return either an adjacency matrix,
    ``(adjacency, metadata)`` or ``GraphBuildResult``.
    """

    z_mat: np.ndarray
    mode: str
    graph_source: str = "unknown"
    aggregation_target: str = "unknown"
    correction_family: str = "real_graph"
    knn_k: int = 2
    edge_threshold: float = 0.0
    rng: Optional[np.random.Generator] = None
    graph_scale_sigma: float = 1.0
    learned_graph_lambda: float = 1.0
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphBuildResult:
    adjacency: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


GraphBuilder = Callable[[GraphBuildContext], GraphBuildResult | tuple[np.ndarray, Mapping[str, Any]] | np.ndarray]

_GRAPH_BUILDERS: dict[str, GraphBuilder] = {}


def normalize_graph_adjacency(
    adjacency: np.ndarray,
    *,
    num_clients: int,
    mode: str,
) -> np.ndarray:
    """Validate and normalize a plugin adjacency for Laplacian aggregation."""
    adj = np.asarray(adjacency, dtype=np.float64)
    expected_shape = (int(num_clients), int(num_clients))
    if adj.shape != expected_shape:
        raise ValueError(
            f"Graph builder {mode!r} returned shape {adj.shape}; "
            f"expected {expected_shape}"
        )
    if not bool(np.all(np.isfinite(adj))):
        raise ValueError(f"Graph builder {mode!r} returned non-finite weights")
    adj = np.maximum(adj, 0.0)
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def register_graph_builder(
    *names: str,
    override: bool = False,
) -> Callable[[GraphBuilder], GraphBuilder]:
    """Register a graph builder under one or more ``--graph-mode`` names."""
    if not names:
        raise ValueError("register_graph_builder requires at least one mode name")

    def decorator(func: GraphBuilder) -> GraphBuilder:
        for name in names:
            key = normalize_graph_mode(name)
            if not key:
                raise ValueError("graph mode name cannot be empty")
            if key in _GRAPH_BUILDERS and not bool(override):
                raise ValueError(
                    f"Graph mode {name!r} is already registered; "
                    "pass override=True to replace it"
                )
            _GRAPH_BUILDERS[key] = func
        return func

    return decorator


def unregister_graph_builder(name: str) -> None:
    """Remove a registered builder. Intended mainly for tests."""
    _GRAPH_BUILDERS.pop(normalize_graph_mode(name), None)


def graph_mode_names() -> list[str]:
    return sorted(_GRAPH_BUILDERS)


def require_graph_context(
    context: GraphBuildContext,
    *,
    graph_sources: Sequence[str] = (),
    aggregation_targets: Sequence[str] = (),
) -> None:
    """Raise if a plugin is used with an unsupported source/target context."""
    if graph_sources:
        allowed_sources = {normalize_graph_mode(x) for x in graph_sources}
        if normalize_graph_mode(context.graph_source) not in allowed_sources:
            raise ValueError(
                f"Graph mode {context.mode!r} requires graph_source in "
                f"{sorted(allowed_sources)}, got {context.graph_source!r}"
            )
    if aggregation_targets:
        allowed_targets = {normalize_graph_mode(x) for x in aggregation_targets}
        if normalize_graph_mode(context.aggregation_target) not in allowed_targets:
            raise ValueError(
                f"Graph mode {context.mode!r} requires aggregation_target in "
                f"{sorted(allowed_targets)}, got {context.aggregation_target!r}"
            )


def _coerce_result(
    value: GraphBuildResult | tuple[np.ndarray, Mapping[str, Any]] | np.ndarray,
    *,
    context: GraphBuildContext,
    mode_key: str,
) -> GraphBuildResult:
    if isinstance(value, GraphBuildResult):
        adjacency = value.adjacency
        metadata = dict(value.metadata)
    elif isinstance(value, tuple):
        adjacency, metadata_raw = value
        metadata = dict(metadata_raw)
    else:
        adjacency = value
        metadata = {}

    adj = normalize_graph_adjacency(
        adjacency,
        num_clients=int(context.z_mat.shape[0]),
        mode=mode_key,
    )
    if "graph_kind" in metadata and "base_graph_kind" not in metadata:
        metadata["base_graph_kind"] = metadata.pop("graph_kind")
    metadata.setdefault("base_graph_builder", "registered")
    metadata.setdefault("base_graph_mode", mode_key)
    return GraphBuildResult(adjacency=adj, metadata=metadata)


def build_registered_graph(context: GraphBuildContext) -> GraphBuildResult | None:
    mode_key = normalize_graph_mode(context.mode)
    builder = _GRAPH_BUILDERS.get(mode_key)
    if builder is None:
        return None
    result = _coerce_result(builder(context), context=context, mode_key=mode_key)
    metadata = dict(result.metadata)
    metadata.setdefault("component_kind", "TopologyOperator")
    metadata.setdefault("component_name", mode_key)
    metadata.setdefault("plugin_module", builder.__module__)
    metadata.setdefault(
        "parameters",
        {
            "knn_k": int(context.knn_k),
            "edge_threshold": float(context.edge_threshold),
            "graph_scale_sigma": float(context.graph_scale_sigma),
            "learned_graph_lambda": float(context.learned_graph_lambda),
        },
    )
    metadata.setdefault("input_shape", list(context.z_mat.shape))
    metadata.setdefault("output_shape", list(result.adjacency.shape))
    metadata.setdefault("source_used", str(context.graph_source))
    metadata.setdefault("target_used", str(context.aggregation_target))
    trace_values = {
        key: value
        for key, value in metadata.items()
        if key != "lifecycle_trace"
    }
    metadata.setdefault(
        "lifecycle_trace",
        [
            {
                "schema_version": "lifecycle_trace_v1",
                "phase": "topology",
                "module": "TopologyOperator",
                "name": mode_key,
                "round": None,
                "values": trace_values,
            }
        ],
    )
    return GraphBuildResult(adjacency=result.adjacency, metadata=metadata)


def _split_plugin_spec(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        raw_items = spec.replace(";", ",").split(",")
    else:
        raw_items = []
        for item in spec:
            raw_items.extend(str(item).replace(";", ",").split(","))
    return [item.strip() for item in raw_items if item.strip()]


def load_graph_plugins(spec: str | Sequence[str] | None) -> list[str]:
    """Import graph plugin modules or filesystem packages."""
    loaded: list[str] = []
    for item in _split_plugin_spec(spec):
        path = Path(item).expanduser()
        if path.exists():
            resolved = path.resolve()
            if resolved.is_dir():
                init_path = resolved / "__init__.py"
                if not init_path.is_file():
                    raise ValueError(
                        f"Graph plugin directory must contain __init__.py: {resolved}"
                    )
                source_path = init_path
                submodule_locations = [str(resolved)]
            elif resolved.suffix.lower() == ".py":
                source_path = resolved
                submodule_locations = None
            else:
                raise ValueError(
                    f"Graph plugin path must be a Python file or package: {resolved}"
                )
            module_key = abs(hash(str(resolved)))
            module_name = f"graphfl_user_plugin_{module_key:x}"
            if module_name in sys.modules:
                loaded.append(str(resolved))
                continue
            module_spec = importlib.util.spec_from_file_location(
                module_name,
                source_path,
                submodule_search_locations=submodule_locations,
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Unable to load graph plugin: {resolved}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
            loaded.append(str(resolved))
            continue
        importlib.import_module(item)
        loaded.append(item)
    return loaded


__all__ = [
    "GraphBuildContext",
    "GraphBuildResult",
    "GraphBuilder",
    "build_registered_graph",
    "graph_mode_names",
    "load_graph_plugins",
    "normalize_graph_adjacency",
    "normalize_graph_mode",
    "register_graph_builder",
    "require_graph_context",
    "unregister_graph_builder",
]
