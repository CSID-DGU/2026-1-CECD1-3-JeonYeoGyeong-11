"""Registry for pluggable graph-source extraction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources.config import GraphSourceConfig, normalize_key


@dataclass(frozen=True)
class GraphSourceContext:
    """Inputs exposed to graph-source plugins."""

    local_weights: List[NDArrays]
    local_updates: List[NDArrays]
    ema_updates: Optional[List[NDArrays]]
    config: GraphSourceConfig


@dataclass(frozen=True)
class GraphSourceResult:
    vectors: List[np.ndarray]
    source_used: str
    metadata: Mapping[str, object] | None = None


GraphSourceBuilder = Callable[[GraphSourceContext], GraphSourceResult | tuple[List[np.ndarray], str]]

_GRAPH_SOURCE_BUILDERS: dict[str, GraphSourceBuilder] = {}


def register_graph_source(
    *names: str,
    override: bool = False,
) -> Callable[[GraphSourceBuilder], GraphSourceBuilder]:
    """Register a source builder under one or more ``--graph-source`` names."""
    if not names:
        raise ValueError("register_graph_source requires at least one source name")

    def decorator(func: GraphSourceBuilder) -> GraphSourceBuilder:
        for name in names:
            key = normalize_key(name)
            if not key:
                raise ValueError("graph source name cannot be empty")
            if key in _GRAPH_SOURCE_BUILDERS and not bool(override):
                raise ValueError(
                    f"Graph source {name!r} is already registered; "
                    "pass override=True to replace it"
                )
            _GRAPH_SOURCE_BUILDERS[key] = func
        return func

    return decorator


def unregister_graph_source(name: str) -> None:
    """Remove a registered source. Intended mainly for tests."""
    _GRAPH_SOURCE_BUILDERS.pop(normalize_key(name), None)


def graph_source_names() -> list[str]:
    return sorted(_GRAPH_SOURCE_BUILDERS)


def build_registered_graph_source(
    context: GraphSourceContext,
) -> GraphSourceResult | None:
    builder = _GRAPH_SOURCE_BUILDERS.get(normalize_key(context.config.source))
    if builder is None:
        return None
    result = builder(context)
    if isinstance(result, GraphSourceResult):
        return result
    vectors, source_used = result
    return GraphSourceResult(vectors=vectors, source_used=source_used)


__all__ = [
    "GraphSourceBuilder",
    "GraphSourceContext",
    "GraphSourceResult",
    "build_registered_graph_source",
    "graph_source_names",
    "register_graph_source",
    "unregister_graph_source",
]
