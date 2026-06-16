"""Projected graph-source vector preparation for GraphFL rounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.graph.sources import (
    GraphSourceConfig,
    graph_vectors_for_graphfl,
    resolve_graph_source_result,
)


@dataclass(frozen=True)
class ProjectedGraphSpace:
    graph_vectors: List[np.ndarray]
    graph_source_used: str
    graph_source_norms: np.ndarray
    z_mat: np.ndarray
    z_norms: np.ndarray
    graph_source_metadata: Mapping[str, Any]


def select_graph_source_vectors(
    *,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: Optional[List[NDArrays]],
    graph_source: str,
    graph_layer_start: int,
    graph_layer_end: int,
) -> Tuple[List[np.ndarray], str]:
    return graph_vectors_for_graphfl(
        local_weights=local_weights,
        local_updates=local_updates,
        ema_updates=ema_updates,
        config=GraphSourceConfig(
            source=graph_source,
            layer_start=graph_layer_start,
            layer_end=graph_layer_end,
        ),
    )


def build_projected_graph_space(
    *,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: Optional[List[NDArrays]],
    graph_source: str,
    graph_layer_start: int,
    graph_layer_end: int,
    project_fn: Callable[[np.ndarray], np.ndarray],
) -> ProjectedGraphSpace:
    source_result = resolve_graph_source_result(
        local_weights=local_weights,
        local_updates=local_updates,
        ema_updates=ema_updates,
        config=GraphSourceConfig(
            source=graph_source,
            layer_start=graph_layer_start,
            layer_end=graph_layer_end,
        ),
    )
    graph_vectors = source_result.vectors
    graph_source_used = source_result.source_used
    graph_source_norms = np.array(
        [float(np.linalg.norm(vector)) for vector in graph_vectors]
    )
    z_list = [project_fn(vector) for vector in graph_vectors]
    z_mat = np.stack(z_list, axis=0)
    z_norms = np.array([float(np.linalg.norm(z)) for z in z_list])
    return ProjectedGraphSpace(
        graph_vectors=graph_vectors,
        graph_source_used=graph_source_used,
        graph_source_norms=graph_source_norms,
        z_mat=z_mat,
        z_norms=z_norms,
        graph_source_metadata=dict(source_result.metadata or {}),
    )


__all__ = [
    "ProjectedGraphSpace",
    "build_projected_graph_space",
    "select_graph_source_vectors",
]
