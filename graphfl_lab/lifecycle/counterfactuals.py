"""Counterfactual diagnostic specifications and result envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .traces import TraceRecord


@dataclass(frozen=True)
class CounterfactualSpec:
    name: str
    correction_family: str = "real_graph"
    graph_source: str = "actual"
    relation_mode: str = "actual"
    topology_mode: str = "actual"
    aggregation_target: str = "spectral_filtered_update"
    graph_free_mode: str = "none"
    cluster_method: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip().lower().replace("-", "_"))
        object.__setattr__(self, "correction_family", str(self.correction_family).strip().lower().replace("-", "_"))
        object.__setattr__(self, "topology_mode", str(self.topology_mode).strip().lower().replace("-", "_"))
        object.__setattr__(self, "aggregation_target", str(self.aggregation_target).strip().lower().replace("-", "_"))
        object.__setattr__(self, "graph_free_mode", str(self.graph_free_mode).strip().lower().replace("-", "_"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.name:
            raise ValueError("counterfactual name cannot be empty")


@dataclass(frozen=True)
class CounterfactualResult:
    name: str
    adjacency: np.ndarray
    weights_post: np.ndarray
    post_flat_updates: np.ndarray
    metrics: Mapping[str, Any]
    trace_records: Sequence[TraceRecord] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "adjacency", np.asarray(self.adjacency, dtype=np.float64))
        object.__setattr__(self, "weights_post", np.asarray(self.weights_post, dtype=np.float64))
        object.__setattr__(self, "post_flat_updates", np.asarray(self.post_flat_updates, dtype=np.float64))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "trace_records", tuple(self.trace_records))


def default_counterfactual_specs() -> tuple[CounterfactualSpec, ...]:
    return (
        CounterfactualSpec(name="actual", topology_mode="actual"),
        CounterfactualSpec(name="matched_random", correction_family="control_graph", topology_mode="matched_random"),
        CounterfactualSpec(name="shuffled", correction_family="control_graph", topology_mode="shuffled"),
        CounterfactualSpec(name="uniform", correction_family="control_graph", topology_mode="uniform"),
        CounterfactualSpec(name="identity", correction_family="control_graph", topology_mode="identity"),
        CounterfactualSpec(name="clustering_only", correction_family="clustering_only", topology_mode="cluster_block", cluster_method="deterministic_split"),
        CounterfactualSpec(
            name="graphfree_dominance_reweight",
            correction_family="graph_free",
            topology_mode="identity",
            aggregation_target="graphfree_dominance_reweight",
            graph_free_mode="dominance_reweight",
        ),
    )


__all__ = [
    "CounterfactualResult",
    "CounterfactualSpec",
    "default_counterfactual_specs",
]
