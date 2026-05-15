"""Graph input-signal source selection."""

from spectral_fl.graph.sources.config import GraphSourceConfig, normalize_key
from spectral_fl.graph.sources.fedsim import graph_vectors_for_fedsim
from spectral_fl.graph.sources.registry import (
    GraphSourceContext,
    GraphSourceResult,
    build_registered_graph_source,
    graph_source_names,
    register_graph_source,
    unregister_graph_source,
)
from spectral_fl.graph.sources.selection import (
    flatten_layerwise,
    normalize_vector,
    select_classifier_head,
    select_graph_layers,
)
from spectral_fl.graph.sources.spectral import graph_vectors_for_spectral

__all__ = [
    "GraphSourceConfig",
    "GraphSourceContext",
    "GraphSourceResult",
    "build_registered_graph_source",
    "flatten_layerwise",
    "graph_source_names",
    "graph_vectors_for_fedsim",
    "graph_vectors_for_spectral",
    "normalize_key",
    "normalize_vector",
    "register_graph_source",
    "select_classifier_head",
    "select_graph_layers",
    "unregister_graph_source",
]
