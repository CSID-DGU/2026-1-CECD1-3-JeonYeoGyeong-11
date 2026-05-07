"""Graph input-signal source selection."""

from spectral_fl.graph.sources.config import GraphSourceConfig, normalize_key
from spectral_fl.graph.sources.fedsim import graph_vectors_for_fedsim
from spectral_fl.graph.sources.selection import (
    flatten_layerwise,
    normalize_vector,
    select_classifier_head,
    select_graph_layers,
)
from spectral_fl.graph.sources.spectral import graph_vectors_for_spectral

__all__ = [
    "GraphSourceConfig",
    "flatten_layerwise",
    "graph_vectors_for_fedsim",
    "graph_vectors_for_spectral",
    "normalize_key",
    "normalize_vector",
    "select_classifier_head",
    "select_graph_layers",
]
