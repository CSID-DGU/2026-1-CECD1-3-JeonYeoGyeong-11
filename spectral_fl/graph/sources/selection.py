"""Backward-compatible facade for graph signal selection helpers.

New code should import signal extraction from ``spectral_fl.graph.signals``.
This module remains for existing graph source implementations and external
imports that still use ``spectral_fl.graph.sources.selection``.
"""

from spectral_fl.graph.signals import (
    flatten_layerwise,
    normalize_vector,
    select_classifier_head,
    select_graph_layers,
)

__all__ = [
    "flatten_layerwise",
    "normalize_vector",
    "select_classifier_head",
    "select_graph_layers",
]
