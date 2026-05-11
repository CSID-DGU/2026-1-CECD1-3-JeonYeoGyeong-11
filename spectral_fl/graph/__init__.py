"""Client graph construction utilities.

The graph package separates three questions that were previously mixed in
``spectral_fl.update_graph``:

* similarity: how two client signals are scored,
* sparsification: how dense scores become edges,
* builders: named graph construction modes used by experiments.
"""

from spectral_fl.graph.builders import (
    build_client_graph,
    global_alignment_graph,
    learned_smooth_graph,
    magnitude_aware_graph,
    rbf_graph,
)
from spectral_fl.graph.diagnostics import compute_graph_diagnostics
from spectral_fl.graph.presets import (
    apply_graph_preset_to_namespace,
    graph_preset_names,
    resolve_graph_preset_spec,
)
from spectral_fl.graph.similarity import (
    cosine_nonnegative,
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    dense_signed_cosine,
)
from spectral_fl.graph.sparsification import (
    keep_mutual_topk,
    keep_threshold,
    keep_topk,
    random_edges_matched_to_knn,
    uniform_graph,
)
from spectral_fl.graph.sources import (
    GraphSourceConfig,
    graph_vectors_for_fedsim,
    graph_vectors_for_spectral,
    normalize_key,
)

__all__ = [
    "GraphSourceConfig",
    "build_client_graph",
    "compute_graph_diagnostics",
    "cosine_nonnegative",
    "dense_absolute_cosine",
    "dense_negative_cosine",
    "dense_positive_cosine",
    "dense_signed_cosine",
    "global_alignment_graph",
    "keep_mutual_topk",
    "keep_threshold",
    "keep_topk",
    "learned_smooth_graph",
    "magnitude_aware_graph",
    "graph_vectors_for_fedsim",
    "graph_vectors_for_spectral",
    "apply_graph_preset_to_namespace",
    "graph_preset_names",
    "normalize_key",
    "random_edges_matched_to_knn",
    "resolve_graph_preset_spec",
    "rbf_graph",
    "uniform_graph",
]
