"""Backward-compatible facade for client graph construction.

New code should prefer the clearer ``graphfl_lab.graph`` package.  This module
keeps the previous import path stable for scripts, tests, and result tooling.
"""

from graphfl_lab.graph import (
    build_client_graph,
    compute_graph_diagnostics,
    cosine_nonnegative,
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    dense_signed_cosine,
)
from graphfl_lab.graph.builders import (
    global_alignment_graph as _global_alignment_graph,
    learned_smooth_graph as _learned_smooth_graph,
    magnitude_aware_graph as _magnitude_aware_graph,
    project_simplex as _project_simplex,
    rbf_graph as _rbf_graph,
)
from graphfl_lab.graph.similarity import (
    pairwise_sq_dists as _pairwise_sq_dists,
    positive_upper_values as _positive_upper_values,
    resolve_distance_sigma as _resolve_distance_sigma,
)
from graphfl_lab.graph.sparsification import (
    directed_topk_mask as _directed_topk_mask,
    keep_mutual_topk as _mutual_knn_keep_topk,
    keep_threshold as _threshold_keep,
    keep_topk as _knn_keep_topk,
    random_edges_matched_to_knn as _random_edges_matched_to_knn,
    random_edges_with_edge_count as _random_edges_with_edge_count,
    uniform_graph as _uniform_graph,
)

__all__ = [
    "build_client_graph",
    "compute_graph_diagnostics",
    "cosine_nonnegative",
    "dense_absolute_cosine",
    "dense_negative_cosine",
    "dense_positive_cosine",
    "dense_signed_cosine",
]
