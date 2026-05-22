"""Client graph construction utilities.

The graph package separates three questions that were previously mixed in
``graphfl_lab.update_graph``:

* similarity: how two client signals are scored,
* sparsification: how dense scores become edges,
* builders: named graph construction modes used by experiments.
"""

from graphfl_lab.graph.builders import (
    build_client_graph,
    build_relation_graph,
    global_alignment_graph,
    learned_smooth_graph,
    magnitude_aware_graph,
    pfedgraph_qp_graph,
    rbf_graph,
)
from graphfl_lab.graph.controls import (
    build_control_graph,
    build_identity_graph,
    build_random_matched_graph,
    build_shuffled_graph,
    build_uniform_control_graph,
)
from graphfl_lab.graph.clustering import build_block_uniform_graph, cluster_clients
from graphfl_lab.graph.diagnostics import compute_graph_diagnostics
from graphfl_lab.graph.presets import (
    apply_graph_preset_to_namespace,
    graph_method_names,
    graph_preset_names,
    resolve_graph_method_spec,
    resolve_graph_preset_spec,
)
from graphfl_lab.graph.registry import (
    GraphBuildContext,
    GraphBuildResult,
    build_registered_graph,
    graph_mode_names,
    load_graph_plugins,
    register_graph_builder,
    require_graph_context,
    unregister_graph_builder,
)
from graphfl_lab.graph.similarity import (
    cosine_nonnegative,
    dense_absolute_cosine,
    dense_negative_cosine,
    dense_positive_cosine,
    dense_signed_cosine,
)
from graphfl_lab.graph.sparsification import (
    keep_mutual_topk,
    keep_threshold,
    keep_topk,
    random_edges_matched_to_knn,
    uniform_graph,
)
from graphfl_lab.graph.sources import (
    GraphSourceConfig,
    GraphSourceContext,
    GraphSourceResult,
    graph_source_names,
    graph_vectors_for_fedsim,
    graph_vectors_for_graphfl,
    normalize_key,
    register_graph_source,
    unregister_graph_source,
)

__all__ = [
    "GraphSourceConfig",
    "GraphSourceContext",
    "GraphSourceResult",
    "GraphBuildContext",
    "GraphBuildResult",
    "build_client_graph",
    "build_relation_graph",
    "build_registered_graph",
    "build_control_graph",
    "build_identity_graph",
    "build_random_matched_graph",
    "build_shuffled_graph",
    "build_uniform_control_graph",
    "build_block_uniform_graph",
    "cluster_clients",
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
    "pfedgraph_qp_graph",
    "graph_vectors_for_fedsim",
    "graph_vectors_for_graphfl",
    "apply_graph_preset_to_namespace",
    "graph_method_names",
    "graph_preset_names",
    "graph_mode_names",
    "graph_source_names",
    "load_graph_plugins",
    "normalize_key",
    "random_edges_matched_to_knn",
    "register_graph_builder",
    "register_graph_source",
    "require_graph_context",
    "resolve_graph_method_spec",
    "resolve_graph_preset_spec",
    "rbf_graph",
    "unregister_graph_builder",
    "unregister_graph_source",
    "uniform_graph",
]
