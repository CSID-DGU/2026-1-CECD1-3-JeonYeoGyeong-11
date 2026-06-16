"""Public authoring API for Graph-FL extensions."""

from graphfl_lab.designs import (
    ComponentSpec,
    GraphFLDesign,
    register_design,
)
from graphfl_lab.graph import (
    GraphBuildContext,
    GraphBuildResult,
    GraphSourceContext,
    GraphSourceResult,
    register_graph_builder,
    register_graph_source,
)
from graphfl_lab.projection import flatten_weights as flatten_client_arrays
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetContext,
    AggregationTargetResult,
    add_weighted_delta,
    graph_filter_client_arrays,
    mix_client_arrays,
    register_aggregation_target,
)

__all__ = [
    "AggregationTargetContext",
    "AggregationTargetResult",
    "ComponentSpec",
    "GraphBuildContext",
    "GraphBuildResult",
    "GraphFLDesign",
    "GraphSourceContext",
    "GraphSourceResult",
    "add_weighted_delta",
    "flatten_client_arrays",
    "graph_filter_client_arrays",
    "mix_client_arrays",
    "register_aggregation_target",
    "register_design",
    "register_graph_builder",
    "register_graph_source",
]
