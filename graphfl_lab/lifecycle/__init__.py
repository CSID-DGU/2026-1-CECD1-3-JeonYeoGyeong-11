"""Lifecycle contracts and trace utilities for graph-FL designs."""

from .context import (
    AggregationContext,
    RelationContext,
    RoundContext,
    StateExtractionContext,
    TopologyContext,
    make_round_context,
    make_state_extraction_context,
)
from .aggregation import (
    AggregationResult,
    GraphAggregationOperator,
    aggregation_trace_record,
)
from .counterfactuals import (
    CounterfactualResult,
    CounterfactualSpec,
    default_counterfactual_specs,
)
from .client_state import (
    ClientStateOutput,
    ClientStatePayload,
    GraphSourceClientStateExtractor,
    UnsupportedClientStateExtractor,
)
from .diagnostic_runner import (
    CounterfactualDiagnosticRunner,
    MinimalAggregationAdapter,
    MinimalAggregationOutput,
)
from .delivery import (
    DeliveryContext,
    GlobalDeliveryPolicy,
    InterfaceTargetDeliveryPolicy,
    MissingPersonalizedStateError,
    PreviousPersonalizedDeliveryPolicy,
)
from .local_hooks import (
    InterfaceTargetLocalObjectiveHook,
    LocalHookContext,
    NoneLocalObjectiveHook,
    ProximalToDeliveredModelHook,
)
from .modules import (
    AggregationOperator,
    ClientStateExtractor,
    DeliveryPolicy,
    LocalObjectiveHook,
    MODULE_STATUSES,
    ModuleResult,
    RelationEstimator,
    SUPPORT_LEVELS,
    TopologyOperator,
)
from .relation import (
    GraphRelationEstimator,
    RelationOutput,
    UnsupportedRelationEstimator,
    estimate_relation_from_vectors,
    relation_kind_for_graph_mode,
)
from .state_store import StateStore, state_store_from_mapping
from .topology import (
    ClusterBlockTopologyOperator,
    GraphTopologyOperator,
    TopologyOutput,
    UnsupportedTopologyOperator,
    build_topology_from_relation,
)
from .traces import TRACE_SCHEMA_VERSION, RoundTraceBundle, TraceRecord, json_safe

__all__ = [
    "AggregationContext",
    "AggregationOperator",
    "AggregationResult",
    "ClientStateOutput",
    "ClientStateExtractor",
    "ClientStatePayload",
    "ClusterBlockTopologyOperator",
    "CounterfactualDiagnosticRunner",
    "CounterfactualResult",
    "CounterfactualSpec",
    "DeliveryContext",
    "DeliveryPolicy",
    "GlobalDeliveryPolicy",
    "GraphAggregationOperator",
    "GraphRelationEstimator",
    "GraphSourceClientStateExtractor",
    "GraphTopologyOperator",
    "InterfaceTargetDeliveryPolicy",
    "InterfaceTargetLocalObjectiveHook",
    "LocalHookContext",
    "LocalObjectiveHook",
    "MODULE_STATUSES",
    "MinimalAggregationAdapter",
    "MinimalAggregationOutput",
    "MissingPersonalizedStateError",
    "ModuleResult",
    "NoneLocalObjectiveHook",
    "PreviousPersonalizedDeliveryPolicy",
    "ProximalToDeliveredModelHook",
    "RelationContext",
    "RelationOutput",
    "RoundTraceBundle",
    "RoundContext",
    "SUPPORT_LEVELS",
    "RelationEstimator",
    "StateExtractionContext",
    "StateStore",
    "TRACE_SCHEMA_VERSION",
    "TraceRecord",
    "TopologyContext",
    "TopologyOutput",
    "TopologyOperator",
    "UnsupportedClientStateExtractor",
    "UnsupportedRelationEstimator",
    "UnsupportedTopologyOperator",
    "aggregation_trace_record",
    "build_topology_from_relation",
    "default_counterfactual_specs",
    "estimate_relation_from_vectors",
    "json_safe",
    "make_round_context",
    "make_state_extraction_context",
    "relation_kind_for_graph_mode",
    "state_store_from_mapping",
]
