"""Backward-compatible facade for server strategies.

The graph-FL diagnostic strategy implementation now lives under
``graphfl_lab.strategies.graphfl``.  This module intentionally stays thin so
older scripts can continue importing ``graphfl_lab.strategy`` while new work
edits the smaller strategy/baseline modules directly.
"""

from graphfl_lab.strategies.baselines import (
    TracingDominanceAwareFedAvgM,
    TracingFedAdagrad,
    TracingFedAdam,
    TracingFedAvg,
    TracingFedAvgM,
    TracingFedMedian,
    TracingFedNova,
    TracingFedProx,
    TracingFedSim,
    TracingFedTrimmedAvg,
    TracingFedYogi,
    _EvalTracer,
    _fit_result_cid_key,
    sort_fit_results_by_cid as _sort_fit_results_by_cid,
)
from graphfl_lab.strategies.graphfl import (
    AggregationTargetConfig,
    GraphFLStrategyState,
    SpectralState,
    aggregate_target,
)
from graphfl_lab.strategies.graphfl.strategy import (
    GraphFLDiagnosticStrategy,
    SpectralConflictAwareStrategy,
)

__all__ = [
    "AggregationTargetConfig",
    "GraphFLDiagnosticStrategy",
    "GraphFLStrategyState",
    "SpectralConflictAwareStrategy",
    "SpectralState",
    "TracingFedAdagrad",
    "TracingFedAdam",
    "TracingFedAvg",
    "TracingFedAvgM",
    "TracingDominanceAwareFedAvgM",
    "TracingFedMedian",
    "TracingFedNova",
    "TracingFedProx",
    "TracingFedSim",
    "TracingFedTrimmedAvg",
    "TracingFedYogi",
    "_EvalTracer",
    "_fit_result_cid_key",
    "_sort_fit_results_by_cid",
    "aggregate_target",
]
