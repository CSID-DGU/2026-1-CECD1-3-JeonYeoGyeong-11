"""Backward-compatible facade for server strategies.

The spectral strategy implementation now lives under
``spectral_fl.strategies.spectral``.  This module intentionally stays thin so
older scripts can continue importing ``spectral_fl.strategy`` while new work
edits the smaller strategy/baseline modules directly.
"""

from spectral_fl.strategies.baselines import (
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
from spectral_fl.strategies.spectral import (
    AggregationTargetConfig,
    SpectralState,
    aggregate_target,
)
from spectral_fl.strategies.spectral.strategy import SpectralConflictAwareStrategy

__all__ = [
    "AggregationTargetConfig",
    "SpectralConflictAwareStrategy",
    "SpectralState",
    "TracingFedAdagrad",
    "TracingFedAdam",
    "TracingFedAvg",
    "TracingFedAvgM",
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
