"""Baseline strategy helpers."""

from spectral_fl.strategies.baselines.fednova import TracingFedNova
from spectral_fl.strategies.baselines.fedsim import TracingFedSim
from spectral_fl.strategies.baselines.graph_smooth import TracingGraphSmoothFedAvgM
from spectral_fl.strategies.baselines.dominance_aware import TracingDominanceAwareFedAvgM
from spectral_fl.strategies.baselines.ordering import (
    _fit_result_cid_key,
    sort_fit_results_by_cid,
)
from spectral_fl.strategies.baselines.tracing import (
    TracingFedAdagrad,
    TracingFedAdam,
    TracingFedAvg,
    TracingFedAvgM,
    TracingFedMedian,
    TracingFedProx,
    TracingFedTrimmedAvg,
    TracingFedYogi,
    _EvalTracer,
)

__all__ = [
    "TracingFedAdagrad",
    "TracingFedAdam",
    "TracingFedAvg",
    "TracingFedAvgM",
    "TracingFedMedian",
    "TracingFedNova",
    "TracingFedProx",
    "TracingGraphSmoothFedAvgM",
    "TracingDominanceAwareFedAvgM",
    "TracingFedSim",
    "TracingFedTrimmedAvg",
    "TracingFedYogi",
    "_EvalTracer",
    "_fit_result_cid_key",
    "sort_fit_results_by_cid",
]
