"""Baseline strategy helpers."""

from spectral_fl.strategies.baselines.fednova import TracingFedNova
from spectral_fl.strategies.baselines.fedsim import TracingFedSim
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
    "TracingFedSim",
    "TracingFedTrimmedAvg",
    "TracingFedYogi",
    "_EvalTracer",
    "_fit_result_cid_key",
    "sort_fit_results_by_cid",
]
