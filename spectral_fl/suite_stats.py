"""Backward-compatible facade for suite statistics helpers."""

from spectral_fl.experiments.suites.stats import (
    final_acc,
    load_json,
    round_trace_field,
    safe_max,
    safe_mean,
    safe_min,
    safe_pstdev,
)

__all__ = [
    "final_acc",
    "load_json",
    "round_trace_field",
    "safe_max",
    "safe_mean",
    "safe_min",
    "safe_pstdev",
]
