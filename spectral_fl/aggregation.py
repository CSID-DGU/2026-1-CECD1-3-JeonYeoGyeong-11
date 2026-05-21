"""Backward-compatible facade for graph-FL aggregation helpers.

New code should import from ``graphfl_lab.strategies.graphfl.aggregation``.
This module remains so older scripts and tests keep their import path.
"""

from graphfl_lab.strategies.graphfl.aggregation import (
    apply_min_client_weight,
    compute_conflict_weights,
    compute_effective_clients,
    compute_entropy,
    compute_tau,
    weighted_average_by_alpha,
)

__all__ = [
    "apply_min_client_weight",
    "compute_conflict_weights",
    "compute_effective_clients",
    "compute_entropy",
    "compute_tau",
    "weighted_average_by_alpha",
]
