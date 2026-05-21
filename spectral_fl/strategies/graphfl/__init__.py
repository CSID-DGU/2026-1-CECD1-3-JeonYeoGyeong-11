"""Composable graph-FL strategy internals."""

from graphfl_lab.strategies.graphfl.aggregation import (
    apply_min_client_weight,
    compute_conflict_weights,
    compute_effective_clients,
    compute_entropy,
    compute_tau,
    weighted_average_by_alpha,
)
from graphfl_lab.strategies.graphfl.config import (
    GraphFLStrategyState,
    SpectralState,
)
from graphfl_lab.strategies.graphfl.diagnostics import (
    heterogeneity,
    spectral_energy_diagnostics,
)
from graphfl_lab.strategies.graphfl.filtering import (
    apply_spectral_filter_with_diagnostics,
    laplacian,
    normalized_conflicts,
    spectral_filter,
)
from graphfl_lab.strategies.graphfl.momentum import apply_server_optimizer
from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    aggregate_target,
)


def __getattr__(name: str):
    if name == "GraphFLDiagnosticStrategy":
        from graphfl_lab.strategies.graphfl.strategy import (
            GraphFLDiagnosticStrategy,
        )

        return GraphFLDiagnosticStrategy
    if name == "SpectralConflictAwareStrategy":
        from graphfl_lab.strategies.graphfl.strategy import (
            SpectralConflictAwareStrategy,
        )

        return SpectralConflictAwareStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AggregationTargetConfig",
    "GraphFLDiagnosticStrategy",
    "GraphFLStrategyState",
    "SpectralConflictAwareStrategy",
    "SpectralState",
    "aggregate_target",
    "apply_min_client_weight",
    "apply_server_optimizer",
    "apply_spectral_filter_with_diagnostics",
    "compute_conflict_weights",
    "compute_effective_clients",
    "compute_entropy",
    "compute_tau",
    "heterogeneity",
    "laplacian",
    "normalized_conflicts",
    "spectral_energy_diagnostics",
    "spectral_filter",
    "weighted_average_by_alpha",
]
