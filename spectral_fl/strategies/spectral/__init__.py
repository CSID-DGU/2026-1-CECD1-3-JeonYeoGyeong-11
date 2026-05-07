"""Spectral conflict-aware strategy internals."""

from spectral_fl.strategies.spectral.aggregation import (
    apply_min_client_weight,
    compute_conflict_weights,
    compute_effective_clients,
    compute_entropy,
    compute_tau,
    weighted_average_by_alpha,
)
from spectral_fl.strategies.spectral.config import SpectralState
from spectral_fl.strategies.spectral.diagnostics import (
    heterogeneity,
    spectral_energy_diagnostics,
)
from spectral_fl.strategies.spectral.filtering import (
    apply_spectral_filter_with_diagnostics,
    laplacian,
    normalized_conflicts,
    spectral_filter,
)
from spectral_fl.strategies.spectral.momentum import apply_server_optimizer
from spectral_fl.strategies.spectral.targets import (
    AggregationTargetConfig,
    aggregate_target,
)


def __getattr__(name: str):
    if name == "SpectralConflictAwareStrategy":
        from spectral_fl.strategies.spectral.strategy import (
            SpectralConflictAwareStrategy,
        )

        return SpectralConflictAwareStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AggregationTargetConfig",
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
