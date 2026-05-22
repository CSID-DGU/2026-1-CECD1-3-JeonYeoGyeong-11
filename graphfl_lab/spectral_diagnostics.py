"""Backward-compatible facade for spectral diagnostics and filtering.

New code should import from ``graphfl_lab.strategies.graphfl.filtering`` or
``graphfl_lab.strategies.graphfl.diagnostics``.  This module keeps the older
analysis and test import path stable.
"""

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

__all__ = [
    "apply_spectral_filter_with_diagnostics",
    "heterogeneity",
    "laplacian",
    "normalized_conflicts",
    "spectral_energy_diagnostics",
    "spectral_filter",
]
