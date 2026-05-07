"""Backward-compatible facade for spectral diagnostics and filtering.

New code should import from ``spectral_fl.strategies.spectral.filtering`` or
``spectral_fl.strategies.spectral.diagnostics``.  This module keeps the older
analysis and test import path stable.
"""

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

__all__ = [
    "apply_spectral_filter_with_diagnostics",
    "heterogeneity",
    "laplacian",
    "normalized_conflicts",
    "spectral_energy_diagnostics",
    "spectral_filter",
]
