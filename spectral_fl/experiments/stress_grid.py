"""Backward-compatible facade for General FL stress-grid orchestration."""

from spectral_fl.experiments.general import stress_grid as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
