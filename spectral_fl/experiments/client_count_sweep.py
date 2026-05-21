"""Backward-compatible facade for vision FL client-count sweeps."""

from graphfl_lab.experiments.vision import client_count_sweep as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
