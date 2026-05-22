"""Backward-compatible facade for vision FL suite orchestration."""

from graphfl_lab.experiments.vision import suite as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
