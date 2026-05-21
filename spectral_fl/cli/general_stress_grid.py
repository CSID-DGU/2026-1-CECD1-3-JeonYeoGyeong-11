"""Compatibility wrapper for ``graphfl_lab.cli.vision_stress_grid``."""

from graphfl_lab.cli import vision_stress_grid as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
main = _impl.main
