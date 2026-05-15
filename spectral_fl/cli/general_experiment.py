"""Compatibility wrapper for ``spectral_fl.cli.vision_experiment``."""

from spectral_fl.cli import vision_experiment as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
main = _impl.main
