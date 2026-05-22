"""Compatibility wrapper for the vision FL suite CLI.

Prefer ``run_vision_suite.py`` for new scripts and docs. This entrypoint remains
until Gate 6 removes legacy runner names.
"""

from graphfl_lab.cli import vision_suite as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
main = _impl.main


if __name__ == "__main__":
    main()
