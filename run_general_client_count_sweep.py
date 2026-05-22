"""Compatibility wrapper for the vision FL client-count sweep CLI.

Prefer ``run_vision_client_count_sweep.py`` for new scripts and docs.
"""

from graphfl_lab.cli import vision_client_count_sweep as _impl

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
