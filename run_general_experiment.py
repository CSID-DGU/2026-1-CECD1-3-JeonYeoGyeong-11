"""Compatibility wrapper for the vision FL experiment CLI.

Prefer ``run_vision_experiment.py`` for new scripts and docs. This entrypoint
remains until Gate 6 removes legacy runner names.
"""

from graphfl_lab.cli import experiment_dispatcher as _dispatcher
from graphfl_lab.cli import vision_experiment as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
main = _dispatcher.vision_main


if __name__ == "__main__":
    main()
