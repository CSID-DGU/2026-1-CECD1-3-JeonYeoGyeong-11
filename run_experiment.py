"""Compatibility wrapper for the Cora experiment CLI."""

from graphfl_lab.cli import cora_experiment as _impl

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
