"""Canonical launcher for vision FL client-count sweeps."""

from spectral_fl.cli import vision_client_count_sweep as _impl

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
