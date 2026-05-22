"""Deprecated compatibility shim for the old ``spectral_fl`` import root."""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any


if os.environ.get("GRAPHFL_LAB_SILENCE_DEPRECATION", "").lower() not in {
    "1",
    "true",
    "yes",
}:
    warnings.warn(
        "`spectral_fl` is deprecated and will be removed after the GraphFL Lab "
        "cleanup gates. Use `graphfl_lab` for new imports.",
        DeprecationWarning,
        stacklevel=2,
    )

_canonical_pkg = importlib.import_module("graphfl_lab")

# Resolve legacy submodule imports from the real package tree. This keeps
# ``spectral_fl.foo`` and old pickle module paths alive until Gate 6.
__path__ = list(getattr(_canonical_pkg, "__path__", []))
__all__ = list(getattr(_canonical_pkg, "__all__", []))


def __getattr__(name: str) -> Any:
    return getattr(_canonical_pkg, name)
