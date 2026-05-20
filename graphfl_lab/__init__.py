"""Canonical package alias for GraphFL Lab.

Implementation still lives under ``spectral_fl`` during the staged migration.
This package exposes the canonical import root while Gate 3 import batches move
internals over incrementally.
"""

from __future__ import annotations

import importlib
import os
import sys


_old_silence = os.environ.get("GRAPHFL_LAB_SILENCE_DEPRECATION")
os.environ["GRAPHFL_LAB_SILENCE_DEPRECATION"] = "1"
try:
    _legacy_pkg = importlib.import_module("spectral_fl")
finally:
    if _old_silence is None:
        os.environ.pop("GRAPHFL_LAB_SILENCE_DEPRECATION", None)
    else:
        os.environ["GRAPHFL_LAB_SILENCE_DEPRECATION"] = _old_silence

# Let ``import graphfl_lab.some_module`` resolve modules from the current
# implementation tree until the real package move is complete.
__path__ = list(getattr(_legacy_pkg, "__path__", []))
__all__ = list(getattr(_legacy_pkg, "__all__", []))

# Keep the old root explicitly present for pickle/import compatibility during
# deprecation. ``spectral_fl`` remains the real module until Gate 6.
sys.modules.setdefault("spectral_fl", _legacy_pkg)
