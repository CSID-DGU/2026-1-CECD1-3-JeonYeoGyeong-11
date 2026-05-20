"""Compatibility package root for GraphFL Lab."""

from __future__ import annotations

import os
import warnings


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
