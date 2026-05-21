"""Classifier-head signal selection for graph construction."""

from __future__ import annotations

from typing import Tuple

from flwr.common import NDArrays


def select_classifier_head(arrays: NDArrays) -> Tuple[NDArrays, int, int]:
    """Select the final weight/bias tensor pair as the classifier head."""
    n = len(arrays)
    if n <= 0:
        raise ValueError("Cannot build a graph from an empty parameter list")
    start = max(0, n - 2)
    return list(arrays[start:n]), int(start), int(n)


__all__ = ["select_classifier_head"]
