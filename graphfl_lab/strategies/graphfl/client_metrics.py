"""Client metric helpers for GraphFL strategy aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def extract_metric(
    client_metrics: List[Dict[str, Any]],
    *keys: str,
) -> Optional[List[Optional[float]]]:
    """Look up a metric by any of the given key aliases per client."""
    out: List[Optional[float]] = []
    any_found = False
    for metrics in client_metrics:
        value: Optional[float] = None
        for key in keys:
            if key not in metrics:
                continue
            try:
                value = float(metrics[key])
                any_found = True
                break
            except (TypeError, ValueError):
                continue
        out.append(value)
    return out if any_found else None


def weighted_optional_mean(
    values: Optional[List[Optional[float]]],
    weights: np.ndarray,
) -> float:
    if values is None:
        return float("nan")
    num = 0.0
    den = 0.0
    for value, weight in zip(values, weights):
        if value is None:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    if den <= 0.0:
        return float("nan")
    return float(num / den)


__all__ = ["extract_metric", "weighted_optional_mean"]
