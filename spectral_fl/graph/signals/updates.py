"""Update and weight tensor-block signals for graph construction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from flwr.common import NDArrays


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec_f = vec.astype(np.float32, copy=False)
    return vec_f / (float(np.linalg.norm(vec_f)) + 1e-12)


def flatten_layerwise(arrays: NDArrays, normalize_layers: bool = True) -> np.ndarray:
    """Flatten arrays while giving each tensor block comparable graph weight."""
    parts: List[np.ndarray] = []
    for arr in arrays:
        flat = arr.reshape(-1).astype(np.float32, copy=False)
        if normalize_layers:
            flat = normalize_vector(flat)
        parts.append(flat)
    if not parts:
        return np.array([], dtype=np.float32)
    merged = np.concatenate(parts, axis=0)
    if normalize_layers:
        merged = merged / float(np.sqrt(len(parts)))
    return merged.astype(np.float32, copy=False)


def select_graph_layers(
    arrays: NDArrays, layer_start: int, layer_end: int
) -> Tuple[NDArrays, int, int]:
    """Select tensor blocks used by layer-slice graph sources.

    ``layer_start`` follows Python indexing. Negative values count from the
    end, so -2 usually selects the last weight/bias tensor pair. ``layer_end=0``
    means "to the end" for convenient CLI defaults.
    """
    n = len(arrays)
    if n <= 0:
        raise ValueError("Cannot build a graph from an empty parameter list")

    start_raw = int(layer_start)
    start = n + start_raw if start_raw < 0 else start_raw
    start = max(0, min(start, n))

    end_raw = int(layer_end)
    if end_raw == 0:
        end = n
    else:
        end = n + end_raw if end_raw < 0 else end_raw
        end = max(0, min(end, n))

    if end <= start:
        raise ValueError(
            "Invalid graph layer slice: "
            f"start={layer_start}, end={layer_end}, "
            f"resolved={start}:{end}, n_tensors={n}"
        )
    return list(arrays[start:end]), int(start), int(end)


__all__ = ["flatten_layerwise", "normalize_vector", "select_graph_layers"]
