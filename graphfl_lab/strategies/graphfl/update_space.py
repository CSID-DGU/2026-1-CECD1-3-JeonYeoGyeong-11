"""Update-space array preparation for GraphFL strategy rounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from flwr.common import NDArrays

from graphfl_lab.projection import flatten_weights


@dataclass(frozen=True)
class UpdateSpaceArrays:
    flat_deltas: list[np.ndarray]
    flat_delta_matrix: np.ndarray
    delta_norms: np.ndarray
    flat_ema_deltas: list[np.ndarray]
    ema_delta_norms: np.ndarray
    flat_weights: list[np.ndarray]
    weight_norms: np.ndarray


def compute_local_updates(
    *,
    local_weights: List[NDArrays],
    current_global: NDArrays,
) -> List[NDArrays]:
    return [
        [local_param - global_param for local_param, global_param in zip(local, current_global)]
        for local in local_weights
    ]


def compute_update_space_arrays(
    *,
    local_weights: List[NDArrays],
    local_updates: List[NDArrays],
    ema_updates: List[NDArrays],
) -> UpdateSpaceArrays:
    flat_deltas = [flatten_weights(update) for update in local_updates]
    flat_ema_deltas = [flatten_weights(update) for update in ema_updates]
    flat_weights = [flatten_weights(weights) for weights in local_weights]
    return UpdateSpaceArrays(
        flat_deltas=flat_deltas,
        flat_delta_matrix=np.stack(flat_deltas, axis=0),
        delta_norms=_norms(flat_deltas),
        flat_ema_deltas=flat_ema_deltas,
        ema_delta_norms=_norms(flat_ema_deltas),
        flat_weights=flat_weights,
        weight_norms=_norms(flat_weights),
    )


def _norms(vectors: list[np.ndarray]) -> np.ndarray:
    return np.array([float(np.linalg.norm(vector)) for vector in vectors])


__all__ = [
    "UpdateSpaceArrays",
    "compute_local_updates",
    "compute_update_space_arrays",
]
