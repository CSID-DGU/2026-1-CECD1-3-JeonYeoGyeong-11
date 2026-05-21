"""Projection helpers for GraphFL strategy vectors."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from graphfl_lab.projection import make_gaussian_projection


def project_with_cached_matrix(
    vec: np.ndarray,
    *,
    projection_matrix: Optional[np.ndarray],
    compression_dim: int,
    compression_seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if vec.size <= int(compression_dim):
        return vec.astype(np.float32, copy=False), projection_matrix

    matrix = projection_matrix
    if matrix is None:
        matrix = make_gaussian_projection(
            n_features=int(vec.size),
            n_dim=int(compression_dim),
            seed=int(compression_seed),
        )
    projected = vec.astype(np.float32, copy=False) @ matrix
    return projected, matrix


__all__ = ["project_with_cached_matrix"]
