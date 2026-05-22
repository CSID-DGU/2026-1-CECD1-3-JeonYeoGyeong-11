"""Distance and magnitude helpers for client graph similarities."""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(z_mat: np.ndarray) -> np.ndarray:
    z = z_mat.astype(np.float64, copy=False)
    sq_norms = np.sum(z * z, axis=1)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (z @ z.T)
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def positive_upper_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    if n <= 1:
        return np.array([], dtype=np.float64)
    vals = mat[np.triu_indices(n, k=1)]
    return vals[vals > 1e-12]


def resolve_distance_sigma(d2: np.ndarray, sigma: float) -> float:
    if float(sigma) > 0.0:
        return max(float(sigma), 1e-12)
    vals = np.sqrt(positive_upper_values(d2))
    if vals.size == 0:
        return 1.0
    return max(float(np.median(vals)), 1e-12)


__all__ = [
    "pairwise_sq_dists",
    "positive_upper_values",
    "resolve_distance_sigma",
]
