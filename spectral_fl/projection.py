"""Projection helpers for client update vectors."""

from __future__ import annotations

import numpy as np
from flwr.common import NDArrays


def flatten_weights(weights: NDArrays) -> np.ndarray:
    """Flatten a Flower parameter/update list into one vector."""
    return np.concatenate([w.reshape(-1) for w in weights], axis=0)


def unflatten_like(flat: np.ndarray, template: NDArrays) -> NDArrays:
    """Split a flat vector into NDArrays with shapes/dtypes matching template."""
    out: NDArrays = []
    offset = 0
    for arr in template:
        size = int(arr.size)
        chunk = flat[offset : offset + size].reshape(arr.shape)
        out.append(chunk.astype(arr.dtype, copy=False))
        offset += size
    if offset != int(flat.size):
        raise ValueError("Flat vector size does not match template shapes")
    return out


def make_gaussian_projection(n_features: int, n_dim: int, seed: int) -> np.ndarray:
    """Create a fixed JL-style Gaussian projection matrix."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n_features, n_dim)).astype(np.float32)
    r /= np.sqrt(np.float32(n_dim))
    return r
