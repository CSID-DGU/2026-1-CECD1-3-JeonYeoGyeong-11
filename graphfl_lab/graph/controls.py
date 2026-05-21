"""Control-graph builders used for graph-gain diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _symmetrize_zero_diag(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float64)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


def build_identity_graph(num_clients: int) -> np.ndarray:
    """Return a self-only control under zero-diagonal adjacency convention."""
    n = max(int(num_clients), 0)
    return np.zeros((n, n), dtype=np.float64)


def build_uniform_control_graph(reference_adj: np.ndarray) -> np.ndarray:
    """Keep topology and replace all positive edges with one uniform weight."""
    ref = _symmetrize_zero_diag(reference_adj)
    mask = ref > 0.0
    out = np.zeros_like(ref, dtype=np.float64)
    if bool(np.any(mask)):
        fill = float(np.mean(ref[mask]))
        out[mask] = fill
    np.fill_diagonal(out, 0.0)
    return out


def build_shuffled_graph(
    reference_adj: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Shuffle client identity while preserving full graph structure/weights."""
    ref = _symmetrize_zero_diag(reference_adj)
    n = int(ref.shape[0])
    if n <= 1:
        return ref
    if rng is None:
        rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    out = ref[np.ix_(perm, perm)]
    np.fill_diagonal(out, 0.0)
    return out


def build_random_matched_graph(
    reference_adj: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample random edges with matched edge count and matched weight multiset."""
    ref = _symmetrize_zero_diag(reference_adj)
    n = int(ref.shape[0])
    if n <= 1:
        return ref
    if rng is None:
        rng = np.random.default_rng(0)

    iu = np.triu_indices(n, k=1)
    upper = ref[iu]
    weights = upper[upper > 0.0]
    edge_count = int(weights.size)
    if edge_count <= 0:
        return np.zeros_like(ref, dtype=np.float64)

    all_indices = np.arange(upper.size, dtype=np.int64)
    pick = rng.choice(all_indices, size=edge_count, replace=False)
    shuffled_weights = np.array(weights, copy=True)
    rng.shuffle(shuffled_weights)

    out_upper = np.zeros_like(upper, dtype=np.float64)
    out_upper[pick] = shuffled_weights
    out = np.zeros_like(ref, dtype=np.float64)
    out[iu] = out_upper
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def build_control_graph(
    *,
    reference_adj: np.ndarray,
    control_mode: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Dispatch control-graph construction by mode."""
    key = str(control_mode).strip().lower().replace("-", "_")
    if key == "identity":
        return build_identity_graph(reference_adj.shape[0])
    if key == "uniform":
        return build_uniform_control_graph(reference_adj)
    if key == "shuffled":
        return build_shuffled_graph(reference_adj, rng=rng)
    if key == "random":
        return build_random_matched_graph(reference_adj, rng=rng)
    raise ValueError(f"Unknown control graph mode: {control_mode}")


__all__ = [
    "build_control_graph",
    "build_identity_graph",
    "build_random_matched_graph",
    "build_shuffled_graph",
    "build_uniform_control_graph",
]
