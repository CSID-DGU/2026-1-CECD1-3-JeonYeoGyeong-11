"""Rules that turn dense pairwise scores into graph edges."""

from __future__ import annotations

import numpy as np


def keep_topk(base: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k positive neighbors per row, then symmetrize by max."""
    n = base.shape[0]
    w = np.zeros_like(base)
    if k <= 0 or n <= 1:
        return w
    k_eff = min(int(k), n - 1)
    for i in range(n):
        sims = base[i].copy()
        sims[i] = -np.inf
        order = np.argsort(sims)[::-1]
        kept = 0
        for j in order:
            if kept >= k_eff:
                break
            v = float(base[i, j])
            if v <= 0.0:
                break
            w[i, j] = max(w[i, j], v)
            w[j, i] = max(w[j, i], v)
            kept += 1
    np.fill_diagonal(w, 0.0)
    return w


def directed_topk_mask(base: np.ndarray, k: int) -> np.ndarray:
    """Return a directed mask for positive top-k neighbors per row."""
    n = base.shape[0]
    keep = np.zeros_like(base, dtype=bool)
    if k <= 0 or n <= 1:
        return keep
    k_eff = min(int(k), n - 1)
    for i in range(n):
        sims = base[i].copy()
        sims[i] = -np.inf
        order = np.argsort(sims)[::-1]
        kept = 0
        for j in order:
            if kept >= k_eff:
                break
            if float(base[i, j]) <= 0.0:
                break
            keep[i, j] = True
            kept += 1
    return keep


def keep_mutual_topk(base: np.ndarray, k: int) -> np.ndarray:
    """Keep edge i-j only when both endpoints choose each other in top-k."""
    keep = directed_topk_mask(base, k)
    mutual = keep & keep.T
    w = np.where(mutual, base, 0.0)
    np.fill_diagonal(w, 0.0)
    return w


def keep_threshold(base: np.ndarray, theta: float) -> np.ndarray:
    """Keep only edges whose similarity is greater than theta."""
    w = np.where(base > theta, base, 0.0)
    np.fill_diagonal(w, 0.0)
    w = 0.5 * (w + w.T)
    return w


def uniform_graph(n: int) -> np.ndarray:
    """All off-diagonal entries are 1, diagonal is 0."""
    w = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    return w


def random_edges_with_edge_count(
    n: int, num_edges: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform random simple graph with exactly num_edges undirected edges."""
    w = np.zeros((n, n), dtype=np.float64)
    if n <= 1 or num_edges <= 0:
        return w
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ne = min(int(num_edges), len(pairs))
    rng.shuffle(pairs)
    for i, j in pairs[:ne]:
        w[i, j] = 1.0
        w[j, i] = 1.0
    return w


def random_edges_matched_to_knn(
    base: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Random binary graph with the same undirected edge count as kNN(base, k)."""
    n = base.shape[0]
    w_knn = keep_topk(base, k)
    iu = np.triu_indices(n, k=1)
    n_edges = int(np.sum(w_knn[iu] > 0.0))
    return random_edges_with_edge_count(n, n_edges, rng)
